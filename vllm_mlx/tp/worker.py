"""Tensor parallelism worker (rank 1).

Receives prompts from rank 0 via TCP side channel, then calls
mlx_lm.generate() with the same model. The sharded model's all_sum
operations synchronize both ranks' forward passes over RDMA.

Key design: TCP for control, RDMA for compute. Never mix them.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import threading
import time

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.tp.config import TPConfig
from vllm_mlx.tp.sync_channel import SyncClient
from vllm_mlx.tp.watchdog import GPUWatchdog, MemoryMonitor

logger = logging.getLogger("vllm_mlx.tp")


class TPBatchWorker:
    """Rank 1 worker: mirrors rank 0's generate() calls.

    Lifecycle:
    1. __init__: Model loaded + sharded, distributed group ready
    2. run(): Connect to rank 0 via TCP, enter generate loop
    3. shutdown(): Clean exit with os._exit(0)

    The worker calls mlx_lm.generate() for each prompt received from
    rank 0. The all_sum operations in the sharded layers synchronize
    both ranks automatically — no explicit coordination during inference.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        group: mx.distributed.Group,
        tp_config: TPConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.group = group
        self.tp_config = tp_config

        self.watchdog = GPUWatchdog(timeout=180)
        self.memory_monitor = MemoryMonitor()

        logger.info(f"TPBatchWorker initialized: rank={tp_config.rank}")

    def run(self) -> None:
        """Main worker loop. Blocks until shutdown."""
        # Set up signal handlers
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())

        # Start health endpoint
        self._start_health_server()

        # Connect to rank 0's sync channel via TCP
        peer_ip = self.tp_config.peer_address.split(":")[0]
        sync_port = int(self.tp_config.peer_address.split(":")[-1]) + 100
        logger.info(f"Connecting to rank 0 sync channel at {peer_ip}:{sync_port}")
        self._sync = SyncClient(peer_ip, sync_port)
        logger.info("Sync channel connected")

        logger.info("TPBatchWorker entering main loop")

        try:
            self._main_loop()
        except Exception as e:
            logger.error(f"TPBatchWorker error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        """Receive prompts from rank 0 via TCP, call generate() in lockstep."""
        from mlx_lm import generate as mlx_generate

        while True:
            self.watchdog.heartbeat()

            # Wait for next message from rank 0 (blocks on TCP recv)
            try:
                msg = self._sync.recv_message()
            except ConnectionError:
                logger.info("Sync channel closed — rank 0 disconnected")
                break

            msg_type = msg.get("type")

            if msg_type == "shutdown":
                logger.info("Received shutdown signal")
                break

            elif msg_type == "generate":
                prompt = msg["prompt"]
                max_tokens = msg.get("max_tokens", 256)
                temperature = msg.get("temperature", 0.0)
                top_p = msg.get("top_p", 1.0)

                logger.debug(
                    f"Generating: max_tokens={max_tokens}, "
                    f"prompt_len={len(prompt)}"
                )

                # Create sampler matching rank 0's params
                from mlx_lm.utils import make_sampler

                sampler = make_sampler(temperature, top_p)

                # Call generate — the model's all_sum ops sync with rank 0
                try:
                    _output = mlx_generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        verbose=False,
                    )
                    self.watchdog.heartbeat()
                    logger.debug(f"Generate complete")
                except Exception as e:
                    logger.error(f"Generate failed: {e}", exc_info=True)
                    # Don't crash — try to stay alive for next request

    def _start_health_server(self) -> None:
        """Minimal HTTP health endpoint."""

        def _serve() -> None:
            import http.server

            class Handler(http.server.BaseHTTPRequestHandler):
                worker = self

                def do_GET(self_h) -> None:
                    if self_h.path == "/health":
                        avail, total = self.memory_monitor.check()
                        body = json.dumps({
                            "status": "ok",
                            "rank": self.tp_config.rank,
                            "model_loaded": True,
                            "memory_available_gb": round(avail, 1),
                            "memory_total_gb": round(total, 1),
                        }).encode()
                        self_h.send_response(200)
                        self_h.send_header("Content-Type", "application/json")
                        self_h.end_headers()
                        self_h.wfile.write(body)
                    else:
                        self_h.send_error(404)

                def log_message(self_h, *args) -> None:
                    pass

            server = http.server.HTTPServer(
                ("0.0.0.0", self.tp_config.worker_port), Handler
            )
            server.serve_forever()

        t = threading.Thread(target=_serve, daemon=True, name="tp-health")
        t.start()
        logger.info(f"Health endpoint on port {self.tp_config.worker_port}")

    def _shutdown(self) -> None:
        logger.info("TPBatchWorker shutting down")
        self.watchdog.stop()
        os._exit(0)
