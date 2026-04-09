"""Tensor parallelism worker process (rank 1).

Mirrors rank 0's BatchGenerator in lockstep. Receives control messages
from rank 0, applies the same batch operations, and participates in
forward passes via all_sum-synchronized model calls.

The worker does NOT run an HTTP server — it only participates in
distributed inference and exposes a health endpoint.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import struct
import socket
import threading
import time
from enum import IntEnum
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.tp.config import TPConfig
from vllm_mlx.tp.watchdog import GPUWatchdog, MemoryMonitor

logger = logging.getLogger("vllm_mlx.tp")


class TPSignal(IntEnum):
    """Control signals sent from rank 0 to rank 1."""

    SHUTDOWN = 0
    PREFILL = 1       # New request: followed by token count + tokens
    DECODE_STEP = 2   # One decode step: followed by batch_size + last tokens
    FILTER = 3        # Remove finished seqs: followed by keep_count + keep_indices
    RESET = 4         # Reset all state (new session)


class TPBatchWorker:
    """Rank 1 worker that mirrors rank 0's inference.

    Lifecycle:
    1. __init__: Model already loaded + sharded, distributed group ready
    2. run(): Main loop — receives signals, mirrors forward passes
    3. shutdown(): Clean exit with os._exit(0)

    The worker maintains its own KV cache and BatchGenerator state,
    kept in sync by processing the same tokens as rank 0.
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

        self.watchdog = GPUWatchdog(timeout=120)
        self.memory_monitor = MemoryMonitor()

        # KV cache — will be initialized on first prefill
        self._cache: list | None = None
        self._batch_size: int = 0

        # Control channel: TCP socket from rank 0
        self._control_sock: socket.socket | None = None

        # Health server
        self._health_thread: threading.Thread | None = None

        logger.info(f"TPBatchWorker initialized: rank={tp_config.rank}")

    def run(self) -> None:
        """Main worker loop. Blocks until shutdown signal."""
        # Start health endpoint
        self._start_health_server()

        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())

        logger.info("TPBatchWorker entering main loop")

        try:
            self._setup_control_channel()
            self._main_loop()
        except Exception as e:
            logger.error(f"TPBatchWorker error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        """Process control signals from rank 0."""
        while True:
            self.watchdog.heartbeat()

            # Receive signal from rank 0 via distributed
            try:
                sig_tensor = mx.distributed.recv(
                    shape=(1,), dtype=mx.int32, src=0, group=self.group
                )
                mx.eval(sig_tensor)
                sig = TPSignal(sig_tensor.item())
            except Exception as e:
                logger.error(f"Failed to receive signal: {e}")
                break

            if sig == TPSignal.SHUTDOWN:
                logger.info("Received SHUTDOWN signal")
                break

            elif sig == TPSignal.PREFILL:
                self._handle_prefill()

            elif sig == TPSignal.DECODE_STEP:
                self._handle_decode_step()

            elif sig == TPSignal.FILTER:
                self._handle_filter()

            elif sig == TPSignal.RESET:
                self._handle_reset()

    def _handle_prefill(self) -> None:
        """Receive prompt tokens and run prefill forward pass."""
        # Receive token count
        count_tensor = mx.distributed.recv(
            shape=(2,), dtype=mx.int32, src=0, group=self.group
        )
        mx.eval(count_tensor)
        batch_size = count_tensor[0].item()
        seq_len = count_tensor[1].item()

        # Receive prompt tokens
        tokens = mx.distributed.recv(
            shape=(batch_size, seq_len), dtype=mx.int32, src=0, group=self.group
        )
        mx.eval(tokens)

        # Create new cache if needed
        if self._cache is None:
            from mlx_lm.utils import make_prompt_cache

            self._cache = make_prompt_cache(self.model)

        # Run prefill forward pass (all_sum synchronizes with rank 0)
        logits = self.model(tokens, cache=self._cache)
        mx.eval(logits)
        self._batch_size = batch_size

        self.watchdog.heartbeat()
        logger.debug(f"Prefill: batch={batch_size}, seq_len={seq_len}")

    def _handle_decode_step(self) -> None:
        """Receive last tokens and run one decode step."""
        # Receive batch info
        info_tensor = mx.distributed.recv(
            shape=(1,), dtype=mx.int32, src=0, group=self.group
        )
        mx.eval(info_tensor)
        batch_size = info_tensor[0].item()

        # Receive last tokens
        tokens = mx.distributed.recv(
            shape=(batch_size, 1), dtype=mx.int32, src=0, group=self.group
        )
        mx.eval(tokens)

        # Run decode step (all_sum synchronizes with rank 0)
        logits = self.model(tokens, cache=self._cache)
        mx.eval(logits)

        self.watchdog.heartbeat()

    def _handle_filter(self) -> None:
        """Remove finished sequences from the KV cache."""
        # Receive keep count
        count_tensor = mx.distributed.recv(
            shape=(1,), dtype=mx.int32, src=0, group=self.group
        )
        mx.eval(count_tensor)
        keep_count = count_tensor[0].item()

        if keep_count == 0:
            # All sequences finished — reset cache
            self._cache = None
            self._batch_size = 0
            return

        # Receive keep indices
        indices = mx.distributed.recv(
            shape=(keep_count,), dtype=mx.int32, src=0, group=self.group
        )
        mx.eval(indices)
        keep_idx = indices.tolist()

        # Filter cache
        if self._cache is not None:
            for c in self._cache:
                if hasattr(c, "filter"):
                    c.filter(keep_idx)

        self._batch_size = keep_count
        logger.debug(f"Filtered: keeping {keep_count} sequences")

    def _handle_reset(self) -> None:
        """Reset all state."""
        self._cache = None
        self._batch_size = 0
        gc.collect()
        logger.info("State reset")

    def _setup_control_channel(self) -> None:
        """Set up TCP control channel for non-tensor data from rank 0."""
        # For now, all control is via mx.distributed.send/recv
        # TCP channel can be added for richer control messages later
        pass

    def _start_health_server(self) -> None:
        """Start a minimal HTTP health endpoint."""

        def _serve_health() -> None:
            import http.server

            class HealthHandler(http.server.BaseHTTPRequestHandler):
                worker_ref = self

                def do_GET(self_h) -> None:
                    if self_h.path == "/health":
                        avail, total = self.memory_monitor.check()
                        body = json.dumps({
                            "status": "ok",
                            "rank": self.tp_config.rank,
                            "model_loaded": True,
                            "batch_size": self._batch_size,
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
                    pass  # Suppress HTTP access logs

            server = http.server.HTTPServer(
                ("0.0.0.0", self.tp_config.worker_port), HealthHandler
            )
            server.serve_forever()

        self._health_thread = threading.Thread(
            target=_serve_health, daemon=True, name="tp-health"
        )
        self._health_thread.start()
        logger.info(f"Health endpoint started on port {self.tp_config.worker_port}")

    def _shutdown(self) -> None:
        """Clean exit."""
        logger.info("TPBatchWorker shutting down")
        self.watchdog.stop()
        os._exit(0)
