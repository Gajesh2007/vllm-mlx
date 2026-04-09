"""TCP side channel for TP control messages.

Separates control (prompts, stop signals) from compute (all_sum over RDMA).
The experiments proved this: TCP for coordination, RDMA only for tensor ops.
Mixing mx.distributed.send/recv with all_sum on the same channel causes
garbage data and GPU timeouts.

Rank 0 runs a TCP server. Rank 1 connects on startup.
Before each generate() call, rank 0 sends the prompt. Rank 1 receives it
and calls generate() too. all_sum in the sharded layers synchronizes
the forward passes automatically.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("vllm_mlx.tp")

_SYNC_PORT_OFFSET = 100  # Added to JACCL coordinator port


@dataclass
class GenerateRequest:
    """Prompt + params sent from rank 0 to rank 1 before each generate() call."""

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None


class SyncServer:
    """Rank 0's TCP server for sending control messages to rank 1."""

    def __init__(self, port: int):
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.listen(1)
        self._conn: socket.socket | None = None
        self._accepted = threading.Event()

        # Accept connection in background (rank 1 connects during startup)
        self._accept_thread = threading.Thread(
            target=self._accept, daemon=True, name="sync-accept"
        )
        self._accept_thread.start()
        logger.info(f"SyncServer listening on port {port}")

    def _accept(self) -> None:
        self._sock.settimeout(120)
        try:
            self._conn, addr = self._sock.accept()
            self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._accepted.set()
            logger.info(f"SyncServer: rank 1 connected from {addr}")
        except socket.timeout:
            logger.error("SyncServer: no connection from rank 1 within 120s")

    def wait_for_worker(self, timeout: float = 120) -> bool:
        """Block until rank 1 connects."""
        return self._accepted.wait(timeout)

    def send_generate(self, req: GenerateRequest) -> None:
        """Send a generate request to rank 1."""
        if self._conn is None:
            raise RuntimeError("Rank 1 not connected")
        data = json.dumps({
            "type": "generate",
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stop": req.stop,
        }).encode()
        self._conn.sendall(struct.pack("!I", len(data)))
        self._conn.sendall(data)

    def send_shutdown(self) -> None:
        """Tell rank 1 to shut down."""
        if self._conn is None:
            return
        data = json.dumps({"type": "shutdown"}).encode()
        try:
            self._conn.sendall(struct.pack("!I", len(data)))
            self._conn.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def close(self) -> None:
        if self._conn:
            self._conn.close()
        self._sock.close()


class SyncClient:
    """Rank 1's TCP client for receiving control messages from rank 0."""

    def __init__(self, server_ip: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Retry connection (rank 0 may not be listening yet)
        for attempt in range(60):
            try:
                self._sock.connect((server_ip, port))
                logger.info(f"SyncClient: connected to rank 0 at {server_ip}:{port}")
                return
            except (ConnectionRefusedError, OSError):
                if attempt == 59:
                    raise RuntimeError(
                        f"SyncClient: failed to connect to {server_ip}:{port} "
                        f"after 60 attempts"
                    )
                time.sleep(1)

    def recv_message(self) -> dict:
        """Receive a control message from rank 0. Blocks until message arrives."""
        header = self._recv_exact(4)
        (length,) = struct.unpack("!I", header)
        data = self._recv_exact(length)
        return json.loads(data.decode())

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Sync channel closed")
            buf.extend(chunk)
        return bytes(buf)

    def close(self) -> None:
        self._sock.close()
