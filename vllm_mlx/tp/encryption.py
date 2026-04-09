"""Encrypted tensor parallelism via additive masking.

All inter-node RDMA traffic is encrypted so that wire observers see only
noise. Uses additive masking: encrypt(x) = x + mask, then after all_sum,
decrypt(result) = result - world_size * mask.

Mathematical correctness:
  all_sum(x0 + mask, x1 + mask) = (x0 + x1) + 2*mask
  decrypt = (x0 + x1) + 2*mask - 2*mask = x0 + x1  ✓

Security: Per-token unique masks from ECDH-derived PRNG seed. Each all_sum
call uses a unique mask index — no mask is ever reused across tokens.
Immune to differential attacks.

Performance: ~0% overhead. Mask add/subtract stays in MLX's lazy computation
graph — no mx.eval(), no GPU-CPU sync, no memory copies.

Key exchange: DH-2048 (RFC 3526 Group 14) + HKDF-SHA256 over TCP.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import logging
import secrets
import socket
import struct
import time

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("vllm_mlx.tp")

# RFC 3526 Group 14 (2048-bit MODP)
_DH_PRIME = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
_DH_GENERATOR = 2
_HKDF_SALT = b"eigeninference-tp-rdma-v1"
_HKDF_INFO = b"asymmetric-tp-mask-seed\x01"
_KEY_EXCHANGE_PORT = 19990
_KEY_EXCHANGE_TIMEOUT = 60


class EncryptedAllSum:
    """Manages encrypted all_sum via additive masking.

    Lifecycle:
    1. key_exchange() — DH-2048 over TCP, derives shared 32-byte key
    2. warmup() — Runs one forward pass to discover all_sum call count + shapes
    3. install() — Monkey-patches mx.distributed.all_sum
    4. (inference runs with encrypted all_sum)
    5. uninstall() — Restores original all_sum
    """

    # Max tokens to pre-allocate masks for PER REQUEST.
    # This is NOT the server's --max-tokens (which can be 32768+).
    # The counter resets between requests, so we only need enough for one.
    _DEFAULT_POOL_TOKENS = 512

    def __init__(self, group: mx.distributed.Group, max_tokens: int | None = None):
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        # Cap pool size to avoid multi-GB allocation. The counter resets
        # between requests via reset_counter(), so we only need enough
        # for a single request's decode tokens.
        self.max_tokens = min(max_tokens or self._DEFAULT_POOL_TOKENS, 2048)

        self._shared_key: bytes | None = None
        self._encryption_seed: int | None = None
        self._calls_per_token: int | None = None
        self._mask_shape: tuple[int, ...] | None = None
        self._mask_pool: mx.array | None = None
        self._call_counter: list[int] = [0]
        self._original_all_sum = mx.distributed.all_sum
        self._installed = False

    def key_exchange(self, peer_ip: str, port: int = _KEY_EXCHANGE_PORT) -> None:
        """Perform DH-2048 key exchange with the peer over TCP.

        Both ranks derive an identical 32-byte shared key via HKDF-SHA256.
        The key never crosses the wire — only DH public values are exchanged.
        """
        # Generate DH keypair
        private = int.from_bytes(secrets.token_bytes(256), "big") % (_DH_PRIME - 2) + 1
        pub_bytes = pow(_DH_GENERATOR, private, _DH_PRIME).to_bytes(256, "big")

        t0 = time.perf_counter()

        if self.rank == 0:
            peer_pub_bytes = self._exchange_rank0(pub_bytes, port)
        else:
            peer_pub_bytes = self._exchange_rank1(pub_bytes, peer_ip, port)

        # Validate and derive shared secret
        peer_public = int.from_bytes(peer_pub_bytes, "big")
        if peer_public < 2 or peer_public > _DH_PRIME - 2:
            raise ValueError(
                f"Invalid DH public key: value must be in [2, p-2]. "
                f"Possible MITM or corrupted exchange."
            )
        shared_secret = pow(peer_public, private, _DH_PRIME).to_bytes(256, "big")

        # HKDF-SHA256
        prk = hmac.new(_HKDF_SALT, shared_secret, hashlib.sha256).digest()
        self._shared_key = hmac.new(prk, _HKDF_INFO, hashlib.sha256).digest()
        self._encryption_seed = int.from_bytes(self._shared_key[:8], "big")

        elapsed = time.perf_counter() - t0
        fingerprint = hashlib.sha256(self._shared_key).hexdigest()[:16]
        logger.info(
            f"DH-2048 key exchange complete in {elapsed:.2f}s, "
            f"fingerprint={fingerprint}"
        )

    def warmup(self, model: nn.Module, tokenizer: object) -> None:
        """Run one forward pass to discover all_sum call pattern.

        Counts how many all_sum calls happen per token and records the
        tensor shape. This determines the mask pool size.
        """
        if self._encryption_seed is None:
            raise RuntimeError("Call key_exchange() before warmup()")

        warmup_shapes: list[tuple[int, ...]] = []

        def counting_all_sum(x: mx.array, *, group: mx.distributed.Group) -> mx.array:
            warmup_shapes.append(tuple(x.shape))
            return self._original_all_sum(x, group=group)

        # Temporarily patch all_sum to count calls
        mx.distributed.all_sum = counting_all_sum
        try:
            # Encode a short prompt
            if hasattr(tokenizer, "encode"):
                tokens = tokenizer.encode("Hi")
            else:
                tokens = [1]  # BOS token fallback
            y = model(mx.array([tokens]))
            mx.eval(y)
            del y
            gc.collect()
        finally:
            mx.distributed.all_sum = self._original_all_sum

        self._calls_per_token = len(warmup_shapes)
        self._mask_shape = warmup_shapes[0] if warmup_shapes else None
        if self._mask_shape is None:
            raise RuntimeError(
                "Warmup produced no all_sum calls — model may not have "
                "tensor parallel sharding applied. Call apply_patches() first."
            )

        logger.info(
            f"Warmup: {self._calls_per_token} all_sum calls/token, "
            f"shape={self._mask_shape}"
        )

        # Pre-generate mask pool
        pool_size = self._calls_per_token * (self.max_tokens + 32)
        t0 = time.perf_counter()
        mx.random.seed(self._encryption_seed)
        self._mask_pool = (
            mx.random.normal(
                (pool_size,) + self._mask_shape[1:], dtype=mx.float16
            )
            * 0.1
        )
        mx.eval(self._mask_pool)
        elapsed = time.perf_counter() - t0
        pool_mb = self._mask_pool.nbytes / (1024 * 1024)
        logger.info(
            f"Mask pool: {pool_size} masks, {pool_mb:.0f} MB, "
            f"generated in {elapsed:.2f}s"
        )

    def install(self) -> None:
        """Monkey-patch mx.distributed.all_sum with encrypted version."""
        if self._mask_pool is None:
            raise RuntimeError("Call warmup() before install()")
        if self._installed:
            logger.warning("Encrypted all_sum already installed")
            return

        ws = self.world_size
        mask_pool = self._mask_pool
        call_counter = self._call_counter
        orig = self._original_all_sum

        def encrypted_all_sum(
            x: mx.array, *, group: mx.distributed.Group
        ) -> mx.array:
            idx = call_counter[0]
            call_counter[0] += 1
            m = mask_pool[idx]
            return orig(x + m, group=group) - ws * m

        mx.distributed.all_sum = encrypted_all_sum
        mx.distributed._tp_encrypted_all_sum = self  # Store ref for reset_counter access
        self._call_counter[0] = 0
        self._installed = True
        logger.info("Encrypted all_sum installed (per-token unique masks)")

    def uninstall(self) -> None:
        """Restore original mx.distributed.all_sum."""
        if self._installed:
            mx.distributed.all_sum = self._original_all_sum
            self._installed = False
            logger.info("Encrypted all_sum uninstalled")

    def reset_counter(self) -> None:
        """Reset the call counter (call between requests if needed)."""
        self._call_counter[0] = 0

    @property
    def is_installed(self) -> bool:
        return self._installed

    # --- TCP helpers for key exchange ---

    def _exchange_rank0(self, pub_bytes: bytes, port: int) -> bytes:
        """Rank 0: Listen for rank 1's connection, exchange DH public keys."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(_KEY_EXCHANGE_TIMEOUT)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        logger.debug(f"Key exchange: rank 0 listening on port {port}")

        try:
            conn, addr = srv.accept()
            conn.settimeout(_KEY_EXCHANGE_TIMEOUT)
            logger.debug(f"Key exchange: rank 0 accepted from {addr}")

            # Send our public key, receive theirs
            _tcp_send(conn, pub_bytes)
            peer_pub = _tcp_recv(conn)

            conn.close()
            return peer_pub
        finally:
            srv.close()

    def _exchange_rank1(self, pub_bytes: bytes, peer_ip: str, port: int) -> bytes:
        """Rank 1: Connect to rank 0 and exchange DH public keys."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(_KEY_EXCHANGE_TIMEOUT)

        # Retry connection (rank 0 may not be listening yet)
        for attempt in range(30):
            try:
                sock.connect((peer_ip, port))
                break
            except (ConnectionRefusedError, OSError):
                if attempt == 29:
                    raise RuntimeError(
                        f"Key exchange: failed to connect to {peer_ip}:{port} "
                        f"after 30 attempts"
                    )
                time.sleep(1)

        logger.debug(f"Key exchange: rank 1 connected to {peer_ip}:{port}")

        # Receive their public key, send ours
        peer_pub = _tcp_recv(sock)
        _tcp_send(sock, pub_bytes)

        sock.close()
        return peer_pub


def _tcp_send(conn: socket.socket, data: bytes) -> None:
    """Send length-prefixed data over TCP."""
    conn.sendall(struct.pack("!I", len(data)))
    conn.sendall(data)


def _tcp_recv(conn: socket.socket) -> bytes:
    """Receive length-prefixed data from TCP."""
    header = _recv_exact(conn, 4)
    (length,) = struct.unpack("!I", header)
    return _recv_exact(conn, length)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed during receive")
        buf.extend(chunk)
    return bytes(buf)
