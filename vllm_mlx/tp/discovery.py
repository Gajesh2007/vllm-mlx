"""Auto-discovery of tensor parallelism peers on Thunderbolt 5.

Detects:
1. Active RDMA devices via librdma.dylib (ibverbs API)
2. TB5 bridge interfaces with link-local IPs
3. Peer machines via UDP broadcast on link-local

Negotiation: Higher memory bandwidth = rank 0 (gets the larger weight share).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import os
import re
import socket
import struct
import subprocess
import time
from dataclasses import dataclass

from vllm_mlx.tp.config import TPConfig, detect_chip_bandwidth

logger = logging.getLogger("vllm_mlx.tp")

_DISCOVERY_PORT = 19980
_DISCOVERY_MAGIC = b"EIGEN-TP-DISC-V1"


@dataclass
class PeerInfo:
    """Information about a discovered TP peer."""

    ip: str
    rdma_device: str
    bandwidth_gbps: float
    memory_gb: float
    hostname: str


@dataclass
class LocalInfo:
    """Information about this machine's TP capabilities."""

    rdma_devices: list[str]
    bridge_ips: dict[str, str]  # interface → IP
    bandwidth_gbps: float
    memory_gb: float
    hostname: str


def discover_local() -> LocalInfo:
    """Gather this machine's TP-relevant hardware info."""
    rdma = find_active_rdma_devices()
    bridges = find_tb5_bridge_ips()
    bandwidth = detect_chip_bandwidth()
    memory = _get_total_memory_gb()
    hostname = socket.gethostname()

    logger.info(
        f"Local: {hostname}, {bandwidth:.0f} GB/s, {memory:.0f} GB, "
        f"RDMA devices: {rdma}, TB bridges: {bridges}"
    )
    return LocalInfo(
        rdma_devices=rdma,
        bridge_ips=bridges,
        bandwidth_gbps=bandwidth,
        memory_gb=memory,
        hostname=hostname,
    )


def find_active_rdma_devices() -> list[str]:
    """Find active RDMA devices using librdma.dylib.

    Uses ctypes to call ibverbs functions directly:
    - ibv_get_device_list() → list of device pointers
    - ibv_get_device_name() → device name (e.g., "rdma_en3")
    - ibv_open_device() → device context
    - ibv_query_port() → port state (4 = ACTIVE)

    Returns list of active device names (e.g., ["rdma_en3"]).
    """
    try:
        lib = ctypes.CDLL("/usr/lib/librdma.dylib")
    except OSError:
        logger.warning("librdma.dylib not found — RDMA not available")
        return []

    class IbvDevice(ctypes.Structure):
        _fields_ = [("_opaque", ctypes.c_char * 256)]

    class IbvPortAttr(ctypes.Structure):
        _fields_ = [
            ("state", ctypes.c_uint),
            ("_pad1", ctypes.c_uint),
            ("_pad2", ctypes.c_uint),
            ("_pad3", ctypes.c_int),
            ("_pad4", ctypes.c_uint32 * 3),
            ("_pad5", ctypes.c_uint16 * 3),
            ("_pad6", ctypes.c_uint8 * 5),
        ]

    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.POINTER(IbvDevice))
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_get_device_name.argtypes = [ctypes.POINTER(IbvDevice)]
    lib.ibv_open_device.restype = ctypes.c_void_p
    lib.ibv_open_device.argtypes = [ctypes.POINTER(IbvDevice)]
    lib.ibv_query_port.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.POINTER(IbvPortAttr),
    ]

    num_devices = ctypes.c_int(0)
    devices = lib.ibv_get_device_list(ctypes.byref(num_devices))

    active: list[str] = []
    for i in range(num_devices.value):
        name = lib.ibv_get_device_name(devices[i]).decode()
        ctx = lib.ibv_open_device(devices[i])
        if not ctx:
            continue
        port_attr = IbvPortAttr()
        lib.ibv_query_port(ctx, 1, ctypes.byref(port_attr))
        if port_attr.state == 4:  # IBV_PORT_ACTIVE
            active.append(name)

    logger.debug(f"RDMA devices: {num_devices.value} total, {len(active)} active: {active}")
    return active


def find_tb5_bridge_ips() -> dict[str, str]:
    """Find Thunderbolt bridge interfaces with link-local IPs.

    Scans en1, en2, en3 for 169.254.x.x addresses (auto-assigned by macOS
    when a TB cable is connected).

    Returns dict: interface → IP address.
    """
    result: dict[str, str] = {}
    try:
        output = subprocess.check_output(["ifconfig"], text=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return result

    current_iface = None
    for line in output.split("\n"):
        # Interface header: "en3: flags=..."
        iface_match = re.match(r"^(\w+):", line)
        if iface_match:
            current_iface = iface_match.group(1)

        # Only check en1-en9 (TB bridge interfaces)
        if current_iface and current_iface.startswith("en") and current_iface[2:].isdigit():
            iface_num = int(current_iface[2:])
            if 1 <= iface_num <= 9:
                # Look for link-local IPv4
                ip_match = re.search(r"inet (169\.254\.\d+\.\d+)", line)
                if ip_match:
                    result[current_iface] = ip_match.group(1)

    return result


def discover_peer(local: LocalInfo, timeout: float = 15.0) -> PeerInfo | None:
    """Discover a TP peer via UDP broadcast on TB5 link-local interfaces.

    Both machines broadcast their info. When a response is received,
    we have a peer. Higher bandwidth machine becomes rank 0.

    Returns PeerInfo if found, None on timeout.
    """
    if not local.bridge_ips:
        logger.warning("No TB5 bridge IPs found — cannot discover peer")
        return None

    # Build discovery payload
    payload = json.dumps({
        "magic": _DISCOVERY_MAGIC.decode(),
        "hostname": local.hostname,
        "bandwidth": local.bandwidth_gbps,
        "memory": local.memory_gb,
        "rdma_devices": local.rdma_devices,
    }).encode()

    # Broadcast on all TB bridge interfaces
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)

    try:
        sock.bind(("0.0.0.0", _DISCOVERY_PORT))
    except OSError:
        # Port in use — try alternate
        sock.bind(("0.0.0.0", _DISCOVERY_PORT + 1))

    deadline = time.monotonic() + timeout
    broadcast_interval = 1.0
    last_broadcast = 0.0

    while time.monotonic() < deadline:
        now = time.monotonic()

        # Broadcast periodically
        if now - last_broadcast >= broadcast_interval:
            for iface, ip in local.bridge_ips.items():
                # Broadcast to 169.254.255.255 on the link-local subnet
                try:
                    sock.sendto(payload, ("169.254.255.255", _DISCOVERY_PORT))
                except OSError:
                    pass
            last_broadcast = now

        # Listen for peer
        try:
            data, addr = sock.recvfrom(4096)
            try:
                msg = json.loads(data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if msg.get("magic") != _DISCOVERY_MAGIC.decode():
                continue
            if msg.get("hostname") == local.hostname:
                continue  # Ignore our own broadcast

            peer_ip = addr[0]
            peer = PeerInfo(
                ip=peer_ip,
                rdma_device=msg.get("rdma_devices", [""])[0],
                bandwidth_gbps=msg.get("bandwidth", 273.0),
                memory_gb=msg.get("memory", 24.0),
                hostname=msg.get("hostname", "unknown"),
            )
            logger.info(f"Discovered peer: {peer.hostname} at {peer.ip} ({peer.bandwidth_gbps} GB/s)")
            sock.close()
            return peer

        except socket.timeout:
            continue

    sock.close()
    logger.warning(f"Peer discovery timed out after {timeout}s")
    return None


def negotiate_config(
    local: LocalInfo,
    peer: PeerInfo,
    ratio: float | None = None,
    backend: str = "jaccl",
    encrypt: bool = True,
    worker_port: int = 8001,
) -> TPConfig:
    """Build TPConfig from discovered local + peer info.

    Higher bandwidth = rank 0 (gets the larger weight share).
    Ties broken by memory.
    """
    # Determine rank
    if local.bandwidth_gbps > peer.bandwidth_gbps:
        rank = 0
    elif local.bandwidth_gbps < peer.bandwidth_gbps:
        rank = 1
    elif local.memory_gb >= peer.memory_gb:
        rank = 0
    else:
        rank = 1

    # Find matching RDMA devices
    local_rdma = local.rdma_devices[0] if local.rdma_devices else ""
    remote_rdma = peer.rdma_device

    # Auto-compute ratio if not specified
    if ratio is None:
        from vllm_mlx.tp.ratio import compute_optimal_ratio

        if rank == 0:
            ratio = compute_optimal_ratio(local.bandwidth_gbps, peer.bandwidth_gbps)
        else:
            ratio = compute_optimal_ratio(peer.bandwidth_gbps, local.bandwidth_gbps)

    # Coordinator address: rank 0's IP on the TB bridge
    if rank == 0:
        coordinator = f"{list(local.bridge_ips.values())[0]}:{_DISCOVERY_PORT + 10}"
    else:
        coordinator = f"{peer.ip}:{_DISCOVERY_PORT + 10}"

    config = TPConfig(
        ratio=ratio,
        rank=rank,
        peer_address=coordinator,
        local_rdma=local_rdma,
        remote_rdma=remote_rdma,
        backend=backend,
        encrypt=encrypt,
        worker_port=worker_port,
    )

    logger.info(
        f"Negotiated: rank={rank}, ratio={ratio:.4f}, "
        f"local_rdma={local_rdma}, remote_rdma={remote_rdma}"
    )
    return config


def check_rdma_prerequisites() -> list[str]:
    """Check RDMA prerequisites and return list of issues (empty = all good)."""
    issues: list[str] = []

    # Check rdma_ctl status
    try:
        result = subprocess.run(
            ["rdma_ctl", "status"], capture_output=True, text=True, timeout=5
        )
        if "enabled" not in result.stdout.lower():
            issues.append("RDMA not enabled. Run: sudo rdma_ctl enable")
    except FileNotFoundError:
        issues.append("rdma_ctl not found — RDMA not available on this system")
    except subprocess.TimeoutExpired:
        issues.append("rdma_ctl timed out")

    # Check bridge0 existence
    try:
        output = subprocess.check_output(["ifconfig", "bridge0"], text=True, timeout=5)
        if "inet" in output or "flags" in output:
            issues.append(
                "bridge0 (Thunderbolt Bridge) exists and breaks RDMA. "
                "Destroy it: sudo ifconfig bridge0 destroy"
            )
    except subprocess.CalledProcessError:
        pass  # bridge0 doesn't exist — good

    return issues


def _get_total_memory_gb() -> float:
    """Get total physical RAM in GB."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5)
        return int(out.strip()) / (1024**3)
    except Exception:
        return 0.0
