"""MLX distributed backend initialization for tensor parallelism.

Supports two backends:
- JACCL (default): RDMA over Thunderbolt 5 via ibverbs. 68 Gbps actual throughput.
- Ring (fallback): TCP sockets over TB5 bridge. ~40 Gbps. Simpler setup.

Both use MLX's mx.distributed.init() with environment variables for configuration.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import mlx.core as mx

from vllm_mlx.tp.config import TPConfig

logger = logging.getLogger("vllm_mlx.tp")


def init_distributed(tp_config: TPConfig) -> mx.distributed.Group:
    """Initialize MLX distributed backend.

    Must be called AFTER model download (HuggingFace deadlocks if download
    happens after distributed init).

    Sets required environment variables and calls mx.distributed.init().
    Runs a sanity check to verify the group is functional.

    Returns the distributed Group for use in all_sum operations.
    """
    # Required for all backends
    os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if tp_config.backend == "jaccl":
        group = _init_jaccl(tp_config)
    elif tp_config.backend == "ring":
        group = _init_ring(tp_config)
    else:
        raise ValueError(f"Unknown backend: {tp_config.backend!r}")

    # Sanity check
    _verify_group(group, tp_config)

    return group


def _init_jaccl(tp_config: TPConfig) -> mx.distributed.Group:
    """Initialize JACCL (RDMA/ibverbs) backend.

    Requires:
    - RDMA enabled on both machines (rdma_ctl enable)
    - bridge0 destroyed on both machines
    - Active RDMA devices (rdma_enX) on connected TB5 ports

    Environment variables set:
    - MLX_IBV_DEVICES: Path to JSON file with 2x2 device matrix
    - MLX_RANK: This process's rank
    - MLX_JACCL_COORDINATOR: TCP address for initial QP setup
    """
    if not tp_config.local_rdma or not tp_config.remote_rdma:
        raise ValueError(
            "JACCL backend requires --tp-local-rdma and --tp-remote-rdma "
            "(e.g., rdma_en3). Use --tp-auto-discover or specify manually."
        )

    # Build device matrix: devices[i][j] = RDMA device for rank i → rank j
    # Diagonal is null (no self-communication)
    devices: list[list[str | None]] = [[None, None], [None, None]]
    if tp_config.rank == 0:
        devices[0][1] = tp_config.local_rdma
        devices[1][0] = tp_config.remote_rdma
    else:
        devices[0][1] = tp_config.remote_rdma
        devices[1][0] = tp_config.local_rdma

    # Write device matrix to temp file
    dev_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="mlx_ibv_", delete=False
    )
    json.dump(devices, dev_file)
    dev_file.close()

    os.environ["MLX_IBV_DEVICES"] = dev_file.name
    os.environ["MLX_RANK"] = str(tp_config.rank)
    os.environ["MLX_JACCL_COORDINATOR"] = tp_config.peer_address

    logger.info(
        f"Initializing JACCL backend: rank={tp_config.rank}, "
        f"local_rdma={tp_config.local_rdma}, remote_rdma={tp_config.remote_rdma}, "
        f"coordinator={tp_config.peer_address}"
    )

    group = mx.distributed.init(backend="jaccl", strict=True)
    logger.info(f"JACCL initialized: rank={group.rank()}, size={group.size()}")
    return group


def _init_ring(tp_config: TPConfig) -> mx.distributed.Group:
    """Initialize ring (TCP) backend.

    Simpler than JACCL — just needs IP:port for each rank.
    Uses MLX_HOSTFILE with a 2D JSON array of address strings.
    """
    if not tp_config.peer_address:
        raise ValueError("Ring backend requires --tp-peer (ip:port)")

    # Parse peer address
    if ":" not in tp_config.peer_address:
        raise ValueError(
            f"--tp-peer must be ip:port, got {tp_config.peer_address!r}"
        )
    peer_ip, peer_port_str = tp_config.peer_address.rsplit(":", 1)
    peer_port = int(peer_port_str)

    # For ring, each rank listens on its own address
    # Hostfile format: 2D array where each row is a rank
    if tp_config.rank == 0:
        local_addr = f"0.0.0.0:{peer_port}"
        hosts = [[local_addr], [tp_config.peer_address]]
    else:
        local_addr = f"0.0.0.0:{peer_port}"
        hosts = [[tp_config.peer_address], [local_addr]]

    # Write hostfile
    host_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="mlx_hosts_", delete=False
    )
    json.dump(hosts, host_file)
    host_file.close()

    os.environ["MLX_HOSTFILE"] = host_file.name
    os.environ["MLX_RANK"] = str(tp_config.rank)

    logger.info(
        f"Initializing ring backend: rank={tp_config.rank}, "
        f"peer={tp_config.peer_address}"
    )

    group = mx.distributed.init(backend="ring", strict=True)
    logger.info(f"Ring initialized: rank={group.rank()}, size={group.size()}")
    return group


def _verify_group(group: mx.distributed.Group, tp_config: TPConfig) -> None:
    """Verify distributed group is functional with a simple all_sum test."""
    rank_tensor = mx.array([float(group.rank())])
    result = mx.distributed.all_sum(rank_tensor, group=group)
    mx.eval(result)

    expected = sum(range(tp_config.world_size))
    actual = result.item()
    if abs(actual - expected) > 0.01:
        raise RuntimeError(
            f"Distributed sanity check failed: all_sum of ranks = {actual}, "
            f"expected {expected}. Check network connectivity."
        )

    logger.info(f"Distributed group verified: all_sum sanity check passed")
