"""Tensor parallelism configuration."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("vllm_mlx.tp")

# Memory bandwidth (GB/s) by Apple Silicon chip variant.
# Key: (chip_family, gpu_core_count) or (chip_family,) for single-variant chips.
CHIP_BANDWIDTH: dict[tuple[str, ...], float] = {
    ("M4 Max", "40"): 546.0,
    ("M4 Max", "32"): 410.0,
    ("M4 Pro",): 273.0,
    ("M4",): 120.0,
    ("M3 Ultra",): 800.0,
    ("M3 Max", "40"): 400.0,
    ("M3 Max", "30"): 300.0,
    ("M3 Pro",): 150.0,
    ("M2 Ultra",): 800.0,
    ("M2 Max",): 400.0,
    ("M2 Pro",): 200.0,
    ("M1 Ultra",): 800.0,
    ("M1 Max",): 400.0,
}


@dataclass(frozen=True)
class TPConfig:
    """Immutable tensor parallelism configuration.

    Derived from hardware auto-detection or manual CLI overrides.
    Shared identically between rank 0 and rank 1.
    """

    ratio: float
    rank: int
    world_size: int = 2
    peer_address: str = ""
    local_rdma: str = ""
    remote_rdma: str = ""
    backend: str = "jaccl"
    encrypt: bool = True
    worker_port: int = 8001

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {self.ratio}")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )
        if self.backend not in ("jaccl", "ring"):
            raise ValueError(f"backend must be 'jaccl' or 'ring', got {self.backend!r}")

    @property
    def local_ratio(self) -> float:
        """Weight fraction for this rank."""
        return self.ratio if self.rank == 0 else 1.0 - self.ratio

    @property
    def peer_ratio(self) -> float:
        """Weight fraction for the other rank."""
        return 1.0 - self.local_ratio

    @property
    def is_server(self) -> bool:
        return self.rank == 0

    @property
    def is_worker(self) -> bool:
        return self.rank != 0


def detect_chip_bandwidth() -> float:
    """Detect this machine's memory bandwidth from chip model.

    Uses sysctl to read chip name and GPU core count, then looks up
    bandwidth from the CHIP_BANDWIDTH table.

    Returns bandwidth in GB/s, or 273.0 as conservative default.
    """
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not detect chip model, defaulting to 273 GB/s")
        return 273.0

    # Get GPU core count from system_profiler (sysctl gives CPU cores, not GPU)
    gpu_cores_str = ""
    try:
        gpu_info = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=10
        )
        for line in gpu_info.split("\n"):
            if "Total Number of Cores" in line:
                gpu_cores_str = line.split(":")[-1].strip()
                break
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try specific variant first (chip family + GPU core count), then family only
    for family_key, bw in CHIP_BANDWIDTH.items():
        chip_family = family_key[0]
        if chip_family not in chip:
            continue
        if len(family_key) > 1:
            # Variant-specific: match GPU core count
            if family_key[1] == gpu_cores_str:
                logger.info(f"Detected {chip} with {gpu_cores_str} GPU cores: {bw} GB/s")
                return bw
        else:
            # Single-variant chip family
            logger.info(f"Detected {chip}: {bw} GB/s")
            return bw

    logger.warning(f"Unknown chip '{chip}' (GPU cores: {gpu_cores_str}), defaulting to 273 GB/s")
    return 273.0
