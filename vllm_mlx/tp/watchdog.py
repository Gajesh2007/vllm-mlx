"""GPU watchdog and memory monitoring for tensor parallelism.

GPUWatchdog: Background thread that force-exits the process if the main thread
hangs in Metal GPU compute for too long. When stuck in GPU compute, no signal
can be delivered to the main thread — but os._exit() from a background thread
ALWAYS works at the kernel level, allowing IOKit to properly free GPU memory.

MemoryMonitor: Periodic memory pressure checks via vm_stat. Warns at
configurable threshold, refuses new work at critical threshold.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger("vllm_mlx.tp")


class GPUWatchdog:
    """Background thread that force-exits if main thread stops heartbeating.

    Solves: When main thread is stuck in Metal GPU compute, signals cannot
    be delivered. kill -9 leaks GPU wired memory permanently. But os._exit()
    from a background thread terminates the process at kernel level, allowing
    IOKit to properly free GPU memory.

    Also watches a stop file for external clean-kill requests (avoids kill -9).
    """

    def __init__(
        self,
        timeout: int = 120,
        stop_file: str = "/tmp/.tp_worker_stop",
        check_interval: float = 2.0,
    ):
        self.timeout = timeout
        self.stop_file = stop_file
        self.check_interval = check_interval
        self.last_heartbeat = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._watch, daemon=True, name="gpu-watchdog")

        # Clean up stale stop file from previous run
        try:
            os.unlink(stop_file)
        except FileNotFoundError:
            pass

        self._thread.start()
        logger.info(f"GPUWatchdog started: timeout={timeout}s, stop_file={stop_file}")

    def heartbeat(self) -> None:
        """Call periodically from main thread to signal liveness."""
        self.last_heartbeat = time.monotonic()

    def stop(self) -> None:
        """Stop the watchdog thread gracefully."""
        self._running = False

    def _watch(self) -> None:
        while self._running:
            time.sleep(self.check_interval)

            # Check external stop file
            if os.path.exists(self.stop_file):
                logger.warning("GPUWatchdog: stop file detected, clean exit")
                os._exit(0)

            # Check heartbeat timeout
            elapsed = time.monotonic() - self.last_heartbeat
            if elapsed > self.timeout:
                logger.error(
                    f"GPUWatchdog: no heartbeat for {elapsed:.0f}s "
                    f"(timeout={self.timeout}s), forcing exit"
                )
                os._exit(1)


class MemoryMonitor:
    """Monitors system memory pressure via vm_stat.

    Apple Silicon unified memory is shared between CPU and GPU.
    When memory pressure is high, Metal allocations start failing
    and the system becomes unstable.
    """

    def __init__(
        self,
        warn_threshold_gb: float = 4.0,
        critical_threshold_gb: float = 2.0,
    ):
        self.warn_threshold_gb = warn_threshold_gb
        self.critical_threshold_gb = critical_threshold_gb
        self._total_gb = self._get_total_memory_gb()

    def check(self) -> tuple[float, float]:
        """Check current memory availability.

        Returns:
            (available_gb, total_gb) tuple.
        """
        available = self._get_available_memory_gb()
        return available, self._total_gb

    def is_critical(self) -> bool:
        """Returns True if memory is critically low."""
        available, _ = self.check()
        return available < self.critical_threshold_gb

    def is_warning(self) -> bool:
        """Returns True if memory is getting low."""
        available, _ = self.check()
        return available < self.warn_threshold_gb

    def log_status(self, prefix: str = "") -> None:
        """Log current memory status."""
        available, total = self.check()
        used = total - available
        pct = (used / total * 100) if total > 0 else 0
        level = logging.WARNING if available < self.warn_threshold_gb else logging.DEBUG
        logger.log(
            level,
            f"{prefix}Memory: {used:.1f}/{total:.1f} GB used ({pct:.0f}%), "
            f"{available:.1f} GB available",
        )

    @staticmethod
    def _get_available_memory_gb() -> float:
        """Get available memory (free + inactive pages) in GB."""
        try:
            out = subprocess.check_output(["vm_stat"], text=True, timeout=5)
            free = inactive = 0
            for line in out.split("\n"):
                if "Pages free" in line:
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    inactive = int(line.split(":")[1].strip().rstrip("."))
            # vm_stat reports in 16KB pages
            return (free + inactive) * 16384 / (1024**3)
        except Exception:
            return 999.0  # Assume plenty if can't check

    @staticmethod
    def _get_total_memory_gb() -> float:
        """Get total physical RAM in GB."""
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
            )
            return int(out.strip()) / (1024**3)
        except Exception:
            return 0.0
