"""Performance metrics for tensor parallelism.

Tracks latency at every critical point to detect regressions and identify
bottlenecks. Reports via structured logging and a /v1/tp/metrics endpoint.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("vllm_mlx.tp")


@dataclass
class TPMetrics:
    """Accumulated tensor parallelism metrics.

    Thread-safe: uses a lock for counter updates. Read-only access to
    computed properties is safe without the lock (eventual consistency OK).
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Counters
    total_tokens_generated: int = 0
    total_prefill_tokens: int = 0
    total_requests: int = 0

    # Timing accumulators (seconds)
    total_prefill_time: float = 0.0
    total_decode_time: float = 0.0
    total_all_sum_time: float = 0.0
    total_encrypt_time: float = 0.0

    # Per-request deques (last N for windowed averages)
    _decode_latencies: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000), repr=False
    )
    _prefill_latencies: deque[float] = field(
        default_factory=lambda: deque(maxlen=100), repr=False
    )
    _tps_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=100), repr=False
    )

    # Model info (set once)
    model_name: str = ""
    ratio: float = 0.0
    rank: int = 0
    world_size: int = 2
    theoretical_tps: float = 0.0

    def record_prefill(self, num_tokens: int, elapsed_s: float) -> None:
        with self._lock:
            self.total_prefill_tokens += num_tokens
            self.total_prefill_time += elapsed_s
            self._prefill_latencies.append(elapsed_s)

    def record_decode_token(self, elapsed_s: float) -> None:
        with self._lock:
            self.total_tokens_generated += 1
            self.total_decode_time += elapsed_s
            self._decode_latencies.append(elapsed_s)
            if elapsed_s > 0:
                self._tps_samples.append(1.0 / elapsed_s)

    def record_request_complete(self) -> None:
        with self._lock:
            self.total_requests += 1

    def record_all_sum(self, elapsed_s: float) -> None:
        with self._lock:
            self.total_all_sum_time += elapsed_s

    def record_encrypt(self, elapsed_s: float) -> None:
        with self._lock:
            self.total_encrypt_time += elapsed_s

    @property
    def avg_decode_tps(self) -> float:
        """Average tokens/second over recent samples."""
        samples = list(self._tps_samples)
        return sum(samples) / len(samples) if samples else 0.0

    @property
    def avg_prefill_latency_ms(self) -> float:
        """Average prefill latency in milliseconds."""
        samples = list(self._prefill_latencies)
        return (sum(samples) / len(samples) * 1000) if samples else 0.0

    @property
    def avg_decode_latency_ms(self) -> float:
        """Average per-token decode latency in milliseconds."""
        samples = list(self._decode_latencies)
        return (sum(samples) / len(samples) * 1000) if samples else 0.0

    @property
    def tp_efficiency(self) -> float:
        """Actual throughput as percentage of theoretical ceiling."""
        if self.theoretical_tps <= 0 or self.avg_decode_tps <= 0:
            return 0.0
        return min(self.avg_decode_tps / self.theoretical_tps * 100.0, 100.0)

    def to_dict(self) -> dict:
        """Serialize metrics to a dict for JSON response."""
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "ratio": self.ratio,
            "model": self.model_name,
            "tokens_generated": self.total_tokens_generated,
            "prefill_tokens": self.total_prefill_tokens,
            "requests_completed": self.total_requests,
            "avg_decode_tps": round(self.avg_decode_tps, 1),
            "avg_prefill_latency_ms": round(self.avg_prefill_latency_ms, 1),
            "avg_decode_latency_ms": round(self.avg_decode_latency_ms, 2),
            "total_all_sum_time_s": round(self.total_all_sum_time, 3),
            "total_encrypt_time_s": round(self.total_encrypt_time, 3),
            "theoretical_tps": round(self.theoretical_tps, 1),
            "tp_efficiency_pct": round(self.tp_efficiency, 1),
        }

    def log_summary(self) -> None:
        """Log a human-readable metrics summary."""
        d = self.to_dict()
        logger.info(
            f"TP Metrics: {d['avg_decode_tps']} tok/s "
            f"({d['tp_efficiency_pct']}% of {d['theoretical_tps']} theoretical), "
            f"prefill={d['avg_prefill_latency_ms']}ms, "
            f"decode={d['avg_decode_latency_ms']}ms/tok, "
            f"all_sum={d['total_all_sum_time_s']}s cumulative"
        )


class TPTimer:
    """Context manager for timing TP operations."""

    def __init__(self, metrics: TPMetrics, category: str):
        self.metrics = metrics
        self.category = category
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> TPTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.category == "decode":
            self.metrics.record_decode_token(self.elapsed)
        elif self.category == "prefill":
            pass  # Caller records with token count
        elif self.category == "all_sum":
            self.metrics.record_all_sum(self.elapsed)
        elif self.category == "encrypt":
            self.metrics.record_encrypt(self.elapsed)
