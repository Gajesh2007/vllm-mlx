"""Ratio solver for asymmetric tensor parallelism.

Finds valid weight split ratios given model dimensions and hardware constraints.
A valid ratio must produce integer head counts and dimensions divisible by
the quantization group size on both ranks.

Adapted from exo's find_valid_ratios() with production hardening:
- Supports dual KV head counts (sliding + full attention, as in Gemma4)
- Hardware bandwidth-aware target ratio
- Clear error messages for impossible configurations
"""

from __future__ import annotations

import logging

logger = logging.getLogger("vllm_mlx.tp")


def find_valid_ratios(
    memory_fractions: list[float],
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int = 0,
    num_global_kv_heads: int = 0,
    quantization_group_size: int = 64,
    allow_kv_replication: bool = True,
) -> list[float] | None:
    """Find the best valid split ratio for asymmetric tensor parallelism.

    A valid ratio r (for rank 0) must satisfy:
    - num_attention_heads * r is a positive integer
    - num_key_value_heads * r is a positive integer (or KV heads are replicated)
    - hidden_size * r is divisible by 8 (quantization alignment)
    - intermediate_size * r is divisible by 8
    - r is in the range (0.5, 0.95] (rank 0 gets the larger share)
    - 1-r satisfies the same constraints for rank 1

    Args:
        memory_fractions: Per-node memory fractions, e.g. [0.60, 0.40].
            Must sum to ~1.0 and have exactly 2 elements.
        hidden_size: Model hidden dimension.
        num_attention_heads: Total query head count.
        num_key_value_heads: KV head count for the primary attention type.
        intermediate_size: MLP intermediate dimension. 0 to skip check.
        num_global_kv_heads: KV head count for secondary attention type
            (e.g., Gemma4's full attention layers with only 4 KV heads).
            0 to skip. When > 0 and allow_kv_replication=True, these heads
            can be replicated instead of split.
        quantization_group_size: Weight quantization group size (typically 64).
        allow_kv_replication: If True and num_global_kv_heads > 0, allow
            ratios where global KV heads cannot be split evenly (they'll
            be replicated on both ranks instead).

    Returns:
        [ratio_rank0, ratio_rank1] if a valid split exists, None otherwise.
    """
    if len(memory_fractions) != 2:
        logger.warning("Asymmetric TP currently only supports 2 nodes")
        return None

    target_ratio = max(memory_fractions)

    # Dimensions that must split cleanly
    key_dims: list[tuple[str, int]] = [
        ("num_attention_heads", num_attention_heads),
        ("hidden_size", hidden_size),
    ]
    if num_key_value_heads > 0:
        key_dims.append(("num_key_value_heads", num_key_value_heads))
    if intermediate_size > 0:
        key_dims.append(("intermediate_size", intermediate_size))

    # Global KV heads: either must split cleanly or will be replicated
    global_kv_dims: list[tuple[str, int]] = []
    if num_global_kv_heads > 0 and not allow_kv_replication:
        key_dims.append(("num_global_kv_heads", num_global_kv_heads))
    elif num_global_kv_heads > 0:
        global_kv_dims.append(("num_global_kv_heads", num_global_kv_heads))

    best_ratio: float | None = None
    best_distance = float("inf")
    best_replicates_global_kv = False

    # Test ratios n/d for d in powers-of-2 and common denominators
    for denom in [2, 4, 8, 16, 32]:
        for numer in range(1, denom):
            ratio = numer / denom
            if ratio <= 0.5 or ratio > 0.95:
                continue

            valid = True
            for name, dim in key_dims:
                r0_count = dim * ratio
                if r0_count != int(r0_count):
                    valid = False
                    break
                r0 = int(r0_count)
                r1 = dim - r0
                if r0 <= 0 or r1 <= 0:
                    valid = False
                    break
                # Quantization alignment for large dimensions
                if dim > quantization_group_size and (r0 % 8 != 0 or r1 % 8 != 0):
                    valid = False
                    break

            if not valid:
                continue

            # Check global KV: can it be split, or must it be replicated?
            replicates_global_kv = False
            for name, dim in global_kv_dims:
                r0_count = dim * ratio
                if r0_count != int(r0_count) or int(r0_count) <= 0 or dim - int(r0_count) <= 0:
                    replicates_global_kv = True
                    # This is OK — we'll replicate these heads

            distance = abs(ratio - target_ratio)
            # Prefer ratios that split global KV over replicating
            if replicates_global_kv and not best_replicates_global_kv and best_ratio is not None:
                # Only pick replicating ratio if it's much closer to target
                if distance >= best_distance - 0.02:
                    continue

            if distance < best_distance:
                best_distance = distance
                best_ratio = ratio
                best_replicates_global_kv = replicates_global_kv

    if best_ratio is None:
        return None

    result = [best_ratio, 1.0 - best_ratio]
    if best_replicates_global_kv:
        logger.info(
            f"Ratio {best_ratio:.4f}: global KV heads ({num_global_kv_heads}) "
            f"will be replicated (too few to split at this ratio)"
        )

    return result


def compute_optimal_ratio(
    local_bandwidth_gbps: float, peer_bandwidth_gbps: float
) -> float:
    """Compute the bandwidth-optimal weight split ratio.

    The optimal ratio balances compute time: rank 0 reads ratio*W at BW0,
    rank 1 reads (1-ratio)*W at BW1. Balanced when ratio/BW0 == (1-ratio)/BW1.

    Solving: ratio = BW0 / (BW0 + BW1).

    This is the TARGET ratio — find_valid_ratios will snap to the nearest
    ratio that produces clean integer dimensions.
    """
    total = local_bandwidth_gbps + peer_bandwidth_gbps
    if total <= 0:
        return 0.5
    return local_bandwidth_gbps / total


def estimate_split_balance(
    ratio: float, bw_rank0: float, bw_rank1: float, model_bytes_per_token: float
) -> dict[str, float]:
    """Estimate per-rank compute time and imbalance for a given split.

    Returns dict with:
        rank0_ms: Time for rank 0 to read its weights (ms)
        rank1_ms: Time for rank 1 to read its weights (ms)
        imbalance_pct: How much slower the bottleneck rank is vs optimal
        theoretical_toks: Theoretical max tok/s (limited by slower rank)
    """
    rank0_bytes = model_bytes_per_token * ratio
    rank1_bytes = model_bytes_per_token * (1.0 - ratio)

    # Convert bytes to GB for division by GB/s, then to ms
    rank0_ms = (rank0_bytes / 1e9 / bw_rank0) * 1000.0
    rank1_ms = (rank1_bytes / 1e9 / bw_rank1) * 1000.0

    bottleneck_ms = max(rank0_ms, rank1_ms)
    optimal_ms = (model_bytes_per_token / 1e9 / (bw_rank0 + bw_rank1)) * 1000.0
    imbalance = (bottleneck_ms / optimal_ms - 1.0) * 100.0 if optimal_ms > 0 else 0.0

    return {
        "rank0_ms": rank0_ms,
        "rank1_ms": rank1_ms,
        "imbalance_pct": imbalance,
        "theoretical_toks": 1000.0 / bottleneck_ms if bottleneck_ms > 0 else 0.0,
    }
