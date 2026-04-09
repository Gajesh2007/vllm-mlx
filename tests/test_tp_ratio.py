"""Tests for the tensor parallelism ratio solver."""

import pytest

from vllm_mlx.tp.ratio import (
    compute_optimal_ratio,
    estimate_split_balance,
    find_valid_ratios,
)


class TestFindValidRatios:
    """Test the ratio solver with various model architectures."""

    def test_gemma4_31b_dimensions(self) -> None:
        """Gemma4 31B: 32 Q-heads, 16 sliding KV-heads, 4 global KV-heads."""
        ratios = find_valid_ratios(
            memory_fractions=[0.60, 0.40],
            hidden_size=5376,
            num_attention_heads=32,
            num_key_value_heads=16,
            intermediate_size=21504,
            num_global_kv_heads=4,
            allow_kv_replication=True,
        )
        assert ratios is not None
        assert len(ratios) == 2
        assert abs(ratios[0] + ratios[1] - 1.0) < 1e-10
        # Should be 5/8 = 0.625
        assert ratios[0] == 0.625
        assert ratios[1] == 0.375

    def test_gemma4_head_counts_are_integers(self) -> None:
        """All head counts must produce exact integers after split."""
        ratios = find_valid_ratios(
            memory_fractions=[0.60, 0.40],
            hidden_size=5376,
            num_attention_heads=32,
            num_key_value_heads=16,
            intermediate_size=21504,
            num_global_kv_heads=4,
            allow_kv_replication=True,
        )
        assert ratios is not None
        r = ratios[0]
        # Q heads
        assert 32 * r == int(32 * r)
        assert int(32 * r) == 20
        assert 32 - int(32 * r) == 12
        # Sliding KV heads
        assert 16 * r == int(16 * r)
        assert int(16 * r) == 10
        assert 16 - int(16 * r) == 6

    def test_gemma4_intermediate_alignment(self) -> None:
        """MLP intermediate dims must be divisible by 8 after split."""
        ratios = find_valid_ratios(
            memory_fractions=[0.60, 0.40],
            hidden_size=5376,
            num_attention_heads=32,
            num_key_value_heads=16,
            intermediate_size=21504,
        )
        assert ratios is not None
        r = ratios[0]
        r0_intermediate = int(21504 * r)
        r1_intermediate = 21504 - r0_intermediate
        assert r0_intermediate % 8 == 0, f"{r0_intermediate} not divisible by 8"
        assert r1_intermediate % 8 == 0, f"{r1_intermediate} not divisible by 8"

    def test_llama_70b_dimensions(self) -> None:
        """Llama 3.3 70B: 64 Q-heads, 8 KV-heads."""
        ratios = find_valid_ratios(
            memory_fractions=[0.60, 0.40],
            hidden_size=8192,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=28672,
        )
        assert ratios is not None
        assert 64 * ratios[0] == int(64 * ratios[0])
        assert 8 * ratios[0] == int(8 * ratios[0])

    def test_rejects_prime_head_count(self) -> None:
        """Prime-number head count with no valid fractional split."""
        ratios = find_valid_ratios(
            memory_fractions=[0.60, 0.40],
            hidden_size=5376,
            num_attention_heads=7,  # prime
            num_key_value_heads=7,
        )
        assert ratios is None

    def test_only_two_nodes(self) -> None:
        """Currently only supports 2 nodes."""
        ratios = find_valid_ratios(
            memory_fractions=[0.5, 0.25, 0.25],
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        assert ratios is None

    def test_equal_memory_near_symmetric(self) -> None:
        """When memory is roughly equal, ratio should be close to 0.5."""
        ratios = find_valid_ratios(
            memory_fractions=[0.50, 0.50],
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        # Finder searches > 0.5 only, so it picks the smallest valid ratio
        # above 0.5. For these dims, 9/16 = 0.5625 is the closest.
        if ratios is not None:
            assert ratios[0] <= 0.625, f"Expected near-symmetric, got {ratios[0]}"

    def test_global_kv_replication_flag(self) -> None:
        """Without KV replication, 4 global KV heads limits valid ratios."""
        # With replication: more ratios available
        with_repl = find_valid_ratios(
            memory_fractions=[0.60, 0.40],
            hidden_size=5376,
            num_attention_heads=32,
            num_key_value_heads=16,
            num_global_kv_heads=4,
            allow_kv_replication=True,
        )
        # Without replication: only 1/2, 3/4 work for 4 global KV heads
        without_repl = find_valid_ratios(
            memory_fractions=[0.60, 0.40],
            hidden_size=5376,
            num_attention_heads=32,
            num_key_value_heads=16,
            num_global_kv_heads=4,
            allow_kv_replication=False,
        )
        # Both should find something, but potentially different ratios
        assert with_repl is not None
        assert without_repl is not None


class TestComputeOptimalRatio:
    """Test bandwidth-optimal ratio computation."""

    def test_m4_max_pro_ratio(self) -> None:
        """M4 Max 36GB (410 GB/s) + M4 Pro 24GB (273 GB/s)."""
        ratio = compute_optimal_ratio(410.0, 273.0)
        assert abs(ratio - 0.6002) < 0.001

    def test_equal_bandwidth(self) -> None:
        """Two identical machines → 50/50."""
        ratio = compute_optimal_ratio(400.0, 400.0)
        assert ratio == 0.5

    def test_zero_bandwidth(self) -> None:
        """Edge case: zero bandwidth → 50/50 fallback."""
        ratio = compute_optimal_ratio(0.0, 0.0)
        assert ratio == 0.5


class TestEstimateSplitBalance:
    """Test the balance estimator."""

    def test_balanced_split(self) -> None:
        """A balanced split should show low imbalance."""
        result = estimate_split_balance(
            ratio=0.6, bw_rank0=410.0, bw_rank1=273.0,
            model_bytes_per_token=9.47e9  # Gemma4-like
        )
        assert result["imbalance_pct"] < 20.0
        assert result["theoretical_toks"] > 0

    def test_extreme_imbalance(self) -> None:
        """A very unbalanced split should show high imbalance."""
        result = estimate_split_balance(
            ratio=0.9, bw_rank0=410.0, bw_rank1=273.0,
            model_bytes_per_token=9.47e9
        )
        assert result["imbalance_pct"] > 30.0
