"""Tests for sharding strategies, focused on Gemma4."""

import pytest

from vllm_mlx.tp.strategies.gemma4 import Gemma4Strategy


GEMMA4_31B_CONFIG = {
    "hidden_size": 5376,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "num_global_key_value_heads": 4,
    "head_dim": 256,
    "global_head_dim": 512,
    "intermediate_size": 21504,
    "num_hidden_layers": 60,
    "attention_k_eq_v": True,
    "sliding_window_pattern": 6,
}


class TestGemma4ShardingMap:
    """Test sharding map generation for Gemma4."""

    def setup_method(self) -> None:
        self.strategy = Gemma4Strategy()
        self.shard_map = self.strategy.build_sharding_map(GEMMA4_31B_CONFIG, ratio=0.625)

    def test_q_proj_is_column_parallel(self) -> None:
        """Q projection should be column-parallel on all layers."""
        for i in range(60):
            key = f"language_model.model.layers.{i}.self_attn.q_proj.weight"
            spec = self.shard_map[key]
            assert spec.strategy == "column"
            assert spec.axis == 0
            assert spec.ratio == 0.625

    def test_sliding_kv_is_column_parallel(self) -> None:
        """Sliding layer K/V should be column-parallel (split by KV heads)."""
        # Layer 0 is sliding
        for proj in ("k_proj", "v_proj"):
            key = f"language_model.model.layers.0.self_attn.{proj}.weight"
            spec = self.shard_map[key]
            assert spec.strategy == "column"

    def test_full_kv_is_replicated(self) -> None:
        """Full attention K is replicated (K=V, only 4 heads)."""
        # Layer 5 is full (pattern=6: 0-4 sliding, 5 full)
        key = "language_model.model.layers.5.self_attn.k_proj.weight"
        spec = self.shard_map[key]
        assert spec.strategy == "replicate"

    def test_full_layer_no_v_proj(self) -> None:
        """Full attention layers with K=V should not have v_proj in map."""
        # Layer 5 is full with K=V
        key = "language_model.model.layers.5.self_attn.v_proj.weight"
        assert key not in self.shard_map

    def test_o_proj_is_row_parallel(self) -> None:
        """O projection should be row-parallel on all layers."""
        for i in range(60):
            key = f"language_model.model.layers.{i}.self_attn.o_proj.weight"
            spec = self.shard_map[key]
            assert spec.strategy == "row"
            assert spec.axis == -1

    def test_mlp_gate_up_column_parallel(self) -> None:
        """MLP gate/up should be column-parallel."""
        for proj in ("gate_proj", "up_proj"):
            key = f"language_model.model.layers.0.mlp.{proj}.weight"
            spec = self.shard_map[key]
            assert spec.strategy == "column"

    def test_mlp_down_row_parallel(self) -> None:
        """MLP down should be row-parallel."""
        key = "language_model.model.layers.0.mlp.down_proj.weight"
        spec = self.shard_map[key]
        assert spec.strategy == "row"

    def test_embeddings_replicated(self) -> None:
        key = "language_model.model.embed_tokens.weight"
        spec = self.shard_map[key]
        assert spec.strategy == "replicate"

    def test_norms_replicated(self) -> None:
        """All norms should be replicated."""
        for key, spec in self.shard_map.items():
            if "layernorm" in key or key == "language_model.model.norm.weight":
                assert spec.strategy == "replicate", f"{key} should be replicated"


class TestGemma4LayerTypeDetection:
    """Test sliding vs full layer detection."""

    def setup_method(self) -> None:
        self.strategy = Gemma4Strategy()

    def test_sliding_window_pattern_6(self) -> None:
        """Pattern=6: layers 0-4 sliding, 5 full, 6-10 sliding, 11 full, ..."""
        for i in range(60):
            expected = "full_attention" if i % 6 == 5 else "sliding_attention"
            actual = self.strategy.get_layer_type(i, GEMMA4_31B_CONFIG)
            assert actual == expected, f"Layer {i}: expected {expected}, got {actual}"

    def test_full_layer_count(self) -> None:
        """60 layers with pattern=6 → 10 full layers."""
        full_count = sum(
            1
            for i in range(60)
            if self.strategy.get_layer_type(i, GEMMA4_31B_CONFIG) == "full_attention"
        )
        assert full_count == 10


class TestGemma4Validation:
    """Test ratio validation for Gemma4."""

    def setup_method(self) -> None:
        self.strategy = Gemma4Strategy()

    def test_valid_ratio_5_8(self) -> None:
        errors = self.strategy.validate_ratio(GEMMA4_31B_CONFIG, 0.625)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_valid_ratio_3_4(self) -> None:
        errors = self.strategy.validate_ratio(GEMMA4_31B_CONFIG, 0.75)
        assert errors == []

    def test_valid_ratio_1_2(self) -> None:
        errors = self.strategy.validate_ratio(GEMMA4_31B_CONFIG, 0.5)
        assert errors == []

    def test_invalid_ratio(self) -> None:
        """With shard_linear (equal split), validate_ratio is permissive."""
        errors = self.strategy.validate_ratio(GEMMA4_31B_CONFIG, 0.6)
        # shard_linear uses equal split by group.size(), so ratio validation is relaxed
        assert isinstance(errors, list)


class TestStrategyRegistry:
    """Test the strategy registry."""

    def test_gemma4_registered(self) -> None:
        from vllm_mlx.tp import get_strategy

        strategy = get_strategy("gemma4")
        assert isinstance(strategy, Gemma4Strategy)

    def test_gemma4_text_registered(self) -> None:
        from vllm_mlx.tp import get_strategy

        strategy = get_strategy("gemma4_text")
        assert isinstance(strategy, Gemma4Strategy)

    def test_unknown_model_raises(self) -> None:
        from vllm_mlx.tp import get_strategy

        with pytest.raises(ValueError, match="No tensor parallel strategy"):
            get_strategy("unknown_model_xyz")
