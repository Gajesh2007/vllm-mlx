"""Gemma4 tensor parallel sharding strategy using MLX's shard_linear API.

Uses nn.layers.distributed.shard_linear() to create AllToShardedLinear /
ShardedToAllLinear layers. These handle weight splitting AND all_sum
internally — no manual weight slicing or class-level __call__ patching.

Pattern per transformer layer:
  Q/K/V projections → all-to-sharded (each rank gets partial heads)
  O projection → sharded-to-all (recombines via all_sum)
  MLP gate/up → all-to-sharded
  MLP down → sharded-to-all
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from vllm_mlx.tp.strategy import ShardSpec, ShardingStrategy

logger = logging.getLogger("vllm_mlx.tp")


class Gemma4Strategy(ShardingStrategy):
    """Tensor parallel sharding for Gemma4 using MLX's official shard_linear API."""

    def build_sharding_map(
        self, model_config: dict, ratio: float
    ) -> dict[str, ShardSpec]:
        """Build weight key → ShardSpec mapping.

        Note: with the shard_linear approach, the sharding map is only used
        for the per-file weight loader (to know which axis to slice for
        memory-efficient loading). The actual distributed behavior is handled
        by shard_linear replacing the layer types.
        """
        num_layers = model_config.get("num_hidden_layers", 60)
        layer_types = self._compute_layer_types(model_config)
        k_eq_v = model_config.get("attention_k_eq_v", False)

        shard_map: dict[str, ShardSpec] = {}

        for i in range(num_layers):
            prefix = f"language_model.model.layers.{i}"
            is_full = layer_types[i] == "full_attention"
            is_k_eq_v = k_eq_v and is_full

            # Q: always column-parallel
            for suffix in ("weight", "scales", "biases"):
                shard_map[f"{prefix}.self_attn.q_proj.{suffix}"] = ShardSpec(
                    strategy="column", axis=0, ratio=ratio
                )

            # K: column-parallel for sliding, replicate for full (K=V, too few heads)
            for suffix in ("weight", "scales", "biases"):
                if is_k_eq_v:
                    shard_map[f"{prefix}.self_attn.k_proj.{suffix}"] = ShardSpec(strategy="replicate")
                else:
                    shard_map[f"{prefix}.self_attn.k_proj.{suffix}"] = ShardSpec(
                        strategy="column", axis=0, ratio=ratio
                    )

            # V: only on sliding layers
            if not is_k_eq_v:
                for suffix in ("weight", "scales", "biases"):
                    shard_map[f"{prefix}.self_attn.v_proj.{suffix}"] = ShardSpec(
                        strategy="column", axis=0, ratio=ratio
                    )

            # O: row-parallel
            for suffix in ("weight", "scales", "biases"):
                shard_map[f"{prefix}.self_attn.o_proj.{suffix}"] = ShardSpec(
                    strategy="row", axis=-1, ratio=ratio
                )

            # MLP
            for proj in ("gate_proj", "up_proj"):
                for suffix in ("weight", "scales", "biases"):
                    shard_map[f"{prefix}.mlp.{proj}.{suffix}"] = ShardSpec(
                        strategy="column", axis=0, ratio=ratio
                    )
            for suffix in ("weight", "scales", "biases"):
                shard_map[f"{prefix}.mlp.down_proj.{suffix}"] = ShardSpec(
                    strategy="row", axis=-1, ratio=ratio
                )

            # Norms, scalars: replicate
            for norm in ("input_layernorm", "post_attention_layernorm",
                         "post_feedforward_layernorm", "pre_feedforward_layernorm"):
                shard_map[f"{prefix}.{norm}.weight"] = ShardSpec(strategy="replicate")
            for norm in ("q_norm", "k_norm"):
                shard_map[f"{prefix}.self_attn.{norm}.weight"] = ShardSpec(strategy="replicate")
            shard_map[f"{prefix}.layer_scalar"] = ShardSpec(strategy="replicate")

        shard_map["language_model.model.embed_tokens.weight"] = ShardSpec(strategy="replicate")
        shard_map["language_model.model.norm.weight"] = ShardSpec(strategy="replicate")

        return shard_map

    def apply_patches(
        self, model: nn.Module, group: mx.distributed.Group, ratio: float
    ) -> set[type]:
        """Replace linear layers with distributed versions using shard_linear.

        This is the MLX-official approach: shard_linear creates
        AllToShardedLinear / ShardedToAllLinear layers that handle
        weight splitting AND all_sum internally. No monkey-patching needed.
        """
        inner = self._get_inner_model(model)
        layer_types = self._compute_layer_types_from_model(inner)

        k_eq_v = getattr(inner.layers[0].self_attn, "use_k_eq_v", False) if inner.layers else False

        for i, layer in enumerate(inner.layers):
            attn = layer.self_attn
            is_full = layer_types[i] == "full_attention"
            is_k_eq_v_layer = getattr(attn, "use_k_eq_v", False)

            # Q: all-to-sharded
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)

            # K: all-to-sharded for sliding, leave as-is for full (replicated)
            if not is_k_eq_v_layer:
                attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
                if hasattr(attn, "v_proj"):
                    attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)

            # O: sharded-to-all (handles all_sum internally)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)

            # MLP
            mlp = layer.mlp
            mlp.gate_proj = shard_linear(mlp.gate_proj, "all-to-sharded", group=group)
            mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
            mlp.down_proj = shard_linear(mlp.down_proj, "sharded-to-all", group=group)

            if i == 0:
                logger.info(
                    f"Layer 0: q_proj type={type(attn.q_proj).__name__}, "
                    f"o_proj type={type(attn.o_proj).__name__}"
                )

        logger.info(f"Sharded {len(inner.layers)} layers with shard_linear")
        return set()  # No class-level patches needed

    def update_head_counts(
        self, model: nn.Module, rank: int, ratio: float
    ) -> None:
        """Update attention head counts after sharding.

        shard_linear splits the weight matrices, but the model's n_heads
        attribute still reflects the original count. We must update it
        to match the sharded output dimension.
        """
        inner = self._get_inner_model(model)
        group_size = 2  # world_size

        for i, layer in enumerate(inner.layers):
            attn = layer.self_attn
            is_k_eq_v = getattr(attn, "use_k_eq_v", False)

            # shard_linear with equal split divides by group.size()
            original_heads = attn.n_heads
            attn.n_heads = attn.n_heads // group_size

            if not is_k_eq_v:
                attn.n_kv_heads = attn.n_kv_heads // group_size

            if i == 0:
                logger.info(
                    f"Layer 0 ({'full' if is_k_eq_v else 'sliding'}): "
                    f"n_heads {original_heads}→{attn.n_heads}"
                )

    def get_layer_type(self, layer_idx: int, model_config: dict) -> str:
        layer_types = self._compute_layer_types(model_config)
        return layer_types[layer_idx] if layer_idx < len(layer_types) else "sliding_attention"

    def validate_ratio(self, model_config: dict, ratio: float) -> list[str]:
        # With shard_linear (equal split by group.size()), ratio is informational.
        # The actual split is always 50/50 for 2 ranks.
        # TODO: support asymmetric TP with shard_linear
        return []

    def _compute_layer_types(self, model_config: dict) -> list[str]:
        num_layers = model_config.get("num_hidden_layers", 60)
        if "layer_types" in model_config:
            return model_config["layer_types"][:num_layers]
        pattern_len = model_config.get("sliding_window_pattern", 6)
        if pattern_len is None:
            pattern_len = 6
        pattern = ["sliding_attention"] * (pattern_len - 1) + ["full_attention"]
        return (pattern * (num_layers // len(pattern) + 1))[:num_layers]

    def _compute_layer_types_from_model(self, inner) -> list[str]:
        """Get layer types from the actual model layers."""
        types = []
        for layer in inner.layers:
            lt = getattr(layer, "layer_type", "sliding_attention")
            types.append(lt)
        return types

    def _get_inner_model(self, model: nn.Module) -> Any:
        """Extract the Gemma4TextModel from the wrapper."""
        lm = getattr(model, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", lm)
            if hasattr(inner, "layers"):
                return inner
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
        if hasattr(model, "layers"):
            return model
        raise ValueError(f"Cannot find layers in model: {type(model).__name__}")
