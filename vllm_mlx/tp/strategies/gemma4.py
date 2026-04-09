"""Gemma4 tensor parallel sharding strategy.

Handles the unique aspects of Gemma4's architecture:
- Hybrid attention: sliding (head_dim=256, 16 KV) + full (head_dim=512, 4 KV, K=V)
- K=V on full layers: no v_proj weight, k_proj output used as both K and V
- Global KV replication: 4 full-attention KV heads are replicated (too few to split)
- Sliding window pattern: configurable repeating pattern of layer types
- Logit softcapping: tanh(logits/30)*30 applied post-lm_head
"""

from __future__ import annotations

import logging
import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.tp.strategy import ShardSpec, ShardingStrategy

logger = logging.getLogger("vllm_mlx.tp")

# Track patched classes to avoid double-patching
_patched_classes: set[type] = set()


class Gemma4Strategy(ShardingStrategy):
    """Tensor parallel sharding for Gemma4 31B (and variants)."""

    def build_sharding_map(
        self, model_config: dict, ratio: float
    ) -> dict[str, ShardSpec]:
        """Build weight key → ShardSpec mapping for Gemma4.

        Weight keys follow the pattern:
            model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
            model.layers.{i}.mlp.{gate,up,down}_proj.{weight,scales,biases}
            model.layers.{i}.{input,post_attention,post_feedforward,pre_feedforward}_layernorm.weight
            model.layers.{i}.self_attn.{q,k}_norm.weight
            model.embed_tokens.weight
            model.norm.weight
        """
        num_layers = model_config.get("num_hidden_layers", 60)
        layer_types = self._compute_layer_types(model_config)
        k_eq_v = model_config.get("attention_k_eq_v", False)

        shard_map: dict[str, ShardSpec] = {}

        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            is_full = layer_types[i] == "full_attention"
            is_k_eq_v = k_eq_v and is_full

            # Q projection: always column-parallel (split output dim = head dim)
            for suffix in ("weight", "scales", "biases"):
                key = f"{prefix}.self_attn.q_proj.{suffix}"
                shard_map[key] = ShardSpec(strategy="column", axis=0, ratio=ratio)

            # K projection: column-parallel for sliding, replicate for full (K=V)
            for suffix in ("weight", "scales", "biases"):
                key = f"{prefix}.self_attn.k_proj.{suffix}"
                if is_k_eq_v:
                    shard_map[key] = ShardSpec(strategy="replicate")
                else:
                    shard_map[key] = ShardSpec(strategy="column", axis=0, ratio=ratio)

            # V projection: only exists on sliding layers (not K=V)
            if not is_k_eq_v:
                for suffix in ("weight", "scales", "biases"):
                    key = f"{prefix}.self_attn.v_proj.{suffix}"
                    shard_map[key] = ShardSpec(strategy="column", axis=0, ratio=ratio)

            # O projection: row-parallel (split input dim)
            for suffix in ("weight", "scales", "biases"):
                key = f"{prefix}.self_attn.o_proj.{suffix}"
                shard_map[key] = ShardSpec(strategy="row", axis=-1, ratio=ratio)

            # MLP: gate/up are column-parallel, down is row-parallel
            for proj_name in ("gate_proj", "up_proj"):
                for suffix in ("weight", "scales", "biases"):
                    key = f"{prefix}.mlp.{proj_name}.{suffix}"
                    shard_map[key] = ShardSpec(strategy="column", axis=0, ratio=ratio)

            for suffix in ("weight", "scales", "biases"):
                key = f"{prefix}.mlp.down_proj.{suffix}"
                shard_map[key] = ShardSpec(strategy="row", axis=-1, ratio=ratio)

            # Norms and scalars: replicate
            for norm_name in (
                "input_layernorm",
                "post_attention_layernorm",
                "post_feedforward_layernorm",
                "pre_feedforward_layernorm",
            ):
                key = f"{prefix}.{norm_name}.weight"
                shard_map[key] = ShardSpec(strategy="replicate")

            # Attention norms (q_norm, k_norm if present)
            for norm_name in ("q_norm", "k_norm"):
                key = f"{prefix}.self_attn.{norm_name}.weight"
                shard_map[key] = ShardSpec(strategy="replicate")

            # Per-layer scalar
            key = f"{prefix}.layer_scalar"
            shard_map[key] = ShardSpec(strategy="replicate")

        # Embeddings: replicate (shared across ranks, used for both input + lm_head)
        shard_map["model.embed_tokens.weight"] = ShardSpec(strategy="replicate")
        # Final norm: replicate
        shard_map["model.norm.weight"] = ShardSpec(strategy="replicate")

        return shard_map

    def apply_patches(
        self, model: nn.Module, group: mx.distributed.Group, ratio: float
    ) -> set[type]:
        """Patch Gemma4 attention and MLP classes for all_sum reductions.

        After row-parallel layers (o_proj, down_proj), the partial results from
        each rank must be summed. We patch __call__ at the class level.
        """
        inner = self._get_inner_model(model)
        if not hasattr(inner, "layers") or len(inner.layers) == 0:
            raise ValueError("Model has no layers — cannot apply TP patches")

        patched: set[type] = set()

        # Patch attention class
        attn_sample = inner.layers[0].self_attn
        attn_cls = type(attn_sample)
        if attn_cls not in _patched_classes:
            _patch_attention_class(attn_cls, group)
            _patched_classes.add(attn_cls)
            patched.add(attn_cls)

        # Patch MLP class
        mlp_sample = inner.layers[0].mlp
        mlp_cls = type(mlp_sample)
        if mlp_cls not in _patched_classes:
            _patch_mlp_class(mlp_cls, group)
            _patched_classes.add(mlp_cls)
            patched.add(mlp_cls)

        # Mark all attention layers with the group (used by patched __call__)
        for layer in inner.layers:
            layer.self_attn._tp_group = group
            layer.mlp._tp_group = group

        logger.info(
            f"Patched {len(patched)} classes for all_sum: "
            f"{[c.__name__ for c in patched]}"
        )
        return patched

    def update_head_counts(
        self, model: nn.Module, rank: int, ratio: float
    ) -> None:
        """Update attention head counts after weight sharding."""
        inner = self._get_inner_model(model)
        r = ratio if rank == 0 else (1.0 - ratio)

        for i, layer in enumerate(inner.layers):
            attn = layer.self_attn
            is_full = getattr(layer, "layer_type", "") == "full_attention"
            is_k_eq_v = getattr(attn, "use_k_eq_v", False)

            # Q heads: always split
            original_n_heads = attn.n_heads
            attn.n_heads = int(original_n_heads * r)

            if is_k_eq_v:
                # Full K=V layers: KV heads replicated, don't change count
                pass
            else:
                # Sliding layers: KV heads split
                original_kv = attn.n_kv_heads
                attn.n_kv_heads = int(original_kv * r)

            if i == 0:
                logger.info(
                    f"Layer 0 ({'full' if is_full else 'sliding'}): "
                    f"n_heads {original_n_heads}→{attn.n_heads}, "
                    f"n_kv_heads {'unchanged' if is_k_eq_v else f'{original_kv}→{attn.n_kv_heads}'}"
                )

    def get_layer_type(self, layer_idx: int, model_config: dict) -> str:
        """Return 'sliding_attention' or 'full_attention'."""
        layer_types = self._compute_layer_types(model_config)
        if layer_idx < len(layer_types):
            return layer_types[layer_idx]
        return "sliding_attention"

    def validate_ratio(self, model_config: dict, ratio: float) -> list[str]:
        """Check that ratio produces valid integer head counts."""
        errors: list[str] = []
        num_heads = model_config.get("num_attention_heads", 32)
        num_kv = model_config.get("num_key_value_heads", 16)
        intermediate = model_config.get("intermediate_size", 0)

        r0 = ratio
        r1 = 1.0 - ratio

        for name, dim, r in [
            ("Q heads (rank 0)", num_heads, r0),
            ("Q heads (rank 1)", num_heads, r1),
            ("KV heads (rank 0)", num_kv, r0),
            ("KV heads (rank 1)", num_kv, r1),
        ]:
            count = dim * r
            if count != int(count) or int(count) <= 0:
                errors.append(f"{name}: {dim} * {r:.4f} = {count:.4f} (not integer)")

        for name, dim, r in [
            ("intermediate (rank 0)", intermediate, r0),
            ("intermediate (rank 1)", intermediate, r1),
        ]:
            val = int(dim * r)
            if val % 8 != 0:
                errors.append(f"{name}: {val} not divisible by 8")

        return errors

    def _compute_layer_types(self, model_config: dict) -> list[str]:
        """Compute per-layer type list from config."""
        num_layers = model_config.get("num_hidden_layers", 60)
        # Gemma4 uses sliding_window_pattern to determine the repeating pattern.
        # Config field name varies: sliding_window_pattern or directly layer_types.
        if "layer_types" in model_config:
            return model_config["layer_types"][:num_layers]

        pattern_len = model_config.get("sliding_window_pattern", 6)
        pattern = ["sliding_attention"] * (pattern_len - 1) + ["full_attention"]
        return (pattern * (num_layers // len(pattern) + 1))[:num_layers]

    def _get_inner_model(self, model: nn.Module) -> Any:
        """Extract the Gemma4TextModel from the wrapper.

        Model hierarchy for gemma4 multimodal wrapper:
          gemma4.Model
            └── .language_model → gemma4_text.Model
                  └── .model → Gemma4TextModel (has .layers)

        For gemma4_text directly:
          gemma4_text.Model
            └── .model → Gemma4TextModel (has .layers)
        """
        # Try gemma4 multimodal wrapper: model.language_model.model
        lm = getattr(model, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", lm)
            if hasattr(inner, "layers"):
                return inner

        # Try gemma4_text direct: model.model
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner

        # Model itself has layers
        if hasattr(model, "layers"):
            return model

        raise ValueError(
            "Cannot find inner Gemma4TextModel with .layers attribute. "
            f"Model type: {type(model).__name__}"
        )


def _patch_attention_class(attn_cls: type, group: mx.distributed.Group) -> None:
    """Patch attention class __call__ to add all_sum after o_proj.

    This is applied at the CLASS level because nn.Module.__call__ ignores
    instance-level overrides. The _tp_group attribute on each instance
    controls whether all_sum is applied (allows mixed patched/unpatched).
    """
    original_call = attn_cls.__call__

    def patched_call(
        self: Any,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
    ) -> Any:
        result = original_call(self, x, mask=mask, cache=cache)
        grp = getattr(self, "_tp_group", None)
        if grp is not None:
            # result is (output, (keys, values), offset)
            # Apply all_sum to the output tensor only
            if isinstance(result, tuple) and len(result) >= 1:
                output = mx.distributed.all_sum(result[0], group=grp)
                return (output,) + result[1:]
            return mx.distributed.all_sum(result, group=grp)
        return result

    attn_cls.__call__ = patched_call
    logger.debug(f"Patched {attn_cls.__name__}.__call__ with all_sum")


def _patch_mlp_class(mlp_cls: type, group: mx.distributed.Group) -> None:
    """Patch MLP class __call__ to add all_sum after down_proj output."""
    original_call = mlp_cls.__call__

    def patched_call(self: Any, x: mx.array) -> mx.array:
        result = original_call(self, x)
        grp = getattr(self, "_tp_group", None)
        if grp is not None:
            return mx.distributed.all_sum(result, group=grp)
        return result

    mlp_cls.__call__ = patched_call
    logger.debug(f"Patched {mlp_cls.__name__}.__call__ with all_sum")
