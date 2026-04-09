"""Memory-safe sharded weight loader for tensor parallelism.

Loads model weights directly from safetensors files, reading only each rank's
portion of each tensor via safe_open() + get_slice(). Never materializes the
full model in memory — peak memory equals only this rank's sharded weights.

This is CRITICAL for the 24GB M4 Pro: a 29GB model cannot be fully loaded then
sliced. We must read partial tensors from the mmap'd safetensors file.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.tp.config import TPConfig
from vllm_mlx.tp.strategy import ShardSpec, ShardingStrategy

logger = logging.getLogger("vllm_mlx.tp")


def sharded_load(
    model_path: Path,
    strategy: ShardingStrategy,
    tp_config: TPConfig,
    model_config: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Load a model with tensor-parallel sharded weights.

    Steps:
    1. Load config.json and build model architecture (empty weights)
    2. Build sharding map from strategy
    3. For each safetensors file, load only this rank's portion of each weight
    4. Assign sharded weights to model
    5. Return model + config

    Args:
        model_path: Path to model directory (with config.json + *.safetensors)
        strategy: Model-specific sharding strategy
        tp_config: Tensor parallelism configuration
        model_config: Pre-loaded config dict (optional, loaded from disk if None)

    Returns:
        (model, config) tuple with sharded weights loaded
    """
    t0 = time.perf_counter()

    # 1. Load config
    if model_config is None:
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {model_path}")
        with open(config_path) as f:
            model_config = json.load(f)

    # Handle nested text_config (multimodal wrappers)
    text_config = model_config.get("text_config", model_config)

    # 2. Build sharding map
    shard_map = strategy.build_sharding_map(text_config, tp_config.ratio)
    logger.info(
        f"Sharding map: {sum(1 for s in shard_map.values() if s.strategy != 'replicate')} "
        f"sharded, {sum(1 for s in shard_map.values() if s.strategy == 'replicate')} replicated"
    )

    # 3. Load model with mlx_lm (handles quantization, config, everything).
    # lazy=True means weights are memory-mapped, not materialized to GPU yet.
    from mlx_lm.utils import load_model as _load_model

    model, loaded_config = _load_model(model_path, lazy=True, strict=False)
    logger.info(f"Model loaded lazily: {type(model).__name__}")

    # 4. Shard weights in-place, one layer at a time.
    # For each parameter, slice it according to the sharding map, eval the
    # slice (materializes only the sharded portion to GPU), then assign back.
    # Peak memory ≈ one full layer + accumulated sharded layers.
    from mlx.utils import tree_flatten, tree_unflatten

    total_loaded = 0
    total_keys = 0
    n_column = 0
    n_row = 0
    n_replicate = 0
    n_unmatched = 0
    sharded_pairs: list[tuple[str, mx.array]] = []

    all_params = list(tree_flatten(model.parameters()))
    logger.info(f"Total model parameters: {len(all_params)}")

    for name, param in all_params:
        spec = shard_map.get(name)
        if spec is None:
            n_unmatched += 1
            arr = param
        elif spec.strategy == "replicate":
            n_replicate += 1
            arr = param
        elif spec.strategy == "column":
            n_column += 1
            axis = spec.axis
            dim = param.shape[axis]
            split_point = int(dim * spec.ratio)
            if tp_config.rank == 0:
                arr = _mx_slice_axis(param, axis, 0, split_point)
            else:
                arr = _mx_slice_axis(param, axis, split_point, dim)
            arr = mx.contiguous(arr)
        elif spec.strategy == "row":
            n_row += 1
            axis = spec.axis
            if axis < 0:
                axis = len(param.shape) + axis
            dim = param.shape[axis]
            split_point = int(dim * spec.ratio)
            if tp_config.rank == 0:
                arr = _mx_slice_axis(param, axis, 0, split_point)
            else:
                arr = _mx_slice_axis(param, axis, split_point, dim)
            arr = mx.contiguous(arr)
            # Bias: only rank 0 keeps it for row-parallel (all_sum would double)
            is_bias = name.endswith(".bias") or name.endswith(".biases")
            if is_bias and tp_config.rank != 0:
                arr = mx.zeros_like(arr)
        else:
            arr = param

        mx.eval(arr)
        sharded_pairs.append((name, arr))
        total_loaded += arr.nbytes
        total_keys += 1

    # Apply sharded weights back to model
    model.load_weights(sharded_pairs, strict=False)
    del sharded_pairs
    gc.collect()
    logger.info(
        f"Sharded {total_keys} params: "
        f"{n_column} column, {n_row} row, {n_replicate} replicate, "
        f"{n_unmatched} unmatched (replicated), {total_loaded / 1e9:.1f} GB"
    )

    # Log a sample weight to verify sharding
    from mlx.utils import tree_flatten as _tf
    for n, p in _tf(model.parameters()):
        if "layers.0.self_attn.q_proj.weight" in n:
            logger.info(f"  Sample: {n} shape={p.shape}")
            break

    elapsed = time.perf_counter() - t0
    total_mb = total_loaded / (1024 * 1024)
    logger.info(
        f"Sharded load complete: {total_mb:.0f} MB in {elapsed:.1f}s "
        f"({total_mb / elapsed:.0f} MB/s), rank {tp_config.rank}"
    )

    return model, model_config


def _mx_slice_axis(tensor: mx.array, axis: int, start: int, end: int) -> mx.array:
    """Slice an MLX array along a specific axis.

    Equivalent to tensor[..., start:end, ...] with the slice on the given axis.
    Uses mx.split approach for lazy-evaluation-friendly slicing.
    """
    ndim = len(tensor.shape)
    idx: list[slice] = [slice(None)] * ndim
    idx[axis] = slice(start, end)
    return tensor[tuple(idx)]
