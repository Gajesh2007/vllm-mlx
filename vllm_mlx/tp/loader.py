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
import numpy as np

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

    # 3. Build model architecture (weights will be loaded below)
    from mlx_lm.utils import _get_classes, load_model

    # Determine model class from config
    model_class, model_args_class = _get_classes(config=model_config)
    model_args = model_args_class.from_dict(text_config)
    model = model_class(model_args)

    # 4. Load sharded weights from safetensors
    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    total_loaded = 0
    total_keys = 0

    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError(
            "safetensors package required for sharded loading. "
            "Install with: pip install safetensors"
        )

    # Load weights per-file to bound peak memory. Each file's weights are
    # loaded, eval'd, assigned to the model, then released before the next file.
    for wf_idx, wf in enumerate(weight_files):
        file_weights: list[tuple[str, mx.array]] = []

        with safe_open(str(wf), framework="numpy") as f:
            for key in f.keys():
                spec = shard_map.get(key)
                if spec is None:
                    # Unknown weight — replicate by default (norms, scalars, etc.)
                    logger.debug(f"Unknown weight key {key}, replicating")
                    spec = ShardSpec(strategy="replicate")

                sl = f.get_slice(key)
                shape = sl.get_shape()

                if spec.strategy == "replicate":
                    data = np.array(sl[:])

                elif spec.strategy == "column":
                    axis = spec.axis
                    dim = shape[axis]
                    split_point = int(dim * spec.ratio)
                    if tp_config.rank == 0:
                        start, end = 0, split_point
                    else:
                        start, end = split_point, dim
                    data = _slice_along_axis(sl, axis, start, end, shape)

                elif spec.strategy == "row":
                    axis = spec.axis
                    if axis < 0:
                        axis = len(shape) + axis
                    dim = shape[axis]
                    split_point = int(dim * spec.ratio)
                    if tp_config.rank == 0:
                        start, end = 0, split_point
                    else:
                        start, end = split_point, dim
                    data = _slice_along_axis(sl, axis, start, end, shape)

                    # Bias handling: only rank 0 keeps bias for row-parallel
                    # layers to avoid doubling after all_sum.
                    # Quantized layers use ".biases", non-quantized use ".bias"
                    is_bias = key.endswith(".bias") or key.endswith(".biases")
                    if is_bias and tp_config.rank != 0:
                        data = np.zeros_like(data)
                else:
                    logger.warning(f"Unknown shard strategy {spec.strategy} for {key}")
                    data = np.array(sl[:])

                arr = mx.array(data)
                mx.eval(arr)  # Materialize to GPU, release numpy buffer
                file_weights.append((key, arr))
                total_loaded += data.nbytes
                total_keys += 1
                del data  # Release numpy immediately

        # Assign this file's weights to model, then release the list
        model.load_weights(file_weights, strict=False)
        del file_weights
        gc.collect()
        logger.debug(
            f"Loaded shard {wf_idx + 1}/{len(weight_files)}: "
            f"{wf.name} ({total_keys} keys so far)"
        )

    elapsed = time.perf_counter() - t0
    total_mb = total_loaded / (1024 * 1024)
    logger.info(
        f"Sharded load complete: {total_mb:.0f} MB in {elapsed:.1f}s "
        f"({total_mb / elapsed:.0f} MB/s), rank {tp_config.rank}"
    )

    return model, model_config


def _slice_along_axis(
    sl: object, axis: int, start: int, end: int, shape: list[int]
) -> np.ndarray:
    """Slice a safetensors slice object along a specific axis.

    Builds the appropriate index tuple for the given axis. For example,
    axis=0: sl[start:end, ...]
    axis=1: sl[:, start:end, ...]
    axis=-1 (already resolved): sl[..., start:end]
    """
    ndim = len(shape)
    idx: list[slice] = [slice(None)] * ndim
    idx[axis] = slice(start, end)
    return np.array(sl[tuple(idx)])
