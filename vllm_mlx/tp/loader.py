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

    # 3. Build model architecture (weights will be loaded below)
    from mlx_lm.utils import _get_classes, load_model

    # Determine model class from config.
    # IMPORTANT: pass the FULL config to from_dict(), not text_config.
    # For multimodal wrappers (gemma4), ModelArgs reads text_config as
    # a nested dict. Passing text_config directly makes it empty, causing
    # wrong default head counts (8/1 instead of 32/16).
    model_class, model_args_class = _get_classes(config=model_config)
    model_args = model_args_class.from_dict(model_config)
    model = model_class(model_args)

    # 3b. Apply quantization to match weight format.
    # The model was instantiated as non-quantized (regular nn.Linear/Embedding).
    # The safetensors weights are quantized (packed uint with scales/biases).
    # Without quantizing the model first, embeddings return packed dim (1344)
    # instead of the real hidden dim (5376).
    quant_config = text_config.get("quantization_config", model_config.get("quantization_config"))
    if quant_config:
        from mlx.nn import QuantizedLinear, QuantizedEmbedding

        q_group_size = quant_config.get("group_size", 64)
        q_bits = quant_config.get("bits", 8)

        # Use the model's quant_predicate if available, otherwise default
        quant_pred = getattr(model, "quant_predicate", None)
        if quant_pred:
            nn.quantize(model, group_size=q_group_size, bits=q_bits, class_predicate=quant_pred)
        else:
            nn.quantize(model, group_size=q_group_size, bits=q_bits)
        logger.info(f"Applied quantization: {q_bits}-bit, group_size={q_group_size}")

    # 4. Load sharded weights from safetensors
    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    total_loaded = 0
    total_keys = 0

    # Load weights per-file to bound peak memory. Each file's weights are
    # loaded, sharded, eval'd, assigned to the model, then released.
    #
    # We use mx.load() which handles bfloat16 natively (numpy cannot).
    # Then shard each tensor using mx.split/slice operations.
    # mx.load() returns lazy arrays — they're memory-mapped from safetensors
    # and only materialize on mx.eval(). We shard before eval so only the
    # sharded portion materializes.
    for wf_idx, wf in enumerate(weight_files):
        file_weights: list[tuple[str, mx.array]] = []

        # mx.load returns dict of lazy mx.arrays (memory-mapped)
        raw_weights = mx.load(str(wf))

        for key, tensor in raw_weights.items():
            spec = shard_map.get(key)
            if spec is None:
                logger.debug(f"Unknown weight key {key}, replicating")
                spec = ShardSpec(strategy="replicate")

            shape = tensor.shape

            if spec.strategy == "replicate":
                arr = tensor

            elif spec.strategy == "column":
                axis = spec.axis
                dim = shape[axis]
                split_point = int(dim * spec.ratio)
                if tp_config.rank == 0:
                    arr = _mx_slice_axis(tensor, axis, 0, split_point)
                else:
                    arr = _mx_slice_axis(tensor, axis, split_point, dim)

            elif spec.strategy == "row":
                axis = spec.axis
                if axis < 0:
                    axis = len(shape) + axis
                dim = shape[axis]
                split_point = int(dim * spec.ratio)
                if tp_config.rank == 0:
                    arr = _mx_slice_axis(tensor, axis, 0, split_point)
                else:
                    arr = _mx_slice_axis(tensor, axis, split_point, dim)

                # Bias handling: only rank 0 keeps bias for row-parallel
                is_bias = key.endswith(".bias") or key.endswith(".biases")
                if is_bias and tp_config.rank != 0:
                    arr = mx.zeros_like(arr)
            else:
                logger.warning(f"Unknown shard strategy {spec.strategy} for {key}")
                arr = tensor

            # Force contiguous layout for sharded tensors
            if spec.strategy != "replicate":
                arr = mx.contiguous(arr)

            mx.eval(arr)  # Materialize to GPU
            file_weights.append((key, arr))
            total_loaded += arr.nbytes
            total_keys += 1

        # Assign this file's weights to model, then release
        model.load_weights(file_weights, strict=False)
        del file_weights, raw_weights
        gc.collect()
        logger.info(
            f"Loaded shard {wf_idx + 1}/{len(weight_files)}: "
            f"{wf.name} ({total_keys} keys, {total_loaded / 1e9:.1f} GB so far)"
        )

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
