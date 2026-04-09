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

    # 3. Build model architecture WITHOUT loading weights.
    # We create the model structure, apply quantization (to get QuantizedLinear/
    # QuantizedEmbedding layer types), then load only our sharded portion
    # of the weights from safetensors files.
    #
    # DO NOT use load_model(lazy=True) — it creates 33.8 GB of lazy weight
    # refs that compete with our sharded weights for GPU memory.
    from mlx_lm.utils import _get_classes

    model_class, model_args_class = _get_classes(config=model_config)
    model_args = model_args_class.from_dict(model_config)
    model = model_class(model_args)

    # Apply quantization to convert Linear→QuantizedLinear etc.
    quantization = model_config.get("quantization") or text_config.get("quantization")
    if quantization:
        def _qpred(path: str, m: nn.Module) -> bool | dict:
            if not hasattr(m, "to_quantized"):
                return False
            qp = getattr(model, "quant_predicate", None)
            return qp(path, m) if qp else True

        nn.quantize(
            model,
            group_size=quantization.get("group_size", 64),
            bits=quantization.get("bits", 8),
            mode=quantization.get("mode", "affine"),
            class_predicate=_qpred,
        )
        logger.info(f"Quantized: {quantization.get('bits')}-bit")

    logger.info(f"Model built: {type(model).__name__} (no weights loaded yet)")

    # 4. Replace weights with sharded versions loaded from safetensors.
    #
    # CRITICAL MEMORY CONSTRAINT:
    # mx.eval(lazy_tensor[0:N]) materializes the FULL lazy tensor first,
    # then slices. On 36GB with a 33.8GB model, this OOMs.
    #
    # Solution: load each safetensors file with mx.load() (returns NEW lazy
    # tensors not linked to the model's lazy graph), shard each one, eval
    # the sharded slice, and assign to the model. The model's original lazy
    # weights are overwritten and never evaluated.
    from mlx.utils import tree_flatten

    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.safetensors"))

    n_column = n_row = n_replicate = n_unmatched = total_loaded = total_keys = 0

    for wf in weight_files:
        raw = mx.load(str(wf))  # Fresh lazy tensors from safetensors mmap
        file_pairs: list[tuple[str, mx.array]] = []

        for key, tensor in raw.items():
            spec = shard_map.get(key)

            if spec is None:
                n_unmatched += 1
                arr = tensor
            elif spec.strategy == "replicate":
                n_replicate += 1
                arr = tensor
            elif spec.strategy == "column":
                n_column += 1
                dim = tensor.shape[spec.axis]
                sp = int(dim * spec.ratio)
                if tp_config.rank == 0:
                    arr = mx.contiguous(_mx_slice_axis(tensor, spec.axis, 0, sp))
                else:
                    arr = mx.contiguous(_mx_slice_axis(tensor, spec.axis, sp, dim))
            elif spec.strategy == "row":
                n_row += 1
                axis = spec.axis
                if axis < 0:
                    axis = len(tensor.shape) + axis
                dim = tensor.shape[axis]
                sp = int(dim * spec.ratio)
                if tp_config.rank == 0:
                    arr = mx.contiguous(_mx_slice_axis(tensor, axis, 0, sp))
                else:
                    arr = mx.contiguous(_mx_slice_axis(tensor, axis, sp, dim))
                if (key.endswith(".bias") or key.endswith(".biases")) and tp_config.rank != 0:
                    arr = mx.zeros_like(arr)
            else:
                arr = tensor

            mx.eval(arr)
            file_pairs.append((key, arr))
            total_loaded += arr.nbytes
            total_keys += 1
            del tensor

        # Assign this file's weights and free
        model.load_weights(file_pairs, strict=False)
        del file_pairs, raw
        gc.collect()

    logger.info(
        f"Sharded {total_keys} params: "
        f"{n_column} column, {n_row} row, {n_replicate} replicate, "
        f"{n_unmatched} unmatched, {total_loaded / 1e9:.1f} GB"
    )
    for n, p in tree_flatten(model.parameters()):
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
