"""Model loader for tensor parallelism.

Uses mlx_lm.load_model(lazy=True) to get a properly configured model,
then applies shard_linear() from MLX's distributed API to split layers.

The shard_linear approach is the MLX-official way to do tensor parallelism.
It handles quantized weights, correct axis splitting, and all_sum internally.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.tp.config import TPConfig
from vllm_mlx.tp.strategy import ShardingStrategy

logger = logging.getLogger("vllm_mlx.tp")


def sharded_load(
    model_path: Path,
    strategy: ShardingStrategy,
    tp_config: TPConfig,
    model_config: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Load a model and apply tensor parallel sharding.

    Uses mlx_lm.load_model(lazy=True) which handles:
    - Config parsing, model class resolution
    - Weight sanitization (key remapping)
    - Quantization (QuantizedLinear/QuantizedEmbedding)
    - Lazy weight loading (memory-mapped, not materialized)

    Then applies strategy.apply_patches() which calls shard_linear()
    to replace linear layers with distributed versions.

    Finally mx.eval() materializes only the sharded weights.
    """
    t0 = time.perf_counter()

    # Memory check
    from vllm_mlx.tp.watchdog import MemoryMonitor
    mem = MemoryMonitor()
    avail, total = mem.check()
    logger.info(f"Memory: {avail:.1f}/{total:.1f} GB available")
    if avail < 4.0:
        raise MemoryError(
            f"Only {avail:.1f} GB available. Reboot may be needed."
        )

    # Load config
    if model_config is None:
        with open(model_path / "config.json") as f:
            model_config = json.load(f)

    # Load model with mlx_lm — handles everything correctly.
    # lazy=True: weights are memory-mapped, ~0 GPU memory.
    from mlx_lm.utils import load_model as _load_model

    model, loaded_config = _load_model(model_path, lazy=True, strict=False)
    logger.info(f"Model loaded lazily: {type(model).__name__}")

    elapsed = time.perf_counter() - t0
    logger.info(f"Lazy load in {elapsed:.1f}s")

    return model, model_config
