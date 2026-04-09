"""
Tensor Parallelism for vllm-mlx.

Splits model weights across multiple Apple Silicon machines connected via
Thunderbolt 5 RDMA for distributed inference. Supports asymmetric splits
for heterogeneous hardware (e.g., M4 Max + M4 Pro).

Usage:
    vllm-mlx serve model --tensor-parallel --tp-peer 169.254.x.x:12345
"""

from vllm_mlx.tp.config import TPConfig
from vllm_mlx.tp.strategy import ShardingStrategy, ShardSpec

STRATEGY_REGISTRY: dict[str, type[ShardingStrategy]] = {}
_registered = False


def register_strategy(model_type: str, strategy_cls: type[ShardingStrategy]) -> None:
    STRATEGY_REGISTRY[model_type] = strategy_cls


def _ensure_registered() -> None:
    """Lazily register strategies on first use (avoids MLX import at package load)."""
    global _registered
    if _registered:
        return
    _registered = True
    from vllm_mlx.tp.strategies.gemma4 import Gemma4Strategy

    register_strategy("gemma4", Gemma4Strategy)
    register_strategy("gemma4_text", Gemma4Strategy)


def get_strategy(model_type: str) -> ShardingStrategy:
    _ensure_registered()
    cls = STRATEGY_REGISTRY.get(model_type)
    if cls is None:
        supported = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"No tensor parallel strategy for model type '{model_type}'. "
            f"Supported: {supported}"
        )
    return cls()

__all__ = [
    "TPConfig",
    "ShardingStrategy",
    "ShardSpec",
    "get_strategy",
    "register_strategy",
    "STRATEGY_REGISTRY",
]
