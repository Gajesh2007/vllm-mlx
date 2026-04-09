"""Tensor parallelism sharding strategy interface.

Each model architecture requires a specific sharding strategy that knows
how to split its weight tensors across ranks. The strategy must handle:

1. Building a sharding map (weight key → how to split)
2. Patching model classes to add all_sum reductions after row-parallel layers
3. Updating attention head counts to reflect the sharded state

New model support = new ShardingStrategy subclass. No changes to core TP code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ShardSpec:
    """Describes how to shard a single weight tensor.

    Attributes:
        strategy: "column" (split output dim), "row" (split input dim),
                  or "replicate" (full copy on every rank).
        axis: Tensor axis to split along. Ignored for "replicate".
        ratio: Rank 0's fraction. Rank 1 gets (1 - ratio). Ignored for "replicate".
        segments: For fused projections (e.g., QKV packed into one weight),
                  split at these boundaries before applying the ratio to each segment.
                  None for non-fused weights.
    """

    strategy: Literal["column", "row", "replicate"]
    axis: int = 0
    ratio: float = 0.5
    segments: list[int] | None = None


@dataclass
class ShardingResult:
    """Output of applying a sharding strategy to a model.

    Carries metadata needed by the caller (loader, metrics, health checks).
    """

    sharding_map: dict[str, ShardSpec] = field(default_factory=dict)
    patched_classes: set[type] = field(default_factory=set)
    rank0_head_counts: dict[str, int] = field(default_factory=dict)
    rank1_head_counts: dict[str, int] = field(default_factory=dict)


class ShardingStrategy(ABC):
    """Base class for model-specific tensor parallel sharding.

    Subclasses implement the three operations needed to distribute a model:

    1. build_sharding_map: Decide how each weight tensor is split.
    2. apply_patches: Monkey-patch model classes to add all_sum reductions.
    3. update_head_counts: Adjust attention head counts after sharding.
    """

    @abstractmethod
    def build_sharding_map(
        self, model_config: dict, ratio: float
    ) -> dict[str, ShardSpec]:
        """Map every weight key to its ShardSpec.

        Args:
            model_config: The model's config.json as a dict.
            ratio: Rank 0's weight fraction (e.g. 0.625 for 5/8 split).

        Returns:
            Dict mapping weight key patterns (with * wildcards for layer indices)
            to ShardSpec instances.
        """
        ...

    @abstractmethod
    def apply_patches(
        self, model: nn.Module, group: mx.distributed.Group, ratio: float
    ) -> set[type]:
        """Patch model layer classes to add distributed reductions.

        For tensor parallelism, row-parallel layers need all_sum after their
        forward pass to recombine partial results from all ranks.

        This MUST patch at the class level (not instance level) because
        nn.Module.__call__ ignores instance method overrides.

        Args:
            model: The loaded (and weight-sharded) model.
            group: MLX distributed group for all_sum calls.
            ratio: Rank 0's weight fraction.

        Returns:
            Set of patched classes (for idempotency tracking).
        """
        ...

    @abstractmethod
    def update_head_counts(
        self, model: nn.Module, rank: int, ratio: float
    ) -> None:
        """Update attention head counts on the model's layers.

        After weight sharding, attention layers hold fewer heads per rank.
        The model's internal head count attributes must be updated to match,
        or the attention computation will produce wrong results.

        Args:
            model: The loaded (and weight-sharded) model.
            rank: This process's rank (0 or 1).
            ratio: Rank 0's weight fraction.
        """
        ...

    @abstractmethod
    def get_layer_type(self, layer_idx: int, model_config: dict) -> str:
        """Return a string identifying the layer type (e.g., 'sliding', 'full').

        Used by the loader to select the correct ShardSpec for layers that
        have different architectures (e.g., Gemma4's sliding vs full attention).
        """
        ...

    def validate_ratio(self, model_config: dict, ratio: float) -> list[str]:
        """Check if this ratio produces valid integer dimensions.

        Returns list of error messages (empty = valid).
        """
        return []
