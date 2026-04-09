"""TP model wrapper that broadcasts inputs to rank 1 before each forward pass.

This is the critical integration piece: when SimpleEngine calls model(tokens, cache),
the wrapper intercepts, sends the input tensor to rank 1 (who mirrors the forward
pass), and then both ranks proceed through the sharded layers together. The all_sum
operations in the patched attention/MLP layers synchronize them.

Without this wrapper, rank 0 would hit all_sum in the first layer and hang forever
waiting for rank 1 (who is stuck on mx.distributed.recv waiting for a signal).
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.tp.worker import TPSignal

logger = logging.getLogger("vllm_mlx.tp")


class TPModelWrapper:
    """Wraps a sharded model to broadcast inputs before each forward pass.

    Acts as a transparent proxy: all attribute access is delegated to the
    inner model. The only intercepted method is __call__, which adds
    send/recv synchronization with rank 1 before the actual forward pass.

    This wrapper goes around the TOP-LEVEL model (not individual layers).
    It's what SimpleEngine/mlx-lm's generate() calls as model(tokens, cache).
    """

    def __init__(self, model: nn.Module, group: mx.distributed.Group):
        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, '_inner_model', model)
        object.__setattr__(self, '_group', group)
        object.__setattr__(self, '_rank', group.rank())

    def __call__(self, x: mx.array, cache: list | None = None, **kwargs) -> mx.array:
        """Forward pass with rank 1 synchronization.

        Before running the model, broadcasts the input tensor to rank 1.
        Rank 1's TPBatchWorker receives this and calls model() with the
        same input, keeping both ranks in lockstep via all_sum.
        """
        grp = self._group

        if self._rank == 0:
            # Determine if this is prefill (seq_len > 1) or decode (seq_len == 1)
            seq_len = x.shape[1] if x.ndim > 1 else 1
            is_prefill = seq_len > 1

            if is_prefill:
                sig = mx.array([TPSignal.PREFILL], dtype=mx.int32)
                mx.distributed.send(sig, dst=1, group=grp)
                mx.eval(sig)

                shape_info = mx.array([x.shape[0], x.shape[1]], dtype=mx.int32)
                mx.distributed.send(shape_info, dst=1, group=grp)
                mx.eval(shape_info)

                mx.distributed.send(x, dst=1, group=grp)
                mx.eval(x)
            else:
                sig = mx.array([TPSignal.DECODE_STEP], dtype=mx.int32)
                mx.distributed.send(sig, dst=1, group=grp)
                mx.eval(sig)

                batch_info = mx.array([x.shape[0]], dtype=mx.int32)
                mx.distributed.send(batch_info, dst=1, group=grp)
                mx.eval(batch_info)

                mx.distributed.send(x, dst=1, group=grp)
                mx.eval(x)

        # Both ranks run the actual forward pass.
        # On rank 1, this is called from TPBatchWorker._handle_prefill/decode.
        # On rank 0, this is called here. All_sum in the layers synchronizes them.
        return self._inner_model(x, cache=cache, **kwargs)

    def send_shutdown(self) -> None:
        """Send shutdown signal to rank 1."""
        if self._rank == 0:
            sig = mx.array([TPSignal.SHUTDOWN], dtype=mx.int32)
            mx.distributed.send(sig, dst=1, group=self._group)
            mx.eval(sig)

    def send_reset(self) -> None:
        """Send reset signal to rank 1 (between requests)."""
        if self._rank == 0:
            sig = mx.array([TPSignal.RESET], dtype=mx.int32)
            mx.distributed.send(sig, dst=1, group=self._group)
            mx.eval(sig)

    # --- Transparent proxy: delegate everything to inner model ---

    def __getattr__(self, name: str):
        return getattr(self._inner_model, name)

    def __setattr__(self, name: str, value):
        if name.startswith('_') and name in ('_inner_model', '_group', '_rank'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner_model, name, value)

    def parameters(self):
        return self._inner_model.parameters()

    def eval(self):
        return self._inner_model.eval()

    @property
    def layers(self):
        return self._inner_model.layers

    @property
    def model(self):
        """Some models have model.model (e.g., Gemma4's Model wraps Gemma4TextModel)."""
        return getattr(self._inner_model, 'model', self._inner_model)
