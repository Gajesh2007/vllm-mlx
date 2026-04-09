"""TP-aware generation that coordinates rank 0 and rank 1.

Wraps vllm-mlx's generation path to broadcast inputs/tokens to the worker.
The worker participates in forward passes via all_sum in the sharded model.

Two modes:
1. Simple (SimpleEngine): Single request at a time. Rank 0 broadcasts
   prompt tokens and each sampled token.
2. Batched (Scheduler): Multiple concurrent requests. Rank 0 sends
   INSERT/STEP/FILTER signals to keep worker's BatchGenerator in sync.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.tp.config import TPConfig
from vllm_mlx.tp.metrics import TPMetrics, TPTimer
from vllm_mlx.tp.worker import TPSignal

logger = logging.getLogger("vllm_mlx.tp")


def tp_send_signal(group: mx.distributed.Group, signal: TPSignal) -> None:
    """Send a control signal to rank 1."""
    sig = mx.array([signal.value], dtype=mx.int32)
    mx.distributed.send(sig, dst=1, group=group)
    mx.eval(sig)


def tp_send_tokens(
    group: mx.distributed.Group,
    tokens: mx.array,
    batch_size: int | None = None,
    seq_len: int | None = None,
) -> None:
    """Send token tensor to rank 1, preceded by shape info."""
    if batch_size is None:
        batch_size = tokens.shape[0]
    if seq_len is None:
        seq_len = tokens.shape[1] if tokens.ndim > 1 else 1

    shape_info = mx.array([batch_size, seq_len], dtype=mx.int32)
    mx.distributed.send(shape_info, dst=1, group=group)
    mx.eval(shape_info)

    mx.distributed.send(tokens, dst=1, group=group)
    mx.eval(tokens)


def tp_send_filter(
    group: mx.distributed.Group, keep_indices: list[int]
) -> None:
    """Send filter (sequence removal) signal to rank 1."""
    tp_send_signal(group, TPSignal.FILTER)

    count = mx.array([len(keep_indices)], dtype=mx.int32)
    mx.distributed.send(count, dst=1, group=group)
    mx.eval(count)

    if keep_indices:
        indices = mx.array(keep_indices, dtype=mx.int32)
        mx.distributed.send(indices, dst=1, group=group)
        mx.eval(indices)


def tp_generate_simple(
    model: nn.Module,
    tokenizer: object,
    prompt: str,
    max_tokens: int,
    group: mx.distributed.Group,
    tp_config: TPConfig,
    metrics: TPMetrics | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Generator[int, None, None]:
    """Generate tokens with tensor parallelism (SimpleEngine path).

    Yields token IDs one at a time for streaming.

    Rank 0:
    1. Tokenize prompt
    2. Send PREFILL signal + tokens to rank 1
    3. Run prefill forward pass (all_sum syncs)
    4. For each decode step:
       a. Sample next token
       b. Send DECODE_STEP signal + token to rank 1
       c. Run decode forward pass (all_sum syncs)
       d. Yield token

    Rank 1 mirrors via TPBatchWorker.
    """
    if not tp_config.is_server:
        raise RuntimeError("tp_generate_simple should only be called on rank 0")

    # Reset encrypted all_sum counter for this request (prevents pool overflow)
    from vllm_mlx.tp.encryption import EncryptedAllSum
    # The counter is in the closure — access via the module-level reference
    if hasattr(mx.distributed, '_tp_encrypted_all_sum'):
        mx.distributed._tp_encrypted_all_sum.reset_counter()

    # Tokenize
    if hasattr(tokenizer, "encode"):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = list(prompt) if isinstance(prompt, (list, tuple)) else [int(prompt)]

    prompt_tensor = mx.array([prompt_tokens])  # (1, seq_len)

    # Create cache
    from mlx_lm.utils import make_prompt_cache
    cache = make_prompt_cache(model)

    # Prefill
    tp_send_signal(group, TPSignal.PREFILL)
    tp_send_tokens(group, prompt_tensor)

    t0 = time.perf_counter()
    logits = model(prompt_tensor, cache=cache)
    mx.eval(logits)
    prefill_time = time.perf_counter() - t0

    if metrics:
        metrics.record_prefill(len(prompt_tokens), prefill_time)

    # Get stop tokens
    stop_tokens = set()
    if hasattr(tokenizer, "eos_token_id"):
        eos = tokenizer.eos_token_id
        if isinstance(eos, int):
            stop_tokens.add(eos)
        elif isinstance(eos, list):
            stop_tokens.update(eos)

    # Sample first token from prefill logits
    last_logits = logits[:, -1, :]
    token = _sample(last_logits, temperature, top_p)
    token_id = token.item()

    if token_id in stop_tokens:
        tp_send_signal(group, TPSignal.RESET)
        return

    yield token_id

    # Decode loop
    for i in range(max_tokens - 1):
        token_tensor = mx.array([[token_id]])  # (1, 1)

        # Send decode step to worker
        tp_send_signal(group, TPSignal.DECODE_STEP)
        batch_info = mx.array([1], dtype=mx.int32)  # batch_size
        mx.distributed.send(batch_info, dst=1, group=group)
        mx.eval(batch_info)
        mx.distributed.send(token_tensor, dst=1, group=group)
        mx.eval(token_tensor)

        # Forward pass
        t0 = time.perf_counter()
        logits = model(token_tensor, cache=cache)
        mx.eval(logits)
        decode_time = time.perf_counter() - t0

        if metrics:
            metrics.record_decode_token(decode_time)

        # Sample
        last_logits = logits[:, -1, :]
        token = _sample(last_logits, temperature, top_p)
        token_id = token.item()

        if token_id in stop_tokens:
            break

        yield token_id

    # Signal reset to worker
    tp_send_signal(group, TPSignal.RESET)

    if metrics:
        metrics.record_request_complete()


def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Sample a token from logits."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_p < 1.0:
        # Top-p (nucleus) sampling
        # Sort descending so highest-prob tokens come first
        sorted_indices = mx.argsort(logits, axis=-1)
        # argsort is ascending; reverse for descending
        sorted_indices = sorted_indices[..., ::-1]
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        # Mask tokens whose cumulative probability exceeds top_p
        mask = (cumsum - sorted_probs) > top_p
        sorted_logits = mx.where(mask, float("-inf"), sorted_logits)
        # Sample from sorted distribution, then map back to original vocab index
        probs = mx.softmax(sorted_logits, axis=-1)
        sampled_idx = mx.random.categorical(mx.log(probs + 1e-10))
        # Map sorted position back to original vocabulary position
        return mx.take_along_axis(
            sorted_indices, sampled_idx[..., None], axis=-1
        ).squeeze(-1)

    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))
