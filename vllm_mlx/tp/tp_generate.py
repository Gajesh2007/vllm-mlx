"""Synchronized TP generate loop.

Both ranks call model() in exact lockstep. After each step, rank 0
samples a token, broadcasts it to rank 1 via TCP. Both use the same
token for the next step. This prevents the decode loops from diverging.

This replaces mlx_lm.generate() for TP mode — we can't use the stock
generate because it has internal state (prefill chunking, cache management)
that can't be synchronized between independent processes.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("vllm_mlx.tp")


def tp_generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    rank: int = 0,
    sync_send_token=None,  # callable(int) — rank 0 sends token to rank 1
    sync_recv_token=None,  # callable() -> int — rank 1 receives token from rank 0
) -> str:
    """Generate text with synchronized TP.

    Both ranks must call this simultaneously with the same prompt.
    The model's all_sum operations synchronize the forward passes.
    After each decode step, rank 0 samples and sends the token to rank 1.

    Returns generated text (rank 0) or empty string (rank 1).
    """
    # Tokenize
    tokens = tokenizer.encode(prompt)
    prompt_tokens = mx.array([tokens])

    # Create KV cache
    cache = None
    if hasattr(model, "make_cache"):
        cache = model.make_cache()
    else:
        # Fallback: use mlx_lm's cache creation
        try:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
        except ImportError:
            pass

    # Get stop tokens
    stop_tokens = set()
    if hasattr(tokenizer, "eos_token_id"):
        eos = tokenizer.eos_token_id
        if isinstance(eos, int):
            stop_tokens.add(eos)
        elif isinstance(eos, list):
            stop_tokens.update(eos)

    generated_tokens = []
    t0 = time.perf_counter()

    # Verify all_sum patches are active
    inner = getattr(model, "language_model", model)
    inner = getattr(inner, "model", inner)
    if hasattr(inner, "layers") and len(inner.layers) > 0:
        has_tp = hasattr(inner.layers[0].self_attn, "_tp_group")
        grp = getattr(inner.layers[0].self_attn, "_tp_group", None)
        logger.info(f"TP check: _tp_group set={has_tp}, group={grp}")
        if not has_tp:
            logger.error("WARNING: _tp_group NOT set on attention layers! all_sum will not fire!")

    # Prefill: process entire prompt
    logits = model(prompt_tokens, cache=cache)
    mx.eval(logits)

    prefill_time = time.perf_counter() - t0
    logger.info(f"Prefill: {len(tokens)} tokens in {prefill_time:.2f}s")

    # Sample first token from prefill logits
    last_logits = logits[:, -1, :]

    if rank == 0:
        token_id = _sample(last_logits, temperature, top_p)
        if sync_send_token:
            sync_send_token(token_id)
    else:
        if sync_recv_token:
            token_id = sync_recv_token()
        else:
            token_id = _sample(last_logits, temperature, top_p)

    if token_id in stop_tokens:
        return "" if rank != 0 else tokenizer.decode([])

    generated_tokens.append(token_id)

    # Decode loop
    for i in range(max_tokens - 1):
        token_tensor = mx.array([[token_id]])

        logits = model(token_tensor, cache=cache)
        mx.eval(logits)

        last_logits = logits[:, -1, :]

        if rank == 0:
            token_id = _sample(last_logits, temperature, top_p)
            if sync_send_token:
                sync_send_token(token_id)
        else:
            if sync_recv_token:
                token_id = sync_recv_token()
            else:
                token_id = _sample(last_logits, temperature, top_p)

        if token_id in stop_tokens:
            break

        generated_tokens.append(token_id)

    elapsed = time.perf_counter() - t0
    tps = len(generated_tokens) / (elapsed - prefill_time) if elapsed > prefill_time else 0
    logger.info(
        f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s "
        f"({tps:.1f} tok/s, prefill={prefill_time:.2f}s)"
    )

    if rank == 0:
        return tokenizer.decode(generated_tokens)
    return ""


def _sample(logits: mx.array, temperature: float, top_p: float) -> int:
    """Sample a token from logits. Returns Python int."""
    if temperature <= 0:
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        return token.item()

    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    token = mx.random.categorical(mx.log(probs + 1e-10))
    mx.eval(token)
    return token.item()
