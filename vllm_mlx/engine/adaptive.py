# SPDX-License-Identifier: Apache-2.0
"""
Adaptive engine: SimpleEngine throughput with concurrent request queuing.

Uses SimpleEngine for maximum single-request decode speed (direct mlx-lm
calls, zero scheduler overhead). When multiple requests arrive concurrently,
they are queued and processed sequentially rather than batched, because
the per-token scheduler overhead of BatchedEngine typically outweighs the
batching gains for large models (27B+) at low concurrency.

For high-concurrency workloads (sustained 4+ concurrent requests), use
BatchedEngine directly via --continuous-batching.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .base import BaseEngine, GenerationOutput
from .simple import SimpleEngine

logger = logging.getLogger(__name__)


class AdaptiveEngine(BaseEngine):
    """
    Wraps SimpleEngine with a concurrency-aware request queue.

    - Single request: routed directly to SimpleEngine (maximum throughput)
    - Concurrent requests: queued and processed sequentially (no batching overhead)
    - Tracks concurrency metrics for observability
    """

    def __init__(self, simple_engine: SimpleEngine):
        self._engine = simple_engine
        self._queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue()
        self._active_requests = 0
        self._total_requests = 0
        self._total_queued = 0
        self._lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._engine.model_name

    @property
    def is_mllm(self) -> bool:
        return self._engine.is_mllm

    @property
    def tokenizer(self) -> Any:
        return self._engine.tokenizer

    async def start(self) -> None:
        await self._engine.start()

    async def stop(self) -> None:
        await self._engine.stop()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        async with self._request_slot():
            return await self._engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        async with self._request_slot():
            async for output in self._engine.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ):
                yield output

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        async with self._request_slot():
            return await self._engine.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        async with self._request_slot():
            async for output in self._engine.stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            ):
                yield output

    class _request_slot:
        """Async context manager that serializes access to the engine."""

        def __init__(self_slot):
            self_slot._engine_ref = None

        async def __aenter__(self_slot):
            # Will be bound by AdaptiveEngine.__init__ closure
            return self_slot

        async def __aexit__(self_slot, *args):
            pass

    def _request_slot(self):
        return _AdaptiveSlot(self)

    def get_stats(self) -> dict[str, Any]:
        stats = self._engine.get_stats()
        stats["engine_type"] = "adaptive"
        stats["active_requests"] = self._active_requests
        stats["total_requests"] = self._total_requests
        stats["total_queued"] = self._total_queued
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        return self._engine.get_cache_stats()


class _AdaptiveSlot:
    """
    Async context manager that serializes request execution.

    If the engine is idle, the request proceeds immediately.
    If another request is in flight, this request waits in a FIFO queue.
    """

    def __init__(self, adaptive: AdaptiveEngine):
        self._adaptive = adaptive
        self._event: asyncio.Event | None = None
        self._wait_start: float = 0

    async def __aenter__(self):
        engine = self._adaptive
        async with engine._lock:
            engine._total_requests += 1
            if engine._active_requests == 0:
                engine._active_requests = 1
                logger.debug("adaptive: request proceeding immediately")
                return self

            engine._total_queued += 1
            self._event = asyncio.Event()
            self._wait_start = time.monotonic()
            logger.info(
                "adaptive: request queued (active=%d, queued=%d)",
                engine._active_requests,
                engine._queue.qsize() + 1,
            )
            await engine._queue.put(self)

        await self._event.wait()
        wait_ms = (time.monotonic() - self._wait_start) * 1000
        logger.info("adaptive: queued request proceeding after %.0fms wait", wait_ms)
        return self

    async def __aexit__(self, *args):
        engine = self._adaptive
        async with engine._lock:
            if not engine._queue.empty():
                next_slot = await engine._queue.get()
                next_slot._event.set()
            else:
                engine._active_requests = 0
