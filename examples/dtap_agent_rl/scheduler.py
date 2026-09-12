"""Worker-scoped FIFO attempt scheduler for parallel M4 episodes."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any


class SchedulerSaturated(RuntimeError):
    pass


class AttemptScheduler:
    def __init__(self, *, max_parallel: int, max_queued: int, wait_timeout: float) -> None:
        if isinstance(max_parallel, bool) or max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if isinstance(max_queued, bool) or max_queued < 1:
            raise ValueError("max_queued must be positive")
        if wait_timeout <= 0:
            raise ValueError("wait_timeout must be positive")
        self.max_parallel = max_parallel
        self.max_queued = max_queued
        self.wait_timeout = wait_timeout
        self._lock = asyncio.Lock()
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return len(self._waiters)

    async def _acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._active < self.max_parallel and not self._waiters:
                self._active += 1
                return
            if len(self._waiters) >= self.max_queued:
                raise SchedulerSaturated("attempt queue is full")
            waiter: asyncio.Future[None] = loop.create_future()
            self._waiters.append(waiter)
        granted = False
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout=self.wait_timeout)
            granted = True
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    waiter.cancel()
                elif waiter.done() and not waiter.cancelled():
                    granted = True
            if granted:
                await self._release()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise SchedulerSaturated("attempt queue wait timed out") from exc

    async def _release(self) -> None:
        async with self._lock:
            while self._waiters:
                waiter = self._waiters.popleft()
                if not waiter.done():
                    waiter.set_result(None)
                    return
            self._active -= 1
            if self._active < 0:
                self._active = 0
                raise RuntimeError("scheduler slot accounting underflow")

    @asynccontextmanager
    async def slot(self):
        await self._acquire()
        try:
            yield
        finally:
            await self._release()

    async def run(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        async with self.slot():
            return await operation()
