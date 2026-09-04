import asyncio

import pytest

from examples.dtap_agent_rl.scheduler import AttemptScheduler, SchedulerSaturated


@pytest.mark.asyncio
async def test_scheduler_is_fifo_and_globally_bounded():
    scheduler = AttemptScheduler(max_parallel=1, max_queued=4, wait_timeout=1)
    gate = asyncio.Event()
    order = []

    async def operation(index):
        order.append(index)
        if index == 0:
            await gate.wait()
        await asyncio.sleep(0)
        return index

    tasks = [asyncio.create_task(scheduler.run(lambda i=i: operation(i))) for i in range(4)]
    await asyncio.sleep(0)
    assert scheduler.active == 1
    assert scheduler.queued == 3
    gate.set()
    assert await asyncio.gather(*tasks) == [0, 1, 2, 3]
    assert order == [0, 1, 2, 3]
    assert scheduler.active == 0 and scheduler.queued == 0


@pytest.mark.asyncio
async def test_scheduler_saturation_is_pre_start_failure():
    scheduler = AttemptScheduler(max_parallel=1, max_queued=1, wait_timeout=1)
    gate = asyncio.Event()

    async def blocked():
        await gate.wait()

    first = asyncio.create_task(scheduler.run(blocked))
    await asyncio.sleep(0)
    second = asyncio.create_task(scheduler.run(blocked))
    await asyncio.sleep(0)
    with pytest.raises(SchedulerSaturated):
        await scheduler.run(blocked)
    gate.set()
    await asyncio.gather(first, second)
