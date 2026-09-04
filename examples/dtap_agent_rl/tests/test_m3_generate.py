from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import examples.dtap_agent_rl.generate as generate_module
from examples.dtap_agent_rl.episode_runtime import EpisodeRuntimeState, EpisodeStatus
from examples.dtap_agent_rl.generate import M3GenerateRuntime
from examples.dtap_agent_rl.service import EpisodeRegistry
from examples.dtap_agent_rl.submission import EpisodeSubmissionRegistry


class Sample:
    def __init__(self):
        self.session_id = None
        self.metadata = {"task_dir": "/trusted/task", "prompt": "attack", "workdir": "/workspace"}
        self.prompt = ""
        self.reward = None
        self.remove_sample = False


class Adapter:
    def __init__(self):
        self.opened = []
        self.finished = []
        self.dropped = []

    def open_session(self, session_id, **kwargs):
        self.opened.append((session_id, kwargs))

    async def finish_session(self, session_id, **kwargs):
        self.finished.append((session_id, kwargs))
        kwargs["base_sample"].reward = kwargs["reward"]
        return [kwargs["base_sample"]]

    async def drop_session(self, session_id, *, wait_timeout):
        self.dropped.append((session_id, wait_timeout))


@asynccontextmanager
async def sandbox_factory(_sample):
    yield object()


def _runtime(adapter):
    return M3GenerateRuntime(
        adapter=adapter,
        adapter_url="http://adapter",
        catalog_provider=object(),
        runner=object(),
        sandbox_factory=sandbox_factory,
        attempts_root="/tmp/attempts",
        max_submissions=3,
        view_registry=EpisodeRegistry(),
        submission_registry=EpisodeSubmissionRegistry(),
        snapshot_loader=lambda _task_dir: object(),
    )


@pytest.mark.asyncio
async def test_generate_uses_one_real_slime_adapter_session(monkeypatch):
    adapter = Adapter()
    runtime = _runtime(adapter)
    state = EpisodeRuntimeState(max_submissions=3)
    attempt = state.mark_evaluation_started()
    state.record_attack_result(attempt_index=attempt, attack_success=True)
    monkeypatch.setattr(generate_module, "_RUNTIME", runtime)
    monkeypatch.setattr(
        generate_module,
        "run_m3_episode",
        lambda **_kwargs: _async_result(SimpleNamespace(
            runtime=state, harness_return_code=0
        )),
    )
    sample = Sample()

    results = await generate_module.generate(None, sample, {"temperature": 1.0})

    assert len(adapter.opened) == 1
    assert len(adapter.finished) == 1
    assert len(adapter.dropped) == 1
    assert adapter.finished[0][1]["reward"] == 1.0
    assert results == [sample]
    assert sample.remove_sample is False


@pytest.mark.asyncio
async def test_generate_marks_infrastructure_trajectory_for_removal(monkeypatch):
    adapter = Adapter()
    runtime = _runtime(adapter)
    state = EpisodeRuntimeState(max_submissions=3)
    state.record_infrastructure_failure(stage="judge")
    monkeypatch.setattr(generate_module, "_RUNTIME", runtime)
    monkeypatch.setattr(
        generate_module,
        "run_m3_episode",
        lambda **_kwargs: _async_result(SimpleNamespace(
            runtime=state, harness_return_code=1
        )),
    )
    sample = Sample()

    results = await generate_module.generate(None, sample, {})

    assert adapter.finished[0][1]["reward"] == 0.0
    assert results[0].remove_sample is True
    assert results[0].metadata["abort_reason"] == "dtap_infrastructure_error"
    assert adapter.dropped == [(sample.session_id, 30)]


@pytest.mark.asyncio
async def test_generate_exception_after_open_still_finishes_exactly_once(monkeypatch):
    adapter = Adapter()
    runtime = _runtime(adapter)
    monkeypatch.setattr(generate_module, "_RUNTIME", runtime)

    async def fail(**_kwargs):
        raise RuntimeError("broken harness")

    monkeypatch.setattr(generate_module, "run_m3_episode", fail)
    sample = Sample()

    results = await generate_module.generate(None, sample, {})

    assert len(adapter.opened) == 1
    assert len(adapter.finished) == 1
    assert len(adapter.dropped) == 1
    assert results[0].remove_sample is True
    assert results[0].metadata["abort_reason"] == "exception:RuntimeError"


async def _async_result(value):
    return value
