from contextlib import asynccontextmanager
from types import SimpleNamespace

import examples.dtap_agent_rl.generate_m4 as generate_module
import pytest
from examples.dtap_agent_rl.authority import EpisodeAuthorityRegistry
from examples.dtap_agent_rl.episode_runtime import EpisodeRuntimeState
from examples.dtap_agent_rl.generate_m4 import M4GenerateRuntime
from examples.dtap_agent_rl.sandbox_policy import SandboxPolicyVerifier
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.training_record import TrainingRecordStore
from slime.utils.types import Sample as SlimeSample


class Sample:
    def __init__(self):
        self.session_id = None
        self.metadata = {
            "task_dir": "/trusted/task",
            "prompt": "attack",
            "workdir": "/workspace",
            "sample_id": "safe-id",
            "SECRET": "drop-me",
        }
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


class Runner:
    m4_hardened = True


@asynccontextmanager
async def sandbox_factory(_sample):
    yield object()


def _runtime(adapter):
    return M4GenerateRuntime(
        adapter=adapter,
        adapter_url="http://adapter",
        policy_mcp_url="http://mcp",
        catalog_provider=object(),
        runner=Runner(),
        sandbox_factory=sandbox_factory,
        sandbox_verifier=SandboxPolicyVerifier(),
        attempts_root="/tmp/attempts",
        max_submissions=2,
        security_policy=M4SecurityPolicy(max_submit_calls=3),
        authority_registry=EpisodeAuthorityRegistry(),
        snapshot_loader=lambda _path: object(),
    )


@pytest.mark.asyncio
async def test_m4_generate_keeps_one_adapter_session_and_allowlists_metadata(monkeypatch):
    adapter = Adapter()
    runtime = _runtime(adapter)
    state = EpisodeRuntimeState(max_submissions=2, max_submit_calls=3)
    state.begin_submit_call()
    attempt = state.mark_evaluation_started()
    state.record_attack_result(attempt_index=attempt, attack_success=True)
    monkeypatch.setattr(generate_module, "_RUNTIME", runtime)
    monkeypatch.setattr(
        generate_module,
        "run_m4_episode",
        lambda **_kwargs: _value(SimpleNamespace(runtime=state, public_episode_id="public", harness_return_code=0)),
    )
    sample = Sample()

    results = await generate_module.generate(None, sample, {"temperature": 1.0})

    assert len(adapter.opened) == len(adapter.finished) == len(adapter.dropped) == 1
    assert results[0].reward == 1.0
    assert results[0].remove_sample is False
    assert results[0].metadata["sample_id"] == "safe-id"
    assert "task_dir" not in results[0].metadata
    assert "SECRET" not in results[0].metadata
    assert "agent_exit_code" not in results[0].metadata


@pytest.mark.asyncio
async def test_m4_generate_exception_has_stable_content_free_abort(monkeypatch):
    adapter = Adapter()
    runtime = _runtime(adapter)
    monkeypatch.setattr(generate_module, "_RUNTIME", runtime)

    async def fail(**_kwargs):
        raise RuntimeError("SECRET /trusted/path")

    monkeypatch.setattr(generate_module, "run_m4_episode", fail)
    sample = Sample()
    results = await generate_module.generate(None, sample, {})

    assert len(adapter.opened) == len(adapter.finished) == len(adapter.dropped) == 1
    assert results[0].remove_sample is True
    assert results[0].metadata["abort_reason"] == "evaluation_unavailable"
    assert "SECRET" not in repr(results[0].metadata)
    assert "/trusted" not in repr(results[0].metadata)


@pytest.mark.asyncio
async def test_m4_generate_exception_persists_observable_excluded_record(monkeypatch, tmp_path):
    adapter = Adapter()
    runtime = _runtime(adapter)
    runtime.training_record_store = TrainingRecordStore(tmp_path / "records")
    runtime.training_run_id = "m8-infra"
    monkeypatch.setattr(generate_module, "_RUNTIME", runtime)

    async def fail(**_kwargs):
        raise RuntimeError("private diagnostic")

    monkeypatch.setattr(generate_module, "run_m4_episode", fail)
    results = await generate_module.generate(None, Sample(), {})

    records = list((tmp_path / "records" / "excluded").glob("*.json"))
    assert results[0].remove_sample is True
    assert len(records) == 1
    payload = records[0].read_text()
    assert '"eligible":false' in payload
    assert "private diagnostic" not in payload


@pytest.mark.asyncio
async def test_m4_generate_persists_and_resumes_reward_zero_training_samples(monkeypatch, tmp_path):
    adapter = Adapter()
    runtime = _runtime(adapter)
    runtime.training_record_store = TrainingRecordStore(tmp_path / "records")
    runtime.training_run_id = "m8-run"
    runtime.rollout_seed = 11
    state = EpisodeRuntimeState(max_submissions=2, max_submit_calls=3)
    state.begin_submit_call()
    first = state.mark_evaluation_started()
    state.record_attack_result(attempt_index=first, attack_success=False)
    state.begin_submit_call()
    second = state.mark_evaluation_started()
    state.record_attack_result(attempt_index=second, attack_success=False)
    monkeypatch.setattr(generate_module, "_RUNTIME", runtime)
    monkeypatch.setattr(
        generate_module,
        "run_m4_episode",
        lambda **_kwargs: _value(SimpleNamespace(runtime=state, public_episode_id="public", harness_return_code=0)),
    )
    sample = SlimeSample(
        index=4,
        rollout_id=4,
        metadata={"task_dir": "/trusted/task", "task_id": "task-4", "sample_id": "sample-4"},
        tokens=[1, 2],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.2],
    )

    first_results = await generate_module.generate(None, sample, {"temperature": 1.0})
    assert first_results[0].reward == 0.0
    assert len(list((tmp_path / "records" / "eligible").glob("*.json"))) == 1

    resumed = SlimeSample(
        index=4,
        rollout_id=4,
        metadata={"task_dir": "/different/host/task", "task_id": "task-4", "sample_id": "sample-4"},
    )
    second_results = await generate_module.generate(None, resumed, {"temperature": 1.0})
    assert second_results[0].tokens == [1, 2]
    assert len(adapter.opened) == 1


def test_m4_runtime_configuration_rejects_non_hardened_runner():
    adapter = Adapter()
    runtime = _runtime(adapter)
    runtime.runner = object()
    with pytest.raises(ValueError, match="hardened"):
        generate_module.configure_runtime(runtime)


async def _value(value):
    return value
