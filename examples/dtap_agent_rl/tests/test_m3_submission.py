import asyncio
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.attempt_runner import AttemptResult
from examples.dtap_agent_rl.episode_runtime import EpisodeRuntimeState, EpisodeStatus
from examples.dtap_agent_rl.submission import SubmissionCoordinator
from examples.dtap_agent_rl.tests.test_m2_validation import indirect


VALID_PLAN = {
    "steps": [
        {"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "try this"}
    ]
}


class ScriptedRunner:
    def __init__(self, results, *, pause=None):
        self.results = deque(results)
        self.pause = pause
        self.calls = []

    async def run(self, workspace):
        self.calls.append(workspace)
        if self.pause is not None:
            await self.pause.wait()
        return self.results.popleft()


def _source_task(tmp_path: Path) -> Path:
    import yaml

    source = tmp_path / "source" / "dataset" / "workflow" / "task-7"
    source.mkdir(parents=True)
    (source / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "Task": {"task_id": "task-7"},
                "Agent": {},
                "RedTeamingAgent": {},
                "Attack": {
                    "threat_model": "indirect",
                    "malicious_goal": "goal",
                    "attack_turns": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def _coordinator(tmp_path, runner, h=3):
    return SubmissionCoordinator(
        validation_context=indirect(),
        runtime=EpisodeRuntimeState(max_submissions=h),
        source_task_dir=_source_task(tmp_path),
        episode_root=tmp_path / "attempts" / "episode-1",
        runner=runner,
        candidate_validator=lambda _config, *, expected_steps: SimpleNamespace(
            canonical_steps=expected_steps
        ),
    )


@pytest.mark.asyncio
async def test_invalid_plan_is_rejected_without_runner_or_h_consumption(tmp_path):
    runner = ScriptedRunner([])
    coordinator = _coordinator(tmp_path, runner, h=2)

    receipt = await coordinator.submit({"steps": [{"type": "a2a"}]})

    assert receipt == {
        "accepted": False,
        "terminal": False,
        "submissions_used": 0,
        "remaining_submissions": 2,
        "errors": [{
            "code": "UNSUPPORTED_IN_M2",
            "path": "steps[0].type",
            "message": "a2a injection is not supported in M2",
        }],
    }
    assert runner.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan", "code"),
    [
        ({"steps": [], "secret_override": True}, "UNKNOWN_FIELD"),
        ({"steps": []}, "EMPTY_PLAN"),
        ([], "INVALID_SHAPE"),
    ],
)
async def test_submission_envelope_is_strict_and_fail_closed(tmp_path, plan, code):
    runner = ScriptedRunner([])
    coordinator = _coordinator(tmp_path, runner, h=2)

    receipt = await coordinator.submit(plan)

    assert receipt["accepted"] is False
    assert receipt["errors"][0]["code"] == code
    assert coordinator.runtime.submissions_used == 0
    assert runner.calls == []


@pytest.mark.asyncio
async def test_yaml_gate_failure_never_calls_runner_or_consumes_h(tmp_path, monkeypatch):
    from examples.dtap_agent_rl.candidate_config import CandidateConfigError

    runner = ScriptedRunner([])
    coordinator = _coordinator(tmp_path, runner, h=2)
    monkeypatch.setattr(
        "examples.dtap_agent_rl.submission.materialize_attempt_dir",
        lambda **_kwargs: (_ for _ in ()).throw(CandidateConfigError("semantic mismatch")),
    )

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt["accepted"] is False
    assert receipt["errors"] == [{
        "code": "YAML_SCHEMA_MISMATCH",
        "message": "candidate config failed validation",
    }]
    assert coordinator.runtime.submissions_used == 0
    assert runner.calls == []


@pytest.mark.asyncio
async def test_normal_failure_returns_one_bit_and_keeps_policy_episode_active(tmp_path):
    runner = ScriptedRunner([AttemptResult(evaluation_started=True, attack_success=False)])
    coordinator = _coordinator(tmp_path, runner, h=2)

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt == {
        "accepted": True,
        "submission": 1,
        "success": False,
        "terminal": False,
        "remaining_submissions": 1,
    }
    assert coordinator.runtime.status is EpisodeStatus.ACTIVE


@pytest.mark.asyncio
async def test_success_is_immediate_terminal_and_reward_one(tmp_path):
    runner = ScriptedRunner([AttemptResult(evaluation_started=True, attack_success=True)])
    coordinator = _coordinator(tmp_path, runner, h=3)

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt == {
        "accepted": True,
        "submission": 1,
        "success": True,
        "terminal": True,
        "remaining_submissions": 2,
    }
    assert coordinator.runtime.final_reward == 1.0

    rejected = await coordinator.submit(VALID_PLAN)
    assert rejected["accepted"] is False
    assert rejected["errors"][0]["code"] == "EPISODE_TERMINAL"
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_hth_failure_is_terminal_with_reward_zero(tmp_path):
    runner = ScriptedRunner([
        AttemptResult(evaluation_started=True, attack_success=False),
        AttemptResult(evaluation_started=True, attack_success=False),
    ])
    coordinator = _coordinator(tmp_path, runner, h=2)

    first = await coordinator.submit(VALID_PLAN)
    second = await coordinator.submit(VALID_PLAN)

    assert first["terminal"] is False
    assert second == {
        "accepted": True,
        "submission": 2,
        "success": False,
        "terminal": True,
        "remaining_submissions": 0,
    }
    assert coordinator.runtime.status is EpisodeStatus.EXHAUSTED
    assert coordinator.runtime.final_reward == 0.0


@pytest.mark.asyncio
async def test_pre_start_infrastructure_failure_does_not_become_reward_zero(tmp_path):
    runner = ScriptedRunner([
        AttemptResult.infrastructure_failure(stage="docker_start", evaluation_started=False)
    ])
    coordinator = _coordinator(tmp_path, runner, h=2)

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt["accepted"] is False
    assert receipt["terminal"] is True
    assert receipt["errors"] == [{"code": "INFRA_ERROR", "message": "evaluation unavailable"}]
    assert coordinator.runtime.submissions_used == 0
    assert coordinator.runtime.remove_sample is True
    assert coordinator.runtime.final_reward is None


@pytest.mark.asyncio
async def test_post_start_infrastructure_failure_records_started_h_but_removes_sample(tmp_path):
    runner = ScriptedRunner([
        AttemptResult.infrastructure_failure(stage="judge", evaluation_started=True)
    ])
    coordinator = _coordinator(tmp_path, runner, h=2)

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt["accepted"] is False
    assert receipt["terminal"] is True
    assert coordinator.runtime.submissions_used == 1
    assert coordinator.runtime.remove_sample is True
    assert coordinator.runtime.final_reward is None


@pytest.mark.asyncio
async def test_policy_receipt_never_leaks_trusted_runner_data(tmp_path):
    runner = ScriptedRunner([
        AttemptResult(
            evaluation_started=True,
            attack_success=False,
            task_success=True,
            judge_result={"rationale": "SECRET", "attack_success": False},
            victim_output="SECRET VICTIM OUTPUT",
            trajectory_path=Path("/trusted/trajectory.json"),
        )
    ])
    coordinator = _coordinator(tmp_path, runner, h=2)

    receipt = await coordinator.submit(VALID_PLAN)
    serialized = repr(receipt)

    assert receipt["success"] is False
    assert "SECRET" not in serialized
    assert "trajectory" not in serialized
    assert "task_success" not in serialized
    assert "/trusted" not in serialized


@pytest.mark.asyncio
async def test_episode_lock_prevents_two_parallel_calls_from_spending_same_h(tmp_path):
    pause = asyncio.Event()
    runner = ScriptedRunner(
        [AttemptResult(evaluation_started=True, attack_success=False)], pause=pause
    )
    coordinator = _coordinator(tmp_path, runner, h=1)

    first = asyncio.create_task(coordinator.submit(VALID_PLAN))
    await asyncio.sleep(0)
    second = asyncio.create_task(coordinator.submit(VALID_PLAN))
    await asyncio.sleep(0)
    pause.set()
    receipts = await asyncio.gather(first, second)

    assert sum(receipt["accepted"] is True for receipt in receipts) == 1
    assert sum(receipt["errors"][0]["code"] == "EPISODE_TERMINAL"
               for receipt in receipts if not receipt["accepted"]) == 1
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_each_accepted_submission_gets_a_fresh_attempt_directory(tmp_path):
    runner = ScriptedRunner([
        AttemptResult(evaluation_started=True, attack_success=False),
        AttemptResult(evaluation_started=True, attack_success=True),
    ])
    coordinator = _coordinator(tmp_path, runner, h=2)

    await coordinator.submit(VALID_PLAN)
    await coordinator.submit(VALID_PLAN)

    assert len(runner.calls) == 2
    assert runner.calls[0].task_dir != runner.calls[1].task_dir
    assert runner.calls[0].task_dir.name == runner.calls[1].task_dir.name
    assert runner.calls[0].task_dir.is_relative_to(tmp_path)
    assert runner.calls[1].task_dir.is_relative_to(tmp_path)
