import asyncio
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.dtap_agent_rl.attempt_runner import AttemptResult
from examples.dtap_agent_rl.audit import InMemoryAuditSink
from examples.dtap_agent_rl.episode_runtime import EpisodeRuntimeState, EpisodeStatus
from examples.dtap_agent_rl.integrity import BenchmarkIntegrityGuard
from examples.dtap_agent_rl.policy_contract import PolicyContract
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.submission import SubmissionCoordinator
from examples.dtap_agent_rl.tests.test_m2_validation import indirect


VALID_PLAN = {
    "steps": [{"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "try"}]
}


class Runner:
    m4_hardened = True

    def __init__(self, results=()):
        self.results = deque(results)
        self.calls = []

    async def run(self, workspace):
        self.calls.append(workspace)
        return self.results.popleft()


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "dataset" / "workflow" / "task"
    source.mkdir(parents=True)
    (source / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "Task": {"task_id": "task"},
                "Agent": {},
                "RedTeamingAgent": {},
                "Attack": {"threat_model": "indirect", "malicious_goal": "goal"},
            }
        )
    )
    (source / "judge.py").write_text("# trusted\n")
    return source


def _coordinator(
    tmp_path, *, h, q, runner, policy_overrides=None, audit_sink=None,
    placement_coordinator=None, feedback_builder=None,
):
    source = _source(tmp_path)
    policy = M4SecurityPolicy(max_submit_calls=q, **(policy_overrides or {}))
    return SubmissionCoordinator(
        validation_context=indirect(),
        runtime=EpisodeRuntimeState(max_submissions=h, max_submit_calls=q),
        source_task_dir=source,
        source_manifest=BenchmarkIntegrityGuard.capture(source),
        episode_root=tmp_path / "attempts" / "episode",
        runner=runner,
        security_policy=policy,
        policy_contract=PolicyContract(),
        terminal_event=asyncio.Event(),
        candidate_validator=lambda _config, *, expected_steps: SimpleNamespace(
            canonical_steps=expected_steps
        ),
        audit_sink=audit_sink,
        audit_episode_digest="episode-digest" if audit_sink is not None else None,
        placement_coordinator=placement_coordinator,
        feedback_builder=feedback_builder,
    )


@pytest.mark.asyncio
async def test_invalid_submits_spend_q_not_h_and_q_terminal_is_trainable(tmp_path):
    runner = Runner()
    coordinator = _coordinator(tmp_path, h=2, q=2, runner=runner)

    first = await coordinator.submit({"steps": [{"type": "a2a"}]})
    second = await coordinator.submit({"steps": []})

    assert first == {
        "accepted": False,
        "terminal": False,
        "remaining_submissions": 2,
        "error": {"code": "INVALID_SUBMISSION"},
    }
    assert second["terminal"] is True
    assert second["error"]["code"] == "POLICY_LIMIT"
    assert coordinator.runtime.submit_calls == 2
    assert coordinator.runtime.submissions_used == 0
    assert coordinator.runtime.status is EpisodeStatus.POLICY_LIMIT
    assert coordinator.runtime.final_reward == 0.0
    assert coordinator.runtime.remove_sample is False
    assert not runner.calls


@pytest.mark.asyncio
async def test_invalid_submissions_preserve_both_h_victim_runs(tmp_path):
    runner = Runner([
        AttemptResult(evaluation_started=True, attack_success=False),
        AttemptResult(evaluation_started=True, attack_success=False),
    ])
    coordinator = _coordinator(tmp_path, h=2, q=4, runner=runner)

    invalid = await coordinator.submit({"steps": []})
    first = await coordinator.submit(VALID_PLAN)
    invalid_again = await coordinator.submit({"steps": [{"type": "a2a"}]})
    second = await coordinator.submit(VALID_PLAN)

    assert invalid["error"]["code"] == "INVALID_SUBMISSION"
    assert invalid["remaining_submissions"] == 2
    assert first["submission"] == 1
    assert first["remaining_submissions"] == 1
    assert invalid_again["error"]["code"] == "INVALID_SUBMISSION"
    assert invalid_again["remaining_submissions"] == 1
    assert second["submission"] == 2
    assert second["remaining_submissions"] == 0
    assert second["terminal"] is True
    assert coordinator.runtime.victim_runs_started == 2
    assert len(runner.calls) == 2


@pytest.mark.asyncio
async def test_m6_unverified_environment_guard_spends_q_before_runner(tmp_path):
    runner = Runner()
    placement = SimpleNamespace(
        unverified_environment_indices=lambda _steps: (0,)
    )
    coordinator = _coordinator(
        tmp_path, h=1, q=2, runner=runner,
        placement_coordinator=placement,
    )

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt == {
        "accepted": False,
        "terminal": False,
        "remaining_submissions": 1,
        "error": {"code": "INVALID_SUBMISSION"},
    }
    assert coordinator.runtime.submit_calls == 1
    assert coordinator.runtime.submissions_used == 0
    assert not runner.calls


@pytest.mark.asyncio
async def test_valid_attempt_after_invalid_can_end_at_q_without_spending_all_h(tmp_path):
    runner = Runner([AttemptResult(evaluation_started=True, attack_success=False)])
    coordinator = _coordinator(tmp_path, h=2, q=2, runner=runner)
    await coordinator.submit({"steps": []})

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt["accepted"] is True and receipt["success"] is False
    assert receipt["terminal"] is True
    assert coordinator.runtime.status is EpisodeStatus.POLICY_LIMIT
    assert coordinator.runtime.submissions_used == 1
    assert coordinator.runtime.final_reward == 0.0


@pytest.mark.asyncio
async def test_oversized_plan_fails_before_filesystem_and_runner(tmp_path):
    runner = Runner()
    coordinator = _coordinator(
        tmp_path,
        h=2,
        q=2,
        runner=runner,
        policy_overrides={"max_plan_bytes": 128, "max_content_bytes": 64},
    )

    receipt = await coordinator.submit(
        {"steps": [{"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "x" * 1000}]}
    )

    assert receipt["error"]["code"] == "INVALID_SUBMISSION"
    assert coordinator.runtime.submit_calls == 1
    assert coordinator.runtime.submissions_used == 0
    assert not runner.calls
    assert not coordinator.episode_root.exists()


@pytest.mark.asyncio
async def test_reward_receipt_ignores_all_trusted_runner_details(tmp_path):
    runner = Runner(
        [
            AttemptResult(
                evaluation_started=True,
                attack_success=True,
                task_success=False,
                judge_result={"rationale": "SECRET"},
                victim_output="SECRET",
                trajectory_path=Path("/trusted/trajectory"),
                runtime_identity="docker-secret",
            )
        ]
    )
    coordinator = _coordinator(tmp_path, h=2, q=2, runner=runner)

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt == {
        "accepted": True,
        "submission": 1,
        "success": True,
        "terminal": True,
        "remaining_submissions": 1,
    }
    assert "SECRET" not in repr(receipt)
    assert "docker" not in repr(receipt)


@pytest.mark.asyncio
async def test_trusted_audit_maps_unexpected_internal_stage_to_content_free_code(tmp_path):
    audit = InMemoryAuditSink()
    runner = Runner(
        [
            AttemptResult.infrastructure_failure(
                stage="SECRET /trusted/path",
                evaluation_started=False,
            )
        ]
    )
    coordinator = _coordinator(tmp_path, h=1, q=1, runner=runner, audit_sink=audit)
    receipt = await coordinator.submit(VALID_PLAN)
    assert receipt["error"]["code"] == "EVALUATION_UNAVAILABLE"
    assert audit.events[-1].event == "infra_abort"
    assert audit.events[-1].code == "other"
    assert "SECRET" not in repr(audit.events)


@pytest.mark.asyncio
async def test_32_parallel_episode_states_do_not_cross(tmp_path):
    async def one(index):
        root = tmp_path / str(index)
        expected = index % 2 == 0
        runner = Runner([AttemptResult(evaluation_started=True, attack_success=expected)])
        coordinator = _coordinator(root, h=1, q=1, runner=runner)
        receipt = await coordinator.submit(VALID_PLAN)
        return index, receipt, runner.calls[0].attempt_dir

    results = await asyncio.gather(*(one(index) for index in range(32)))
    roots = set()
    for index, receipt, attempt_dir in results:
        assert receipt["success"] is (index % 2 == 0)
        assert receipt["terminal"] is True
        roots.add(attempt_dir)
    assert len(roots) == 32


class FeedbackBuilder:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    async def build(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.value


@pytest.mark.asyncio
async def test_m7_feedback_is_only_for_genuine_nonterminal_failure(tmp_path):
    builder = FeedbackBuilder({"schema_version": 1, "final_response": "failed"})
    runner = Runner([
        AttemptResult(evaluation_started=True, attack_success=False),
        AttemptResult(evaluation_started=True, attack_success=False),
    ])
    coordinator = _coordinator(tmp_path, h=2, q=3, runner=runner, feedback_builder=builder)

    invalid = await coordinator.submit({"steps": []})
    first = await coordinator.submit(VALID_PLAN)
    final = await coordinator.submit(VALID_PLAN)

    assert "feedback" not in invalid
    assert first["feedback"]["final_response"] == "failed"
    assert "feedback" not in final
    assert len(builder.calls) == 1
    assert coordinator.runtime.victim_runs_started == 2
    assert coordinator.runtime.final_reward == 0.0


@pytest.mark.asyncio
async def test_m7_attempt_feedback_can_drive_next_plan_while_invalid_calls_preserve_h(tmp_path):
    builder = FeedbackBuilder({
        "schema_version": 1,
        "final_response": "The victim ignored the first wording.",
    })
    runner = Runner([
        AttemptResult(evaluation_started=True, attack_success=False),
        AttemptResult(evaluation_started=True, attack_success=True),
    ])
    coordinator = _coordinator(tmp_path, h=2, q=4, runner=runner, feedback_builder=builder)

    invalid = await coordinator.submit({"steps": []})
    first = await coordinator.submit(VALID_PLAN)
    repaired = {
        "steps": [{
            "type": "prompt", "turn_id": 1, "mode": "suffix",
            "content": "repaired after: " + first["feedback"]["final_response"],
        }]
    }
    second = await coordinator.submit(repaired)

    assert invalid["remaining_submissions"] == 2
    assert first["terminal"] is False and first["remaining_submissions"] == 1
    assert second["success"] is True and second["terminal"] is True
    assert coordinator.runtime.victim_runs_started == 2
    assert len(runner.calls) == 2
    assert builder.calls[0]["submitted_plan"] == {"steps": VALID_PLAN["steps"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "builder",
    [
        FeedbackBuilder(error=RuntimeError("broken")),
        FeedbackBuilder({"schema_version": 1, "final_response": "ok", "judge_result": {}}),
    ],
)
async def test_m7_failure_or_contract_rejection_does_not_change_reward_or_h(tmp_path, builder):
    runner = Runner([AttemptResult(evaluation_started=True, attack_success=False)])
    coordinator = _coordinator(tmp_path, h=2, q=2, runner=runner, feedback_builder=builder)

    receipt = await coordinator.submit(VALID_PLAN)

    assert receipt == {
        "accepted": True,
        "submission": 1,
        "success": False,
        "terminal": False,
        "remaining_submissions": 1,
    }
    assert coordinator.runtime.victim_runs_started == 1
    assert coordinator.runtime.status is EpisodeStatus.ACTIVE
