from pathlib import Path

import pytest

from examples.dtap_agent_rl.attempt_runner import DtapAttemptRunner
from examples.dtap_agent_rl.candidate_config import AttemptWorkspace


def _workspace(tmp_path: Path) -> AttemptWorkspace:
    attempt = tmp_path / "attempt-0001"
    task = attempt / "dataset" / "workflow" / "task-1"
    task.mkdir(parents=True)
    return AttemptWorkspace(1, attempt, task, task / "config.yaml", attempt / "results")


def _executable(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake-dtap"
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.asyncio
async def test_runner_uses_judge_attack_success_not_process_return_code(tmp_path):
    executable = _executable(
        tmp_path,
        """
mkdir -p "$EVAL_RESULTS_ROOT/benchmark/fake"
printf '%s\n' '{"attack_success": true, "task_success": false}' > "$EVAL_RESULTS_ROOT/benchmark/fake/judge_result.json"
printf '%s\n' '[DTAP_STATUS] phase=running'
exit 1
""",
    )
    runner = DtapAttemptRunner(python_executable=str(executable), model="fake")

    result = await runner.run(_workspace(tmp_path))

    assert result.evaluation_started is True
    assert result.attack_success is True
    assert result.task_success is False
    assert result.runtime_destroyed is True
    assert result.infrastructure_stage is None


@pytest.mark.asyncio
async def test_runner_missing_judge_is_infrastructure_error_not_attack_failure(tmp_path):
    executable = _executable(
        tmp_path,
        "printf '%s\\n' '[DTAP_STATUS] phase=running'\nexit 0\n",
    )
    runner = DtapAttemptRunner(python_executable=str(executable), model="fake")

    result = await runner.run(_workspace(tmp_path))

    assert result.evaluation_started is True
    assert result.attack_success is None
    assert result.is_infrastructure_failure is True
    assert result.infrastructure_stage == "judge_result"


@pytest.mark.asyncio
async def test_runner_retains_bounded_redacted_stderr_for_infrastructure_failure(
    tmp_path,
):
    executable = _executable(
        tmp_path,
        'printf "%s\\n" "diagnostic=$DIAGNOSTIC_SECRET" >&2\nexit 7\n',
    )
    runner = DtapAttemptRunner(
        python_executable=str(executable),
        model="fake",
        extra_env={"DIAGNOSTIC_SECRET": "do-not-retain-this-value"},
    )
    workspace = _workspace(tmp_path)

    result = await runner.run(workspace)
    diagnostic = (workspace.output_root / ".dtap-stderr.log").read_text(encoding="utf-8")

    assert result.infrastructure_stage == "judge_result"
    assert "do-not-retain-this-value" not in diagnostic
    assert "diagnostic=<redacted>" in diagnostic


@pytest.mark.asyncio
async def test_runner_timeout_kills_process_and_returns_no_reward_verdict(tmp_path):
    executable = _executable(tmp_path, "sleep 10\n")
    runner = DtapAttemptRunner(python_executable=str(executable), model="fake", timeout_seconds=0.01)

    result = await runner.run(_workspace(tmp_path))

    assert result.evaluation_started is False
    assert result.attack_success is None
    assert result.infrastructure_stage == "timeout"
    assert result.runtime_destroyed is True
