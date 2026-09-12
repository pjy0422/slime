from pathlib import Path
from unittest.mock import patch

import pytest
from examples.dtap_agent_rl.attempt_runner import DtapAttemptRunner
from examples.dtap_agent_rl.candidate_config import AttemptWorkspace
from examples.dtap_agent_rl.scheduler import AttemptScheduler
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy


def _workspace(tmp_path: Path) -> AttemptWorkspace:
    attempt = tmp_path / "attempt"
    task = attempt / "dataset" / "workflow" / "task"
    task.mkdir(parents=True)
    return AttemptWorkspace(1, attempt, task, task / "config.yaml", attempt / "results")


def _executable(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "m4-fake"
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _runner(executable: Path, policy: M4SecurityPolicy) -> DtapAttemptRunner:
    return DtapAttemptRunner(
        python_executable=str(executable),
        model="fake",
        security_policy=policy,
        scheduler=AttemptScheduler(
            max_parallel=policy.max_parallel_attempts,
            max_queued=policy.max_queued_attempts,
            wait_timeout=policy.queue_wait_timeout_seconds,
        ),
    )


@pytest.mark.asyncio
async def test_m4_runner_uses_exact_firewall_file_and_minimum_environment(tmp_path):
    executable = _executable(
        tmp_path,
        """
test -z "${HOST_SECRET+x}"
test "$DT_DISABLE_DEFAULT_PORTS" = "1"
test "$DT_PORT_RANGE_START" = "20000"
test "$DT_PORT_RANGE_END" = "20511"
mkdir -p "$EVAL_RESULTS_ROOT"
printf 1 > "$EVAL_RESULTS_ROOT/.m4-started"
printf '%s\n' '{"attack_success":true,"rationale":"SECRET"}' > "$EVAL_RESULTS_ROOT/.m4-verdict.json"
exit 9
""",
    )
    policy = M4SecurityPolicy(max_submit_calls=1)
    runner = _runner(executable, policy)

    with patch.dict("os.environ", {"HOST_SECRET": "must-not-leak"}, clear=False):
        result = await runner.run(_workspace(tmp_path))

    assert runner.m4_hardened is True
    assert result.evaluation_started is True
    assert result.attack_success is True
    assert result.judge_result is None
    assert result.task_success is None


@pytest.mark.asyncio
async def test_m4_runner_rejects_non_boolean_firewall_verdict(tmp_path):
    executable = _executable(
        tmp_path,
        """
mkdir -p "$EVAL_RESULTS_ROOT"
printf 1 > "$EVAL_RESULTS_ROOT/.m4-started"
printf '%s\n' '{"attack_success":"true"}' > "$EVAL_RESULTS_ROOT/.m4-verdict.json"
""",
    )
    policy = M4SecurityPolicy(max_submit_calls=1)
    runner = _runner(executable, policy)
    result = await runner.run(_workspace(tmp_path))
    assert result.evaluation_started is True
    assert result.attack_success is None
    assert result.infrastructure_stage == "judge_result"


@pytest.mark.asyncio
async def test_m4_runner_accepts_a_disjoint_worker_port_range(tmp_path):
    executable = _executable(
        tmp_path,
        """
test "$DT_PORT_RANGE_START" = "41000"
test "$DT_PORT_RANGE_END" = "41511"
mkdir -p "$EVAL_RESULTS_ROOT"
printf 1 > "$EVAL_RESULTS_ROOT/.m4-started"
printf '%s\n' '{"attack_success":false}' > "$EVAL_RESULTS_ROOT/.m4-verdict.json"
""",
    )
    policy = M4SecurityPolicy(max_submit_calls=1, max_queued_attempts=1)
    runner = DtapAttemptRunner(
        python_executable=str(executable),
        model="fake",
        security_policy=policy,
        scheduler=AttemptScheduler(max_parallel=1, max_queued=1, wait_timeout=30),
        port_range_start=41_000,
    )

    result = await runner.run(_workspace(tmp_path))

    assert result.evaluation_started is True
    assert result.attack_success is False
