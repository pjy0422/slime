import asyncio
import os
from pathlib import Path

import pytest

from examples.dtap_agent_rl.actions import ValidatedAttackStep
from examples.dtap_agent_rl.attempt_runner import DtapAttemptRunner
from examples.dtap_agent_rl.candidate_config import materialize_attempt_dir
from examples.dtap_agent_rl.integrity import BenchmarkIntegrityGuard
from examples.dtap_agent_rl.scheduler import AttemptScheduler
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_real_m4_dtap_attempts_are_parallel_and_firewalled(tmp_path):
    task_dir = os.environ.get("DTAP_M4_PARALLEL_TASK_DIR")
    if not task_dir:
        pytest.skip("set DTAP_M4_PARALLEL_TASK_DIR for the real M4 parallel gate")
    source = Path(task_dir).expanduser().resolve()
    manifest = BenchmarkIntegrityGuard.capture(source)
    inherited = tuple(
        name.strip()
        for name in os.environ.get(
            "DTAP_M4_INHERITED_ENV_NAMES",
            "PYTHONPATH,OPENAI_API_KEY,ANTHROPIC_API_KEY,GOOGLE_API_KEY",
        ).split(",")
        if name.strip()
    )
    policy = M4SecurityPolicy(
        max_submit_calls=2,
        max_parallel_attempts=2,
        max_queued_attempts=2,
        inherited_dtap_env_names=inherited,
    )
    scheduler = AttemptScheduler(max_parallel=2, max_queued=2, wait_timeout=300)
    runner = DtapAttemptRunner(
        model=os.environ.get("DTAP_M4_MODEL", "gpt-5.4"),
        agent_type=os.environ.get("DTAP_M4_AGENT_TYPE", "openaisdk"),
        dtap_root=os.environ.get("DTAP_M4_ROOT"),
        timeout_seconds=float(os.environ.get("DTAP_M4_TIMEOUT", "1800")),
        security_policy=policy,
        scheduler=scheduler,
    )
    steps = (
        ValidatedAttackStep(
            type="prompt",
            turn_id=1,
            mode="suffix",
            content="M4-PARALLEL-ISOLATION-PROBE",
        ),
    )
    first = materialize_attempt_dir(
        source_task_dir=source,
        source_manifest=manifest,
        episode_root=tmp_path / "episode-a",
        attempt_index=1,
        steps=steps,
    )
    second = materialize_attempt_dir(
        source_task_dir=source,
        source_manifest=manifest,
        episode_root=tmp_path / "episode-b",
        attempt_index=1,
        steps=steps,
    )

    result_a, result_b = await asyncio.gather(runner.run(first), runner.run(second))

    for result in (result_a, result_b):
        assert result.evaluation_started is True
        assert isinstance(result.attack_success, bool)
        assert result.runtime_destroyed is True
        assert result.judge_result is None
        assert result.victim_output is None
        assert result.trajectory_path is None
    assert result_a.runtime_identity != result_b.runtime_identity
    assert first.output_root != second.output_root
    BenchmarkIntegrityGuard.verify(source, manifest)
