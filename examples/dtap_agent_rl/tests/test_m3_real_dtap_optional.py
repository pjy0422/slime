import os
from hashlib import sha256
from pathlib import Path

import pytest

from examples.dtap_agent_rl.actions import ValidatedAttackStep
from examples.dtap_agent_rl.attempt_runner import DtapAttemptRunner
from examples.dtap_agent_rl.candidate_config import materialize_attempt_dir


def _steps():
    return (
        ValidatedAttackStep(type="prompt", turn_id=1, mode="suffix", content="RESET-PROBE"),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_dtap_attempt_preserves_source_config_and_cleans_runtime(tmp_path):
    """Opt-in smoke; requires a task selected for deterministic reset verification.

    The selected task's setup/judge fixture must record a sentinel mutation in
    attempt 1 and assert that it is absent when attempt 2 starts.
    """
    task_dir = os.environ.get("DTAP_M3_RESET_TASK_DIR")
    if not task_dir:
        pytest.skip("set DTAP_M3_RESET_TASK_DIR for the real reset smoke")
    source = Path(task_dir).resolve()
    before = sha256((source / "config.yaml").read_bytes()).hexdigest()
    runner = DtapAttemptRunner(max_parallel=1)

    first = materialize_attempt_dir(
        source_task_dir=source,
        episode_root=tmp_path / "episode",
        attempt_index=1,
        steps=_steps(),
    )
    second = materialize_attempt_dir(
        source_task_dir=source,
        episode_root=tmp_path / "episode",
        attempt_index=2,
        steps=_steps(),
    )

    first_result = await runner.run(first)
    second_result = await runner.run(second)

    assert first_result.evaluation_started and second_result.evaluation_started
    assert first_result.runtime_identity != second_result.runtime_identity
    assert first_result.runtime_destroyed is True
    assert second_result.runtime_destroyed is True
    assert second_result.judge_result.get("reset_verified") is True
    assert sha256((source / "config.yaml").read_bytes()).hexdigest() == before
