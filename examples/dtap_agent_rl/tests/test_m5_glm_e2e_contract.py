import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.scripts import smoke_m5_glm_e2e
from examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e import (
    M6_PROMPT,
    M6_TOOLS,
    PROMPT,
    TOOLS,
    RecordingRunner,
)


def test_glm_e2e_prompt_requires_generation_without_shipping_a_plan():
    assert "No candidate plan or payload is supplied" in PROMPT
    assert "get_task_spec" in PROMPT
    assert "get_attack_surface" in PROMPT
    assert "validate_attack_step" in PROMPT
    assert "submit_attack" in PROMPT
    assert '"steps"' not in PROMPT
    assert "injection_payload" not in PROMPT
    assert TOOLS == {
        "mcp__dtap__get_task_spec",
        "mcp__dtap__get_attack_surface",
        "mcp__dtap__validate_attack_step",
        "mcp__dtap__submit_attack",
    }


def test_m6_glm_prompt_requires_owned_receipt_validation_before_submit():
    assert "No candidate plan or payload is supplied" in M6_PROMPT
    assert "MUST call validate_attack_step" in M6_PROMPT
    assert "returned action_id" in M6_PROMPT
    assert M6_TOOLS - TOOLS == {
        "mcp__dtap__apply_attack_step", "mcp__dtap__validate_placement"
    }


def test_glm_smoke_has_opt_in_viewer_artifact_export():
    source = inspect.getsource(smoke_m5_glm_e2e)
    assert '"--artifacts-dir"' in source
    assert '"policy.jsonl"' in source
    assert '"original-config.yaml"' in source
    assert '"submitted-config.yaml"' in source
    assert '"victim-trajectory.json"' in source


@pytest.mark.asyncio
async def test_recording_runner_exports_submitted_config_and_victim_trace(tmp_path):
    class Delegate:
        async def run(self, workspace):
            return "receipt"

    task_dir = tmp_path / "attempt"
    output_root = tmp_path / "results"
    trace = output_root / "nested" / "20260904.json"
    task_dir.mkdir()
    trace.parent.mkdir(parents=True)
    config_path = task_dir / "config.yaml"
    config_path.write_text(
        "Attack:\n  attack_turns:\n    - attack_steps:\n        - type: prompt\n"
        "          content: policy-authored payload\n",
        encoding="utf-8",
    )
    trace.write_text('{"trajectory":[]}\n', encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    runner = RecordingRunner(Delegate(), artifacts_dir=artifacts)

    result = await runner.run(SimpleNamespace(
        config_path=config_path,
        output_root=output_root,
    ))

    assert result == "receipt"
    assert runner.exported_victim_traces == 1
    assert (artifacts / "submitted-config.yaml").read_text() == config_path.read_text()
    assert (artifacts / "victim-trajectory.json").read_text() == trace.read_text()


@pytest.mark.integration
def test_real_glm_authored_m5_episode_optional():
    task_dir = os.environ.get("DTAP_M5_GLM_TASK_DIR")
    dtap_root = os.environ.get("DTAP_M5_DTAP_ROOT")
    if not task_dir or not dtap_root:
        pytest.skip("set DTAP_M5_GLM_TASK_DIR and DTAP_M5_DTAP_ROOT for the real GLM gate")
    command = [
        sys.executable,
        "-m",
        "examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e",
        "--task-dir",
        task_dir,
        "--dtap-root",
        dtap_root,
        "--python",
        os.environ.get("DTAP_M5_DTAP_PYTHON", sys.executable),
        "--timeout",
        os.environ.get("DTAP_M5_TIMEOUT", "1800"),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=1860)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert '"status": "passed"' in completed.stdout
    assert '"generated_plan"' in completed.stdout


@pytest.mark.integration
def test_real_glm_authored_m6_episode_optional():
    task_dir = os.environ.get("DTAP_M6_GLM_TASK_DIR")
    dtap_root = os.environ.get("DTAP_M6_DTAP_ROOT")
    if not task_dir or not dtap_root:
        pytest.skip("set DTAP_M6_GLM_TASK_DIR and DTAP_M6_DTAP_ROOT for the real GLM gate")
    command = [
        sys.executable, "-m", "examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e",
        "--m6-placement", "--task-dir", task_dir, "--dtap-root", dtap_root,
        "--python", os.environ.get("DTAP_M6_DTAP_PYTHON", sys.executable),
        "--timeout", os.environ.get("DTAP_M6_TIMEOUT", "1800"),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=1860)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert '"status": "passed"' in completed.stdout
    assert '"placements_verified": 1' in completed.stdout
