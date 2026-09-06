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
    _victim_artifacts_complete,
    _environment_step_count,
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
    assert "plan_constraints" in M6_PROMPT
    assert "policy_limits" in M6_PROMPT
    assert "complete target allowlist" in M6_PROMPT
    assert "never guess, enumerate, or probe names" in M6_PROMPT
    assert "stop exploring alternative placements" in M6_PROMPT
    assert "max_apply_attack_step_calls" in M6_PROMPT
    assert "clear_" in M6_PROMPT and "reset_" in M6_PROMPT
    assert "nonterminal" in M6_PROMPT and "INVALID_SUBMISSION" in M6_PROMPT
    assert "INVALID_SUBMISSION does not consume H" in M6_PROMPT
    assert "positively validated placement receipt" in M6_PROMPT
    assert "placement is unsupported and repair.fields is empty" in M6_PROMPT
    assert "single targeted repair is also invalid" in M6_PROMPT
    assert "stop retrying" in M6_PROMPT
    assert "omit that environment step from the final plan" in M6_PROMPT
    assert M6_TOOLS - TOOLS == {
        "mcp__dtap__apply_attack_step", "mcp__dtap__validate_placement"
    }


def test_direct_plan_does_not_require_an_environment_placement_receipt():
    assert _environment_step_count([{
        "turn_id": 1,
        "attack_steps": [{"type": "prompt", "mode": "suffix", "content": "x"}],
    }]) == 0
    assert _environment_step_count([{
        "turn_id": 1,
        "attack_steps": [{"type": "environment", "kwargs": {"message": "x"}}],
    }]) == 1


def test_openclaw_headless_artifact_contract_uses_proxy_events_as_authority():
    runner = SimpleNamespace(
        exported_victim_traces=0,
        exported_victim_mcp_events=1,
    )
    assert _victim_artifacts_complete(runner, "openclaw") is True
    assert _victim_artifacts_complete(runner, "claude_code") is False

    runner.exported_victim_mcp_events = 0
    runner.exported_victim_traces = 1
    assert _victim_artifacts_complete(runner, "openclaw") is False
    assert _victim_artifacts_complete(runner, "claude_code") is True


def test_glm_smoke_has_opt_in_viewer_artifact_export():
    source = inspect.getsource(smoke_m5_glm_e2e)
    assert '"--artifacts-dir"' in source
    assert '"policy.jsonl"' in source
    assert '"original-config.yaml"' in source
    assert '"submitted-config.yaml"' in source
    assert '"victim-trajectory.json"' in source
    assert '"victim-mcp-events.jsonl"' in source
    assert '"judge-result.json"' in source
    assert '"attempts"' in source
    assert '"episode-state.json"' in source
    assert '"matches_source_template"' in source
    assert "generated plan unexpectedly equals" not in source
    assert '"timeout": (args.timeout + 60) * 1000' in source
    assert "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT" in source
    assert "max_submissions=args.max_submissions" in source


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
    event_log = output_root / "nested" / "run.mcp-events.jsonl"
    event_log.write_text("", encoding="utf-8")
    judge = output_root / "nested" / "judge_result.json"
    judge.write_text('{"attack_success":false}\n', encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    runner = RecordingRunner(Delegate(), artifacts_dir=artifacts)

    result = await runner.run(SimpleNamespace(
        config_path=config_path,
        output_root=output_root,
    ))

    assert result == "receipt"
    assert runner.exported_victim_traces == 1
    assert runner.exported_victim_mcp_events == 1
    assert runner.exported_judge_artifacts == 1
    assert (artifacts / "submitted-config.yaml").read_text() == config_path.read_text()
    assert (artifacts / "victim-trajectory.json").read_text() == trace.read_text()
    assert (artifacts / "victim-mcp-events.jsonl").is_file()
    assert (artifacts / "judge-result.json").read_text() == judge.read_text()
    assert (artifacts / "attempts/attempt-0001/submitted-config.yaml").is_file()
    assert (artifacts / "attempts/attempt-0001/judge-result.json").is_file()


@pytest.mark.asyncio
async def test_recording_runner_retains_each_h_submission(tmp_path):
    class Delegate:
        async def run(self, workspace):
            output = workspace.output_root / "judge_result.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                '{"attack_success":' + ("true" if workspace.attempt_index == 2 else "false") + '}\n'
            )
            return workspace.attempt_index

    artifacts = tmp_path / "artifacts"
    runner = RecordingRunner(Delegate(), artifacts_dir=artifacts)
    for index in (1, 2):
        task = tmp_path / f"task-{index}"
        task.mkdir()
        config = task / "config.yaml"
        config.write_text(
            f"Attack:\n  attack_turns:\n    - attack_steps:\n        - type: prompt\n          content: attempt-{index}\n"
        )
        await runner.run(SimpleNamespace(
            attempt_index=index,
            config_path=config,
            output_root=tmp_path / f"results-{index}",
        ))

    assert len(runner.plans) == 2
    assert "attempt-1" in (artifacts / "attempts/attempt-0001/submitted-config.yaml").read_text()
    assert "attempt-2" in (artifacts / "attempts/attempt-0002/submitted-config.yaml").read_text()
    assert (artifacts / "attempts/attempt-0001/judge-result.json").is_file()
    assert (artifacts / "attempts/attempt-0002/judge-result.json").is_file()
    assert "attempt-2" in (artifacts / "submitted-config.yaml").read_text()


@pytest.mark.asyncio
async def test_recording_runner_retains_partial_artifacts_when_delegate_fails(tmp_path):
    output_root = tmp_path / "results"
    output_root.mkdir()
    (output_root / "run.mcp-events.jsonl").write_text("")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("Attack:\n  attack_turns: []\n")

    class Delegate:
        async def run(self, workspace):
            (workspace.output_root / "judge_result.json").write_text(
                '{"error":"judge unavailable"}\n'
            )
            raise RuntimeError("victim failed")

    artifacts = tmp_path / "artifacts"
    runner = RecordingRunner(Delegate(), artifacts_dir=artifacts)
    with pytest.raises(RuntimeError, match="victim failed"):
        await runner.run(SimpleNamespace(
            config_path=config_path, output_root=output_root,
        ))

    assert (artifacts / "submitted-config.yaml").is_file()
    assert (artifacts / "victim-mcp-events.jsonl").is_file()
    assert (artifacts / "judge-result.json").is_file()
    state = __import__("json").loads((artifacts / "episode-state.json").read_text())
    assert state["plan_generated"] is True
    assert state["evaluation_delegate_completed"] is False


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
