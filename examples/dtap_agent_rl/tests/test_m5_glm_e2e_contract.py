import os
import subprocess
import sys

import pytest

from examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e import PROMPT, TOOLS


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
