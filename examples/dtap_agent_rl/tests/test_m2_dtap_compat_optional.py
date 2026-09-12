import importlib.util

import pytest
from examples.dtap_agent_rl.dtap_action_compat import to_dtap_attack_config
from examples.dtap_agent_rl.tests.test_m2_validation import Surface
from examples.dtap_agent_rl.validation import ValidationContext, validate_attack_step


def _dtap_available():
    return importlib.util.find_spec("dt_arena") is not None


@pytest.mark.skipif(not _dtap_available(), reason="DTAP not installed/importable")
def test_validated_steps_preserve_current_dtap_helper_semantics():
    from utils.injection_helpers import (
        apply_prompt_injections,
        build_skill_injections_from_config,
        build_tool_injections_from_config,
        get_env_injections_from_attack,
    )

    ctx = ValidationContext("indirect", 1, Surface())
    raw = [
        {"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "PROMPT"},
        {"type": "tool", "mode": "override", "content": "TOOL", "injected_tool": "slack:get_messages"},
        {
            "type": "environment",
            "turn_id": 1,
            "injection_mcp_tool": "slack-injection:inject_slack_message",
            "kwargs": {"channel_name": "general", "message": "ENV"},
        },
        {"type": "skill", "mode": "append", "content": "SKILL", "skill_name": "payments", "row": -1},
    ]
    steps = [validate_attack_step(step, ctx).step for step in raw]
    assert all(step is not None for step in steps)
    cfg = to_dtap_attack_config(steps, threat_model="indirect", malicious_goal="goal")

    assert apply_prompt_injections("USER", cfg) == "USER\nPROMPT"
    tool = build_tool_injections_from_config(cfg)["slack"]["get_messages"]
    assert tool.type == "override" and tool.content == "TOOL"
    env = get_env_injections_from_attack(cfg, turn_id=1)
    assert env == [
        {
            "server_name": "slack-injection",
            "tool_name": "inject_slack_message",
            "kwargs": {"channel_name": "general", "message": "ENV"},
            "turn_id": 1,
            "feedback_step_index": 2,
        }
    ]
    skill = build_skill_injections_from_config(cfg)["payments"][0]
    assert skill.mode == "append" and skill.content == "SKILL" and skill.row == -1
