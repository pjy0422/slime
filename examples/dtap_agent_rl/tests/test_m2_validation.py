from dataclasses import dataclass


from examples.dtap_agent_rl.actions import candidate_attack_step_schema
from examples.dtap_agent_rl.validation import ValidationContext, validate_attack_plan, validate_attack_step


@dataclass(frozen=True)
class Tool:
    qualified_name: str
    input_schema: dict


@dataclass(frozen=True)
class Surface:
    prompt_enabled: bool = True
    tool_enabled: bool = True
    environment_enabled: bool = True
    skill_enabled: bool = True
    prompt_modes: tuple[str, ...] = ("suffix", "override")
    tool_modes: tuple[str, ...] = ("suffix", "override")
    skill_modes: tuple[str, ...] = ("insert", "append", "create")
    victim_tools: tuple = (Tool("slack:get_messages", {}),)
    environment_tools: tuple = (
        Tool(
            "slack-injection:inject_slack_message",
            {
                "type": "object",
                "properties": {
                    "channel_name": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["channel_name", "message"],
                "additionalProperties": False,
            },
        ),
    )
    skill_targets: tuple[str, ...] = ("payments",)


def indirect():
    return ValidationContext("indirect", 2, Surface())


def test_candidate_schema_has_four_strict_variants_and_no_a2a():
    schema = candidate_attack_step_schema()
    assert len(schema["oneOf"]) == 4
    assert "a2a" not in repr(schema)
    assert all(item["additionalProperties"] is False for item in schema["oneOf"])


def test_prompt_validation_is_strict_and_turn_bounded():
    ok = validate_attack_step({"type": "prompt", "turn_id": 2, "mode": "suffix", "content": "x"}, indirect())
    assert ok.valid and ok.step.turn_id == 2
    assert (
        validate_attack_step({"type": "prompt", "turn_id": 3, "mode": "suffix", "content": "x"}, indirect())
        .errors[0]
        .code
        == "TURN_OUT_OF_RANGE"
    )
    assert (
        validate_attack_step({"type": "prompt", "turn_id": 1, "mode": "jailbreak", "content": "x"}, indirect())
        .errors[0]
        .code
        == "MODE_NOT_ALLOWED"
    )


def test_direct_prompt_allows_jailbreak_not_suffix():
    ctx = ValidationContext("direct", 0, Surface(prompt_modes=("jailbreak",)))
    assert validate_attack_step({"type": "prompt", "turn_id": 3, "mode": "jailbreak", "content": "x"}, ctx).valid
    assert not validate_attack_step({"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "x"}, ctx).valid


def test_tool_is_global_and_rejects_turn_and_unknown_target():
    assert validate_attack_step(
        {"type": "tool", "mode": "override", "content": "x", "injected_tool": "slack:get_messages"}, indirect()
    ).valid
    result = validate_attack_step(
        {"type": "tool", "turn_id": 1, "mode": "override", "content": "x", "injected_tool": "slack:get_messages"},
        indirect(),
    )
    assert result.errors[0].code == "FIELD_NOT_ALLOWED"
    assert (
        validate_attack_step(
            {"type": "tool", "mode": "override", "content": "x", "injected_tool": "gmail:search"}, indirect()
        )
        .errors[0]
        .code
        == "TARGET_NOT_ALLOWED"
    )


def test_environment_requires_exact_target_and_json_schema():
    base = {
        "type": "environment",
        "turn_id": 1,
        "injection_mcp_tool": "slack-injection:inject_slack_message",
    }
    assert validate_attack_step({**base, "kwargs": {"channel_name": "general", "message": "hello"}}, indirect()).valid
    missing = validate_attack_step({**base, "kwargs": {"message": "hello"}}, indirect())
    assert missing.errors[0].code == "SCHEMA_MISMATCH"
    extra = validate_attack_step(
        {**base, "kwargs": {"channel_name": "general", "message": "x", "secret": 1}}, indirect()
    )
    assert extra.errors[0].code == "SCHEMA_MISMATCH"


def test_skill_existing_create_and_path_rules():
    assert validate_attack_step(
        {"type": "skill", "mode": "append", "content": "x", "skill_name": "payments", "row": -1}, indirect()
    ).valid
    assert validate_attack_step(
        {"type": "skill", "mode": "create", "content": "x", "skill_name": "new-skill", "row": -1}, indirect()
    ).valid
    assert (
        validate_attack_step(
            {"type": "skill", "mode": "create", "content": "x", "skill_name": "../escape", "row": -1}, indirect()
        )
        .errors[0]
        .code
        == "INVALID_SKILL_NAME"
    )
    assert (
        validate_attack_step(
            {"type": "skill", "mode": "create", "content": "x", "skill_name": "payments", "row": -1}, indirect()
        )
        .errors[0]
        .code
        == "TARGET_NOT_ALLOWED"
    )
    assert (
        validate_attack_step(
            {"type": "skill", "mode": "append", "content": "x", "skill_name": "payments", "row": 2}, indirect()
        )
        .errors[0]
        .code
        == "FIELD_NOT_ALLOWED"
    )


def test_a2a_and_unknown_fields_fail_closed():
    assert validate_attack_step({"type": "a2a"}, indirect()).errors[0].code == "UNSUPPORTED_IN_M2"
    result = validate_attack_step(
        {"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "x", "secret": 1}, indirect()
    )
    assert result.errors[0].code == "UNKNOWN_FIELD"


def test_plan_rejects_dtap_silent_overwrite_cases():
    plan = [
        {"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "a"},
        {"type": "prompt", "turn_id": 1, "mode": "override", "content": "b"},
    ]
    result = validate_attack_plan(plan, indirect())
    assert not result.valid
    assert result.errors[0].code == "PLAN_CONFLICT"


def test_direct_plan_requires_contiguous_jailbreak_turns_for_environment_semantics():
    ctx = ValidationContext("direct", 0, Surface(prompt_modes=("jailbreak",)))
    result = validate_attack_plan(
        [
            {"type": "prompt", "turn_id": 1, "mode": "jailbreak", "content": "a"},
            {"type": "prompt", "turn_id": 3, "mode": "jailbreak", "content": "b"},
        ],
        ctx,
    )
    assert not result.valid
