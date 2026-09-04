import pytest

from examples.dtap_agent_rl.security_policy import M4SecurityPolicy, PolicyInputLimitError


def test_m4_q_is_explicit_and_positive():
    with pytest.raises(ValueError):
        M4SecurityPolicy(max_submit_calls=0)
    assert M4SecurityPolicy(max_submit_calls=3).max_submit_calls == 3


@pytest.mark.parametrize(
    "plan",
    [
        {"steps": [{}] * 3},
        {"steps": [{"content": "x" * 9}]},
        {"steps": [{"kwargs": {"a": {"b": {"c": 1}}}}]},
    ],
)
def test_plan_preflight_bounds_steps_content_and_depth(plan):
    policy = M4SecurityPolicy(
        max_submit_calls=2,
        max_steps_per_plan=2,
        max_content_bytes=8,
        max_json_depth=4,
    )
    with pytest.raises(PolicyInputLimitError):
        policy.preflight_plan(plan)


def test_child_environment_is_allowlist_not_host_copy():
    policy = M4SecurityPolicy(
        max_submit_calls=2,
        inherited_dtap_env_names=("OPENAI_API_KEY",),
    )
    result = policy.build_dtap_child_env(
        host_env={
            "PATH": "/safe/bin",
            "OPENAI_API_KEY": "provider-secret",
            "DTAP_EPISODE_TOKEN": "must-not-leak",
            "HOST_SECRET": "must-not-leak",
        },
        explicit_env={"EVAL_MODE": "m4"},
    )
    assert result["PATH"] == "/safe/bin"
    assert result["OPENAI_API_KEY"] == "provider-secret"
    assert result["EVAL_MODE"] == "m4"
    assert "DTAP_EPISODE_TOKEN" not in result
    assert "HOST_SECRET" not in result


def test_child_environment_refuses_policy_and_adapter_credentials():
    with pytest.raises(ValueError):
        M4SecurityPolicy(
            max_submit_calls=1,
            inherited_dtap_env_names=("ANTHROPIC_AUTH_TOKEN",),
        )
    policy = M4SecurityPolicy(max_submit_calls=1)
    with pytest.raises(ValueError):
        policy.build_dtap_child_env(explicit_env={"DTAP_EPISODE_TOKEN": "secret"})


def test_child_environment_allows_only_explicit_victim_provider_aliases():
    policy = M4SecurityPolicy(
        max_submit_calls=1,
        inherited_dtap_env_names=(
            "ANTHROPIC_API_KEY",
            "DTAP_VICTIM_ANTHROPIC_BASE_URL",
            "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN",
        ),
    )
    result = policy.build_dtap_child_env(
        host_env={
            "ANTHROPIC_API_KEY": "victim-provider-key",
            "ANTHROPIC_BASE_URL": "must-not-inherit",
            "ANTHROPIC_AUTH_TOKEN": "must-not-inherit",
            "DTAP_VICTIM_ANTHROPIC_BASE_URL": "https://provider.example",
            "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN": "1",
        }
    )
    assert result["ANTHROPIC_API_KEY"] == "victim-provider-key"
    assert result["DTAP_VICTIM_ANTHROPIC_BASE_URL"] == "https://provider.example"
    assert result["DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN"] == "1"
    assert "ANTHROPIC_BASE_URL" not in result
    assert "ANTHROPIC_AUTH_TOKEN" not in result
