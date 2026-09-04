from unittest.mock import patch

import pytest
from slime.agent.harness import HarnessContext

from examples.dtap_agent_rl.harness import M4ClaudeCodeHarness, M6ClaudeCodeHarness
from .conftest import FakeSandbox


TOKEN = "mcp-token-independent-0123456789abcdef"
SESSION = "adapter-session-0123456789abcdef"


def _ctx():
    return HarnessContext(
        workdir="/workspace/empty",
        session_id=SESSION,
        adapter_url="http://adapter:18001",
    )


@pytest.mark.asyncio
async def test_strict_harness_splits_credentials_and_allows_exactly_four_tools():
    captured = {}

    async def launch(env):
        captured.update(env)
        return 0

    sandbox = FakeSandbox(on_launch=launch)
    harness = M4ClaudeCodeHarness(episode_token=TOKEN)
    await harness.write_config(sandbox, _ctx())
    with patch.dict(
        "os.environ",
        {"DTAP_HARNESS_URL": "http://mcp:19090/mcp/"},
        clear=True,
    ):
        await harness.launch_and_wait(sandbox, _ctx(), "attack", 30)

    assert captured["ANTHROPIC_AUTH_TOKEN"] == SESSION
    assert captured["DTAP_EPISODE_TOKEN"] == TOKEN
    assert captured["ANTHROPIC_AUTH_TOKEN"] != captured["DTAP_EPISODE_TOKEN"]
    body = "\n".join(str(value) for value in sandbox.files.values())
    assert "--strict-mcp-config" in body
    for tool in harness.exact_policy_tools:
        assert tool in body
    assert "disableAllHooks" in "\n".join(cmd for cmd, _ in sandbox.exec_log)


@pytest.mark.asyncio
async def test_strict_harness_rejects_arbitrary_host_overrides():
    harness = M4ClaudeCodeHarness(episode_token=TOKEN)
    with patch.dict(
        "os.environ",
        {
            "DTAP_HARNESS_URL": "http://mcp:19090/mcp/",
            "SLIME_AGENT_CC_EXTRA_ARGS": "--dangerous",
        },
        clear=True,
    ):
        with pytest.raises(RuntimeError, match="forbids"):
            await harness.launch_and_wait(FakeSandbox(), _ctx(), "attack", 30)


def test_m6_harness_adds_only_the_two_receipt_tools():
    m4 = set(M4ClaudeCodeHarness.exact_policy_tools)
    m6 = set(M6ClaudeCodeHarness.exact_policy_tools)
    assert m6 - m4 == {
        "mcp__dtap__apply_attack_step", "mcp__dtap__validate_placement"
    }
