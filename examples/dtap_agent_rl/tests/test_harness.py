import asyncio
from unittest.mock import patch

import pytest

from slime.agent.harness import HarnessContext

from examples.dtap_agent_rl.harness import DTAPClaudeCodeHarness

from .conftest import FakeSandbox


_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):
    await _REAL_SLEEP(0)


def _ctx():
    return HarnessContext(
        workdir="/workspace/repo",
        session_id="episode-session-0123456789abcdef",
        adapter_url="http://adapter:18001",
    )


@pytest.mark.asyncio
async def test_harness_writes_parent_claude_config_and_http_mcp_config_without_secret_material():
    sb = FakeSandbox()
    harness = DTAPClaudeCodeHarness()
    await harness.write_config(sb, _ctx())

    combined = "\n".join(cmd for cmd, _user in sb.exec_log)

    # Observable contract of the real parent ClaudeCodeHarness.write_config().
    assert "/home/agent/.claude/settings.json" in combined
    assert "bypassPermissionsModeAccepted" in combined
    assert "hasCompletedOnboarding" in combined

    # DTAP specialization adds only env-placeholder MCP configuration.
    assert "/home/agent/.dtap/mcp.json" in combined
    assert "DTAP_HARNESS_URL" in combined
    assert "DTAP_EPISODE_TOKEN" in combined
    assert "Authorization" in combined
    assert "episode-session-0123456789abcdef" not in combined
    assert "http://adapter:18001" not in combined


@pytest.mark.asyncio
async def test_harness_launch_preserves_slime_adapter_and_injects_mcp_capability():
    captured = {}

    async def agent(env):
        captured["env"] = dict(env)
        return 0

    sb = FakeSandbox(on_launch=agent)
    harness = DTAPClaudeCodeHarness()

    with (
        patch.dict("os.environ", {"DTAP_HARNESS_URL": "http://dtap-host:19090/mcp/"}, clear=False),
        patch("asyncio.sleep", new=_fast_sleep),
    ):
        rc = await harness.launch_and_wait(sb, _ctx(), "inspect task", 90)

    assert rc == 0
    env = captured["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://adapter:18001"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "episode-session-0123456789abcdef"
    assert env["DTAP_HARNESS_URL"] == "http://dtap-host:19090/mcp/"
    assert env["DTAP_EPISODE_TOKEN"] == "episode-session-0123456789abcdef"

    # Current slime run_agent writes the launched command into its run.sh body.
    body = next(
        value
        for path, value in sb.files.items()
        if str(path).endswith(".run.sh") or str(path).endswith("/run.sh")
    )
    if isinstance(body, bytes):
        body = body.decode()
    assert "claude -p 'inspect task'" in body
    assert "--mcp-config /home/agent/.dtap/mcp.json" in body
    assert '--allowedTools "mcp__dtap"' in body
    assert "--disallowedTools" in body
    for tool in harness.denied_native_tools:
        assert tool in body


@pytest.mark.asyncio
async def test_harness_requires_host_mcp_url():
    sb = FakeSandbox()
    harness = DTAPClaudeCodeHarness()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="DTAP_HARNESS_URL"):
            await harness.launch_and_wait(sb, _ctx(), "inspect", 30)
