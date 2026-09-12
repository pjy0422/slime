from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.dtap_agent_rl.scripts.smoke_m1_api import (
    DENIED_NATIVE_TOOLS,
    REQUIRED_MCP_TOOLS,
    _claude_command,
    _collect_observed_tool_names,
    _direct_api_env,
    _mcp_config,
    _parse_stream_json,
)


def test_smoke_mcp_config_persists_no_secret_or_url():
    rendered = json.dumps(_mcp_config(), sort_keys=True)
    assert "${DTAP_HARNESS_URL}" in rendered
    assert "${DTAP_EPISODE_TOKEN}" in rendered
    assert "episode-secret" not in rendered
    assert "127.0.0.1:19090" not in rendered


def test_direct_api_env_removes_slime_adapter_routing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://slime-adapter:1234")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "slime-session")

    env = _direct_api_env(
        mcp_url="http://127.0.0.1:19090/mcp/",
        episode_token="episode-0123456789abcdef",
    )

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["DTAP_EPISODE_TOKEN"] == "episode-0123456789abcdef"


def test_direct_api_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _direct_api_env(
            mcp_url="http://127.0.0.1:19090/mcp/",
            episode_token="episode-0123456789abcdef",
        )


def test_claude_command_allows_exact_two_m1_tools(tmp_path: Path):
    cmd = _claude_command(
        claude_bin="claude",
        prompt="inspect",
        model="sonnet",
        mcp_config_path=tmp_path / "mcp.json",
        max_turns=6,
    )
    joined = " ".join(cmd)
    for name in REQUIRED_MCP_TOOLS:
        assert name in joined
    for name in DENIED_NATIVE_TOOLS:
        assert name in joined
    assert "--mcp-config" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd


def test_stream_parser_observes_both_mcp_tool_calls():
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "name": "mcp__dtap__get_task_spec", "input": {}}]},
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "tool_use", "name": "mcp__dtap__get_attack_surface", "input": {}}]
                    },
                }
            ),
            json.dumps({"type": "result", "result": "done"}),
        ]
    )
    events = _parse_stream_json(stdout)
    assert _collect_observed_tool_names(events) == REQUIRED_MCP_TOOLS
