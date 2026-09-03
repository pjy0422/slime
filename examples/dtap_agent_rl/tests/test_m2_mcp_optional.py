import importlib.util

import pytest

from examples.dtap_agent_rl.attack_surface import AttackSurface, ToolSpec
from examples.dtap_agent_rl.mcp_server import create_mcp_server
from examples.dtap_agent_rl.service import EpisodeRegistry, EpisodeView
from examples.dtap_agent_rl.task_projection import PolicyTaskSpec

TOKEN = "episode-m2-http-0123456789abcdef"


@pytest.mark.asyncio
async def test_m2_fastmcp_has_three_tools_and_validation_round_trip():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed")
    from fastmcp import Client

    view = EpisodeView(
        task=PolicyTaskSpec("task", "workflow", "goal", "normal", "indirect", None),
        attack_surface=AttackSurface(
            True, True, True, False, (),
            (ToolSpec("slack", "read", "slack:read", None, {"type": "object"}),),
            (ToolSpec("slack-injection", "inject", "slack-injection:inject", None, {
                "type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]
            }),),
            prompt_modes=("suffix", "override"), tool_modes=("suffix", "override"),
        ),
    )
    registry = EpisodeRegistry()
    registry.register(TOKEN, view)
    mcp = create_mcp_server(registry)
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert sorted(t.name for t in tools) == ["get_attack_surface", "get_task_spec", "validate_attack_step"]
