"""Real FastMCP in-memory round trip. Skips only when fastmcp is not installed."""

import importlib.util

import pytest

from examples.dtap_agent_rl.mcp_server import create_mcp_server
from examples.dtap_agent_rl.service import EpisodeRegistry, EpisodeView

from .conftest import sample_surface, sample_task_spec


TOKEN = "episode-mcp-0123456789abcdef"


@pytest.mark.asyncio
async def test_fastmcp_preserves_the_two_m1_read_only_tools():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed in this execution environment")

    from fastmcp import Client

    registry = EpisodeRegistry()
    registry.register(TOKEN, EpisodeView(task=sample_task_spec(), attack_surface=sample_surface()))
    mcp = create_mcp_server(registry)

    # In-memory transport has no HTTP headers, so list_tools is the useful M1
    # protocol-level assertion here; authenticated call_tool is tested in the
    # deployed HTTP smoke test because auth deliberately depends on HTTP headers.
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {"get_attack_surface", "get_task_spec"} <= {tool.name for tool in tools}
