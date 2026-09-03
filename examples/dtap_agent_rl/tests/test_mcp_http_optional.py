"""Real FastMCP HTTP + Bearer round trip. Skips when FastMCP is unavailable."""

import asyncio
import importlib.util
import socket

import pytest

from examples.dtap_agent_rl.mcp_server import create_mcp_server
from examples.dtap_agent_rl.service import EpisodeRegistry, EpisodeView

from .conftest import sample_surface, sample_task_spec


TOKEN = "episode-http-0123456789abcdef"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_port(port: int) -> None:
    for _ in range(100):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.02)
    raise RuntimeError("FastMCP HTTP server did not start")


@pytest.mark.asyncio
async def test_fastmcp_http_bearer_round_trip_and_wrong_token_rejection():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed in this execution environment")

    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    registry = EpisodeRegistry()
    registry.register(TOKEN, EpisodeView(task=sample_task_spec(), attack_surface=sample_surface()))
    mcp = create_mcp_server(registry)
    port = _free_port()
    server_task = asyncio.create_task(
        mcp.run_async(
            transport="http",
            host="127.0.0.1",
            port=port,
            stateless_http=True,
            show_banner=False,
        )
    )
    try:
        await _wait_for_port(port)
        url = f"http://127.0.0.1:{port}/mcp/"

        async with Client(url, auth=TOKEN) as client:
            tools = await client.list_tools()
            assert sorted(t.name for t in tools) == ["get_attack_surface", "get_task_spec"]

            task_result = await client.call_tool("get_task_spec", {})
            assert task_result.data["malicious_goal"] == sample_task_spec().malicious_goal
            assert TOKEN not in repr(task_result.data)

            surface_result = await client.call_tool("get_attack_surface", {})
            assert surface_result.data["environment_tools"][0]["qualified_name"] == (
                "slack-injection:inject_slack_message"
            )

        async with Client(url, auth="wrong-token-0123456789abcdef") as client:
            with pytest.raises(ToolError):
                await client.call_tool("get_task_spec", {})
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task
