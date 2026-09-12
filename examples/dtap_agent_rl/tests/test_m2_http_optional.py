import asyncio
import importlib.util
import socket

import pytest

from examples.dtap_agent_rl.attack_surface import AttackSurface, ToolSpec
from examples.dtap_agent_rl.mcp_server import create_mcp_server
from examples.dtap_agent_rl.service import EpisodeRegistry, EpisodeView
from examples.dtap_agent_rl.task_projection import PolicyTaskSpec

from .conftest import drain_sse_shutdown_watcher

TOKEN = "episode-m2-http-real-0123456789abcdef"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait(port):
    for _ in range(100):
        try:
            r, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            await w.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.02)
    raise RuntimeError("server did not start")


@pytest.mark.asyncio
async def test_m2_http_bearer_validation_round_trip():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed")
    from fastmcp import Client

    view = EpisodeView(
        task=PolicyTaskSpec("task", "workflow", "goal", "normal", "indirect", None),
        attack_surface=AttackSurface(
            True,
            True,
            True,
            False,
            (),
            (),
            (
                ToolSpec(
                    "slack-injection",
                    "inject",
                    "slack-injection:inject",
                    None,
                    {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                ),
            ),
            prompt_modes=("suffix", "override"),
            tool_modes=("suffix", "override"),
        ),
    )
    registry = EpisodeRegistry()
    registry.register(TOKEN, view)
    mcp = create_mcp_server(registry)
    port = _free_port()
    task = asyncio.create_task(
        mcp.run_async(
            transport="http",
            host="127.0.0.1",
            port=port,
            stateless_http=True,
            show_banner=False,
        )
    )
    try:
        await _wait(port)
        async with Client(f"http://127.0.0.1:{port}/mcp/", auth=TOKEN) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools}
            assert {"get_attack_surface", "get_task_spec", "validate_attack_step"} <= names
            good = await client.call_tool(
                "validate_attack_step",
                {
                    "step": {
                        "type": "environment",
                        "turn_id": 1,
                        "injection_mcp_tool": "slack-injection:inject",
                        "kwargs": {"message": "hello"},
                    }
                },
            )
            bad = await client.call_tool(
                "validate_attack_step",
                {
                    "step": {
                        "type": "environment",
                        "turn_id": 1,
                        "injection_mcp_tool": "slack-injection:inject",
                        "kwargs": {},
                    }
                },
            )
            assert good.data["valid"] is True
            assert bad.data["valid"] is False
            assert bad.data["errors"][0]["code"] == "SCHEMA_MISMATCH"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await drain_sse_shutdown_watcher()
