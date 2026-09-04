import asyncio
import importlib.util
import socket

import pytest

from examples.dtap_agent_rl.attack_surface import AttackSurface
from examples.dtap_agent_rl.mcp_server import create_mcp_server
from examples.dtap_agent_rl.service import EpisodeRegistry, EpisodeView
from examples.dtap_agent_rl.task_projection import PolicyTaskSpec

from .conftest import drain_sse_shutdown_watcher

TOKEN_A = "episode-m3-a-0123456789abcdef"
TOKEN_B = "episode-m3-b-0123456789abcdef"


class FakeController:
    def __init__(self, *, terminal=False):
        self.submissions_used = 0
        self.terminal = terminal

    async def submit(self, plan):
        if self.terminal:
            return {
                "accepted": False,
                "terminal": True,
                "submissions_used": self.submissions_used,
                "remaining_submissions": 0,
                "errors": [{"code": "EPISODE_TERMINAL", "message": "episode is terminal"}],
            }
        self.submissions_used += 1
        return {
            "accepted": True,
            "submission": self.submissions_used,
            "success": False,
            "terminal": False,
            "remaining_submissions": 2,
        }


def _view():
    return EpisodeView(
        task=PolicyTaskSpec("task", "workflow", "goal", "normal", "indirect", None),
        attack_surface=AttackSurface(
            prompt_enabled=True,
            tool_enabled=False,
            environment_enabled=False,
            skill_enabled=False,
            skill_modes=(),
            victim_tools=(),
            environment_tools=(),
            prompt_modes=("suffix", "override"),
        ),
    )


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait(port):
    for _ in range(100):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            del reader
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.02)
    raise RuntimeError("server did not start")


def _server(controllers):
    registry = EpisodeRegistry()
    registry.register(TOKEN_A, _view())
    registry.register(TOKEN_B, _view())
    return create_mcp_server(
        registry,
        submission_controller_resolver=lambda token: controllers[token],
    )


@pytest.mark.asyncio
async def test_m3_fastmcp_exposes_exactly_one_new_mutating_tool():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed")
    from fastmcp import Client

    server = _server({TOKEN_A: FakeController(), TOKEN_B: FakeController()})
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert names == {
        "get_task_spec",
        "get_attack_surface",
        "validate_attack_step",
        "submit_attack",
    }


@pytest.mark.asyncio
async def test_submit_attack_routes_to_the_bearer_episode_only():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed")
    from fastmcp import Client

    controller_a, controller_b = FakeController(), FakeController()
    server = _server({TOKEN_A: controller_a, TOKEN_B: controller_b})
    port = _free_port()
    server_task = asyncio.create_task(server.run_async(
        transport="http",
        host="127.0.0.1",
        port=port,
        stateless_http=True,
        show_banner=False,
    ))
    try:
        await _wait(port)
        async with Client(f"http://127.0.0.1:{port}/mcp/", auth=TOKEN_A) as client:
            result = await client.call_tool("submit_attack", {"plan": {"steps": []}})
        assert result.data["accepted"] is True
        assert controller_a.submissions_used == 1
        assert controller_b.submissions_used == 0
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task
        await drain_sse_shutdown_watcher()


@pytest.mark.asyncio
async def test_terminal_episode_rejects_all_further_mutation():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed")
    from fastmcp import Client

    server = _server({TOKEN_A: FakeController(terminal=True), TOKEN_B: FakeController()})
    port = _free_port()
    server_task = asyncio.create_task(server.run_async(
        transport="http",
        host="127.0.0.1",
        port=port,
        stateless_http=True,
        show_banner=False,
    ))
    try:
        await _wait(port)
        async with Client(f"http://127.0.0.1:{port}/mcp/", auth=TOKEN_A) as client:
            result = await client.call_tool("submit_attack", {"plan": {"steps": []}})
        assert result.data["accepted"] is False
        assert result.data["errors"][0]["code"] == "EPISODE_TERMINAL"
    finally:
        server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server_task
        await drain_sse_shutdown_watcher()
