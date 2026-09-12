import sys
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl import live_catalog
from examples.dtap_agent_rl.live_catalog import LiveDtapCatalogProvider


@pytest.mark.asyncio
async def test_environment_catalog_retries_transient_startup_failure(monkeypatch):
    starts = 0
    stopped = []

    class Manager:
        def stop_all(self):
            stopped.append(True)

    def start(config, *, resource_manager, task_id):
        nonlocal starts
        starts += 1
        if starts == 1:
            return None, config
        return Manager(), {
            "environment_servers": {
                "mail-injection": {"url": "http://catalog.test/mcp", "tools": "all"},
            }
        }

    async def list_tools(server_name, url, *, classify_environment=False):
        assert (server_name, url) == ("mail-injection", "http://catalog.test/mcp")
        assert classify_environment is True
        return ["tool"]

    monkeypatch.setattr("utils.injection_mcp_helpers.start_injection_mcp_servers", start)
    monkeypatch.setattr("utils.injection_mcp_helpers.wait_for_injection_mcp_ready", lambda config: None)
    monkeypatch.setattr("utils.resource_manager.ResourceManager.instance", lambda: object())
    monkeypatch.setattr(live_catalog, "_list_url_tools", list_tools)
    monkeypatch.setattr(live_catalog.asyncio, "sleep", _no_sleep)
    snapshot = SimpleNamespace(
        injection_config={
            "environment_servers": {"mail-injection": "all"},
        }
    )

    result = await LiveDtapCatalogProvider(task_runtime_id="episode").list_environment_tools(snapshot)

    assert starts == 2
    assert result == {"mail-injection": ["tool"]}
    assert stopped == [True]


@pytest.mark.asyncio
async def test_environment_catalog_fails_closed_after_startup_retries(monkeypatch):
    starts = 0

    def start(config, *, resource_manager, task_id):
        nonlocal starts
        starts += 1
        return None, config

    monkeypatch.setattr("utils.injection_mcp_helpers.start_injection_mcp_servers", start)
    monkeypatch.setattr("utils.resource_manager.ResourceManager.instance", lambda: object())
    monkeypatch.setattr(live_catalog.asyncio, "sleep", _no_sleep)
    snapshot = SimpleNamespace(
        injection_config={
            "environment_servers": {"mail-injection": "all"},
        }
    )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        await LiveDtapCatalogProvider(task_runtime_id="episode").list_environment_tools(snapshot)
    assert starts == 3


async def _no_sleep(_delay):
    return None


@pytest.mark.asyncio
async def test_environment_catalog_attaches_explicit_placement_capability(monkeypatch):
    class Tool:
        name = "get_payload"
        description = "A deliberately oddly named mutator"
        inputSchema = {"type": "object", "properties": {}}

    class Client:
        def __init__(self, url):
            assert url == "http://catalog.test/mcp"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self):
            return [Tool()]

    monkeypatch.setitem(sys.modules, "fastmcp", SimpleNamespace(Client=Client))
    monkeypatch.setattr(
        live_catalog,
        "_environment_placement_capability",
        lambda server, tool: ("verified" if (server, tool) == ("mail-injection", "get_payload") else "unsupported"),
    )

    tools = await live_catalog._list_url_tools(
        "mail-injection",
        "http://catalog.test/mcp",
        classify_environment=True,
    )

    assert tools[0].placement_capability == "verified"
    assert "placement_capability" not in tools[0].to_dict()
