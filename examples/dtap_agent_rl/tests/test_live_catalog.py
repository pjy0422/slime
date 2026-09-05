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
        return Manager(), {"environment_servers": {
            "mail-injection": {"url": "http://catalog.test/mcp", "tools": "all"},
        }}

    async def list_tools(server_name, url):
        assert (server_name, url) == ("mail-injection", "http://catalog.test/mcp")
        return ["tool"]

    monkeypatch.setattr(
        "utils.injection_mcp_helpers.start_injection_mcp_servers", start
    )
    monkeypatch.setattr(
        "utils.injection_mcp_helpers.wait_for_injection_mcp_ready", lambda config: None
    )
    monkeypatch.setattr(
        "utils.resource_manager.ResourceManager.instance", lambda: object()
    )
    monkeypatch.setattr(live_catalog, "_list_url_tools", list_tools)
    monkeypatch.setattr(live_catalog.asyncio, "sleep", _no_sleep)
    snapshot = SimpleNamespace(injection_config={
        "environment_servers": {"mail-injection": "all"},
    })

    result = await LiveDtapCatalogProvider(
        task_runtime_id="episode"
    ).list_environment_tools(snapshot)

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

    monkeypatch.setattr(
        "utils.injection_mcp_helpers.start_injection_mcp_servers", start
    )
    monkeypatch.setattr(
        "utils.resource_manager.ResourceManager.instance", lambda: object()
    )
    monkeypatch.setattr(live_catalog.asyncio, "sleep", _no_sleep)
    snapshot = SimpleNamespace(injection_config={
        "environment_servers": {"mail-injection": "all"},
    })

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        await LiveDtapCatalogProvider(
            task_runtime_id="episode"
        ).list_environment_tools(snapshot)
    assert starts == 3


async def _no_sleep(_delay):
    return None
