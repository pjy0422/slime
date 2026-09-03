from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .attack_surface import ToolSpec
from .episode import TaskSnapshot


def _iter_agent_configs(agent_cfg: Any):
    yield agent_cfg
    for subagent in getattr(agent_cfg, "sub_agents", None) or []:
        yield from _iter_agent_configs(subagent)


async def _list_url_tools(server_name: str, url: str) -> list[ToolSpec]:
    from fastmcp import Client

    async with Client(url) as client:
        tools = await client.list_tools()

    result: list[ToolSpec] = []
    for tool in tools:
        input_schema = getattr(tool, "input_schema", None)
        if input_schema is None:
            input_schema = getattr(tool, "inputSchema", None)
        if input_schema is None:
            input_schema = {"type": "object", "properties": {}}
        name = str(tool.name)
        result.append(
            ToolSpec(
                server_name=server_name,
                tool_name=name,
                qualified_name=f"{server_name}:{name}",
                description=getattr(tool, "description", None),
                input_schema=dict(input_schema),
            )
        )
    return result


class LiveDtapCatalogProvider:
    def __init__(self, *, task_runtime_id: str):
        self.task_runtime_id = task_runtime_id

    async def list_victim_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]:
        from utils.mcp_helpers import start_task_mcp_servers
        from utils.resource_manager import ResourceManager

        manager = None
        try:
            manager = start_task_mcp_servers(
                snapshot.agent_config,
                self.task_runtime_id,
                snapshot.task_dir,
                ResourceManager.instance(),
            )
            if manager is None:
                return {}
            result: dict[str, Sequence[ToolSpec]] = {}
            for agent_cfg in _iter_agent_configs(snapshot.agent_config):
                for server in getattr(agent_cfg, "mcp_servers", None) or []:
                    if not getattr(server, "enabled", True):
                        continue
                    url = getattr(server, "url", None)
                    if url:
                        result[server.name] = await _list_url_tools(server.name, url)
            return result
        finally:
            if manager is not None:
                manager.stop_all()

    async def list_environment_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]:
        from utils.injection_mcp_helpers import start_injection_mcp_servers, wait_for_injection_mcp_ready
        from utils.resource_manager import ResourceManager

        if not snapshot.injection_config.get("environment_servers"):
            return {}

        manager = None
        cfg = copy.deepcopy(snapshot.injection_config)
        try:
            manager, updated = start_injection_mcp_servers(
                cfg,
                resource_manager=ResourceManager.instance(),
                task_id=self.task_runtime_id,
            )
            if manager is None:
                return {}
            wait_for_injection_mcp_ready(updated)
            result: dict[str, Sequence[ToolSpec]] = {}
            for server_name, server_info in (updated.get("environment_servers") or {}).items():
                if isinstance(server_info, dict) and server_info.get("url"):
                    result[server_name] = await _list_url_tools(server_name, server_info["url"])
            return result
        finally:
            if manager is not None:
                manager.stop_all()
