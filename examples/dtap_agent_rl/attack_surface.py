from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from .episode import TaskSnapshot


@dataclass(frozen=True)
class ToolSpec:
    server_name: str
    tool_name: str
    qualified_name: str
    description: str | None
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttackSurface:
    prompt_enabled: bool
    tool_enabled: bool
    environment_enabled: bool
    skill_enabled: bool
    skill_modes: tuple[str, ...]
    victim_tools: tuple[ToolSpec, ...]
    environment_tools: tuple[ToolSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "injections": {
                "prompt": self.prompt_enabled,
                "tool": self.tool_enabled,
                "environment": self.environment_enabled,
                "skill": self.skill_enabled,
                "skill_modes": list(self.skill_modes),
            },
            "victim_tools": [tool.to_dict() for tool in self.victim_tools],
            "environment_tools": [tool.to_dict() for tool in self.environment_tools],
        }


class ToolCatalogProvider(Protocol):
    async def list_victim_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]: ...
    async def list_environment_tools(self, snapshot: TaskSnapshot) -> Mapping[str, Sequence[ToolSpec]]: ...


def _iter_raw_agent_servers(agent: Mapping[str, Any]):
    for server in agent.get("mcp_servers") or []:
        if isinstance(server, Mapping):
            yield server
    for subagent in agent.get("sub_agents") or []:
        if isinstance(subagent, Mapping):
            yield from _iter_raw_agent_servers(subagent)


def _victim_server_rules(snapshot: TaskSnapshot) -> dict[str, set[str]]:
    rules: dict[str, set[str]] = {}
    raw_agent = snapshot.raw_config.get("Agent") or {}
    if not isinstance(raw_agent, Mapping):
        return rules
    for server in _iter_raw_agent_servers(raw_agent):
        if not server.get("enabled", True):
            continue
        name = str(server.get("name") or "").strip()
        if not name:
            continue
        rules[name] = {str(item) for item in (server.get("tool_blacklist") or []) if item is not None}
    return rules


def _environment_server_rules(snapshot: TaskSnapshot) -> dict[str, str | set[str]]:
    raw = snapshot.injection_config.get("environment_servers") or {}
    if not isinstance(raw, Mapping):
        return {}
    rules: dict[str, str | set[str]] = {}
    for server_name, allowed in raw.items():
        name = str(server_name)
        if allowed == "all":
            rules[name] = "all"
        elif isinstance(allowed, (list, tuple, set)):
            rules[name] = {str(tool) for tool in allowed}
        elif isinstance(allowed, Mapping) and "tools" in allowed:
            tools = allowed.get("tools")
            rules[name] = "all" if tools == "all" else {str(t) for t in (tools or [])}
        else:
            rules[name] = set()
    return rules


def _dedupe_sorted(tools: list[ToolSpec]) -> tuple[ToolSpec, ...]:
    by_name = {tool.qualified_name: tool for tool in tools}
    return tuple(by_name[key] for key in sorted(by_name))


async def build_attack_surface(snapshot: TaskSnapshot, provider: ToolCatalogProvider) -> AttackSurface:
    cfg = snapshot.injection_config

    victim_tools: list[ToolSpec] = []
    if bool(cfg.get("tool_enabled", False)):
        catalog = await provider.list_victim_tools(snapshot)
        for server_name, blacklist in _victim_server_rules(snapshot).items():
            for tool in catalog.get(server_name, ()):
                if tool.tool_name not in blacklist:
                    victim_tools.append(tool)

    environment_tools: list[ToolSpec] = []
    if bool(cfg.get("environment_enabled", False)):
        catalog = await provider.list_environment_tools(snapshot)
        for server_name, allowed in _environment_server_rules(snapshot).items():
            for tool in catalog.get(server_name, ()):
                if allowed == "all" or tool.tool_name in allowed:
                    environment_tools.append(tool)

    return AttackSurface(
        prompt_enabled=bool(cfg.get("prompt_enabled", False)),
        tool_enabled=bool(cfg.get("tool_enabled", False)),
        environment_enabled=bool(cfg.get("environment_enabled", False)),
        skill_enabled=bool(cfg.get("skill_enabled", False)),
        skill_modes=tuple(str(x) for x in (cfg.get("skill_modes") or [])),
        victim_tools=_dedupe_sorted(victim_tools),
        environment_tools=_dedupe_sorted(environment_tools),
    )
