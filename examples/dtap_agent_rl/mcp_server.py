"""Host-side FastMCP surface for M0-M2.

M2 remains read-only with respect to DTAP state. validate_attack_step() validates
only the already-sanitized immutable EpisodeView; it performs no Docker, MCP,
filesystem, config, victim-agent, or judge mutation.
"""

from __future__ import annotations

from typing import Any, Mapping

from .actions import candidate_attack_step_schema
from .service import EpisodeAccessError, EpisodeRegistry
from .validation import ValidationContext, validate_attack_step as validate_candidate_step


def parse_bearer_token(headers: Mapping[str, str]) -> str:
    value = headers.get("authorization") or headers.get("Authorization") or ""
    scheme, sep, token = value.partition(" ")
    if not sep or scheme.lower() != "bearer" or len(token.strip()) < 16:
        raise EpisodeAccessError("unauthorized episode")
    return token.strip()


class ReadOnlyEpisodeService:
    def __init__(self, registry: EpisodeRegistry):
        self.registry = registry

    def get_task_spec(self, token: str) -> dict:
        return self.registry.resolve(token).task.to_dict()

    def get_attack_surface(self, token: str) -> dict:
        view = self.registry.resolve(token)
        result = view.attack_surface.to_dict()
        result["candidate_step_schema"] = candidate_attack_step_schema()
        return result

    def validate_attack_step(self, token: str, step: dict[str, Any]) -> dict:
        view = self.registry.resolve(token)
        ctx = ValidationContext.from_view(view)
        return validate_candidate_step(step, ctx).to_dict()


def create_mcp_server(registry: EpisodeRegistry):
    try:
        from fastmcp import FastMCP
        from fastmcp.server.dependencies import get_http_headers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FastMCP >= 2.6 is required for M1/M2 HTTP bearer routing") from exc

    service = ReadOnlyEpisodeService(registry)
    mcp = FastMCP(name="DTAP RL Harness", stateless_http=True)

    def current_token() -> str:
        return parse_bearer_token(get_http_headers() or {})

    @mcp.tool
    def get_task_spec() -> dict:
        """Return the sanitized DTAP task goal/context for this RL episode."""
        return service.get_task_spec(current_token())

    @mcp.tool
    def get_attack_surface() -> dict:
        """Return allowed attack mechanisms, targets, schemas, and M2 action contract."""
        return service.get_attack_surface(current_token())

    @mcp.tool
    def validate_attack_step(step: dict[str, Any]) -> dict:
        """Strictly validate one candidate attack step without applying it."""
        return service.validate_attack_step(current_token(), step)

    return mcp
