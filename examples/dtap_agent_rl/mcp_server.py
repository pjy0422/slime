"""Read-only, host-side FastMCP surface for M1.

The policy sees exactly two tools:
  - get_task_spec()
  - get_attack_surface()

Episode identity is carried in the HTTP Authorization header by Claude Code's
MCP transport. It is intentionally absent from tool schemas and arguments.
"""

from __future__ import annotations

from typing import Mapping

from .service import EpisodeAccessError, EpisodeRegistry, EpisodeView


def parse_bearer_token(headers: Mapping[str, str]) -> str:
    """Parse one opaque bearer capability without exposing parsing detail."""

    value = headers.get("authorization") or headers.get("Authorization") or ""
    scheme, sep, token = value.partition(" ")
    if not sep or scheme.lower() != "bearer" or len(token.strip()) < 16:
        raise EpisodeAccessError("unauthorized episode")
    return token.strip()


class ReadOnlyEpisodeService:
    """Framework-independent service used by both MCP tools and unit tests."""

    def __init__(self, registry: EpisodeRegistry):
        self.registry = registry

    def get_task_spec(self, token: str) -> dict:
        return self.registry.resolve(token).task.to_dict()

    def get_attack_surface(self, token: str) -> dict:
        return self.registry.resolve(token).attack_surface.to_dict()


def create_mcp_server(registry: EpisodeRegistry):
    """Create a stateless HTTP FastMCP server bound to an EpisodeRegistry."""

    try:
        from fastmcp import FastMCP
        from fastmcp.server.dependencies import get_http_headers
    except ImportError as exc:  # pragma: no cover - depends on DTAP runtime env
        raise RuntimeError(
            "FastMCP >= 2.6 is required for M1 HTTP bearer episode routing."
        ) from exc

    service = ReadOnlyEpisodeService(registry)
    mcp = FastMCP(name="DTAP RL Harness", stateless_http=True)

    def current_token() -> str:
        headers = get_http_headers() or {}
        return parse_bearer_token(headers)

    @mcp.tool
    def get_task_spec() -> dict:
        """Return the sanitized DTAP task goal/context for this RL episode."""

        return service.get_task_spec(current_token())

    @mcp.tool
    def get_attack_surface() -> dict:
        """Return task-authorized injection mechanisms and MCP input schemas."""

        return service.get_attack_surface(current_token())

    return mcp
