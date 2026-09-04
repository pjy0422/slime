"""Atomic M4 episode authority with separated credentials."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .service import EpisodeAccessError, EpisodeView


@dataclass(frozen=True)
class EpisodeCredentials:
    adapter_session_id: str
    mcp_bearer_token: str = field(repr=False)
    public_episode_id: str

    @classmethod
    def issue(cls, adapter_session_id: str) -> "EpisodeCredentials":
        if not isinstance(adapter_session_id, str) or len(adapter_session_id.strip()) < 8:
            raise ValueError("adapter session id is invalid")
        return cls(
            adapter_session_id=adapter_session_id.strip(),
            mcp_bearer_token=secrets.token_urlsafe(32),
            public_episode_id=f"m4-{secrets.token_hex(16)}",
        )


@dataclass
class EpisodeAuthority:
    view: EpisodeView
    coordinator: Any
    terminal_event: Any
    policy_contract: Any = None


class EpisodeAuthorityRegistry:
    """One capability resolves the read view and mutation controller atomically."""

    def __init__(self, *, digest_key: bytes | None = None) -> None:
        self._digest_key = digest_key or secrets.token_bytes(32)
        if len(self._digest_key) < 32:
            raise ValueError("registry digest key must contain at least 256 bits")
        self._lock = threading.RLock()
        self._authorities: dict[bytes, EpisodeAuthority] = {}

    def _digest(self, token: str) -> bytes:
        if not isinstance(token, str) or len(token.strip()) < 32:
            raise EpisodeAccessError("unauthorized episode")
        return hmac.new(
            self._digest_key,
            token.strip().encode("utf-8"),
            hashlib.sha256,
        ).digest()

    def register(self, credentials: EpisodeCredentials, authority: EpisodeAuthority) -> None:
        digest = self._digest(credentials.mcp_bearer_token)
        if hmac.compare_digest(
            credentials.adapter_session_id.encode(),
            credentials.mcp_bearer_token.encode(),
        ):
            raise ValueError("adapter and MCP credentials must be distinct")
        with self._lock:
            if digest in self._authorities:
                raise ValueError("episode capability already registered")
            self._authorities[digest] = authority

    def resolve(self, token: str) -> EpisodeAuthority:
        digest = self._digest(token)
        with self._lock:
            authority = self._authorities.get(digest)
        if authority is None:
            raise EpisodeAccessError("unauthorized episode")
        return authority

    def unregister(self, token: str) -> None:
        digest = self._digest(token)
        with self._lock:
            self._authorities.pop(digest, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._authorities)


@contextmanager
def registered_authority(
    registry: EpisodeAuthorityRegistry,
    *,
    credentials: EpisodeCredentials,
    authority: EpisodeAuthority,
):
    registry.register(credentials, authority)
    try:
        yield
    finally:
        registry.unregister(credentials.mcp_bearer_token)
