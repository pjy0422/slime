from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.authority import (
    EpisodeAuthority,
    EpisodeAuthorityRegistry,
    EpisodeCredentials,
    registered_authority,
)
from examples.dtap_agent_rl.service import EpisodeAccessError


def test_credentials_split_adapter_and_mcp_secrets_and_hide_repr():
    credentials = EpisodeCredentials.issue("adapter-session-123456")
    assert credentials.adapter_session_id != credentials.mcp_bearer_token
    assert len(credentials.mcp_bearer_token) >= 32
    assert credentials.mcp_bearer_token not in repr(credentials)


def test_atomic_registry_accepts_only_mcp_capability_and_revokes_it():
    credentials = EpisodeCredentials.issue("adapter-session-123456")
    authority = EpisodeAuthority(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    registry = EpisodeAuthorityRegistry(digest_key=b"k" * 32)

    with registered_authority(registry, credentials=credentials, authority=authority):
        assert registry.resolve(credentials.mcp_bearer_token) is authority
        with pytest.raises(EpisodeAccessError):
            registry.resolve(credentials.adapter_session_id)
        assert len(registry) == 1

    assert len(registry) == 0
    with pytest.raises(EpisodeAccessError):
        registry.resolve(credentials.mcp_bearer_token)


def test_registry_never_partially_overwrites_existing_authority():
    credentials = EpisodeCredentials.issue("adapter-session-123456")
    registry = EpisodeAuthorityRegistry(digest_key=b"x" * 32)
    first = EpisodeAuthority(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    second = EpisodeAuthority(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    registry.register(credentials, first)
    with pytest.raises(ValueError):
        registry.register(credentials, second)
    assert registry.resolve(credentials.mcp_bearer_token) is first
