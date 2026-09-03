import pytest

from examples.dtap_agent_rl.mcp_server import ReadOnlyEpisodeService, parse_bearer_token
from examples.dtap_agent_rl.service import EpisodeAccessError, EpisodeRegistry, EpisodeView

from .conftest import sample_surface, sample_task_spec


TOKEN_A = "episode-a-0123456789abcdef"
TOKEN_B = "episode-b-0123456789abcdef"


def view():
    return EpisodeView(task=sample_task_spec(), attack_surface=sample_surface())


def test_registry_isolated_lifecycle():
    registry = EpisodeRegistry()
    registry.register(TOKEN_A, view())
    registry.register(TOKEN_B, view())
    assert len(registry) == 2
    assert registry.resolve(TOKEN_A).task.task_id == "task-001"

    registry.unregister(TOKEN_A)
    assert len(registry) == 1
    with pytest.raises(EpisodeAccessError, match="unauthorized episode"):
        registry.resolve(TOKEN_A)
    assert registry.resolve(TOKEN_B).task.task_id == "task-001"


def test_registry_rejects_duplicate_or_weak_tokens():
    registry = EpisodeRegistry()
    registry.register(TOKEN_A, view())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(TOKEN_A, view())
    with pytest.raises(EpisodeAccessError, match="unauthorized episode"):
        registry.register("short", view())


def test_read_only_service_never_returns_token_or_private_registry_state():
    registry = EpisodeRegistry()
    registry.register(TOKEN_A, view())
    service = ReadOnlyEpisodeService(registry)

    task = service.get_task_spec(TOKEN_A)
    surface = service.get_attack_surface(TOKEN_A)
    assert task["malicious_goal"]
    assert surface["environment_tools"]
    assert TOKEN_A not in repr(task)
    assert TOKEN_A not in repr(surface)


def test_bearer_parser():
    assert parse_bearer_token({"Authorization": f"Bearer {TOKEN_A}"}) == TOKEN_A
    assert parse_bearer_token({"authorization": f"bearer {TOKEN_A}"}) == TOKEN_A
    for bad in [{}, {"Authorization": TOKEN_A}, {"Authorization": "Basic xxx"}, {"Authorization": "Bearer x"}]:
        with pytest.raises(EpisodeAccessError, match="unauthorized episode"):
            parse_bearer_token(bad)
