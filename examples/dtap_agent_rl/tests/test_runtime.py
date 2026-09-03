import pytest

from examples.dtap_agent_rl.runtime import registered_episode
from examples.dtap_agent_rl.service import EpisodeAccessError, EpisodeRegistry, EpisodeView

from .conftest import sample_surface, sample_task_spec


TOKEN = "episode-runtime-0123456789"


def test_registered_episode_cleans_up_success_and_exception():
    registry = EpisodeRegistry()
    view = EpisodeView(task=sample_task_spec(), attack_surface=sample_surface())

    with registered_episode(registry, token=TOKEN, view=view):
        assert registry.resolve(TOKEN) == view
    with pytest.raises(EpisodeAccessError):
        registry.resolve(TOKEN)

    with pytest.raises(RuntimeError):
        with registered_episode(registry, token=TOKEN, view=view):
            raise RuntimeError("boom")
    with pytest.raises(EpisodeAccessError):
        registry.resolve(TOKEN)
