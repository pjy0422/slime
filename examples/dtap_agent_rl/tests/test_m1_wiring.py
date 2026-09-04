import pytest

from examples.dtap_agent_rl.attack_surface import ToolSpec
from examples.dtap_agent_rl.episode import load_task_snapshot
from examples.dtap_agent_rl.m1 import run_m1_episode
from examples.dtap_agent_rl.service import EpisodeAccessError, EpisodeRegistry

from .conftest import FAKE_DTAP_API, write_config


SESSION = "episode-wiring-0123456789abcdef"


class Catalog:
    async def list_victim_tools(self, snapshot):
        return {"slack": [ToolSpec("slack", "read", "slack:read", None, {"type": "object"})]}

    async def list_environment_tools(self, snapshot):
        return {
            "slack-injection": [
                ToolSpec(
                    "slack-injection",
                    "inject_slack_message",
                    "slack-injection:inject_slack_message",
                    None,
                    {"type": "object", "properties": {"message": {"type": "string"}}},
                )
            ]
        }


class InspectingHarness:
    def __init__(self, registry, *, fail=False):
        self.registry = registry
        self.fail = fail
        self.calls = []

    async def run(self, sandbox, **kwargs):
        # Registration must already exist while the policy runtime is active.
        view = self.registry.resolve(kwargs["session_id"])
        assert view.task.malicious_goal
        assert view.attack_surface.environment_tools
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("policy runtime failed")
        return 0


@pytest.mark.asyncio
async def test_run_m1_episode_registers_only_for_policy_lifetime(tmp_path):
    snapshot = load_task_snapshot(write_config(tmp_path), dtap_api=FAKE_DTAP_API)
    registry = EpisodeRegistry()
    harness = InspectingHarness(registry)

    rc = await run_m1_episode(
        snapshot=snapshot,
        catalog_provider=Catalog(),
        registry=registry,
        sandbox=object(),
        session_id=SESSION,
        adapter_url="http://adapter:18001",
        prompt="inspect the task",
        workdir="/workspace/empty",
        time_budget_sec=60,
        harness=harness,
    )

    assert rc == 0
    assert len(harness.calls) == 1
    assert len(registry) == 0
    with pytest.raises(EpisodeAccessError):
        registry.resolve(SESSION)
    snapshot.assert_config_unchanged()


@pytest.mark.asyncio
async def test_run_m1_episode_cleans_up_when_policy_runtime_fails(tmp_path):
    snapshot = load_task_snapshot(write_config(tmp_path), dtap_api=FAKE_DTAP_API)
    registry = EpisodeRegistry()
    harness = InspectingHarness(registry, fail=True)

    with pytest.raises(RuntimeError, match="policy runtime failed"):
        await run_m1_episode(
            snapshot=snapshot,
            catalog_provider=Catalog(),
            registry=registry,
            sandbox=object(),
            session_id=SESSION,
            adapter_url="http://adapter:18001",
            prompt="inspect",
            workdir="/workspace/empty",
            time_budget_sec=60,
            harness=harness,
        )
    assert len(registry) == 0
