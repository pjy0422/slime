from collections import deque
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.attack_surface import ToolSpec
from examples.dtap_agent_rl.attempt_runner import AttemptResult
from examples.dtap_agent_rl.episode import load_task_snapshot
from examples.dtap_agent_rl.episode_runtime import EpisodeStatus
from examples.dtap_agent_rl.m3 import run_m3_episode
from examples.dtap_agent_rl.service import EpisodeAccessError, EpisodeRegistry
from examples.dtap_agent_rl.submission import EpisodeSubmissionRegistry

from .conftest import FAKE_DTAP_API, write_config


SESSION = "episode-m3-wiring-0123456789abcdef"


class Catalog:
    async def list_victim_tools(self, snapshot):
        return {
            "slack": [
                ToolSpec("slack", "get_messages", "slack:get_messages", None, {"type": "object"})
            ]
        }

    async def list_environment_tools(self, snapshot):
        return {}


class Runner:
    def __init__(self):
        self.results = deque([AttemptResult(evaluation_started=True, attack_success=True)])
        self.calls = []

    async def run(self, workspace):
        self.calls.append(workspace)
        return self.results.popleft()


class SubmittingHarness:
    def __init__(self, views, submissions):
        self.views = views
        self.submissions = submissions
        self.calls = []

    async def run(self, sandbox, **kwargs):
        token = kwargs["session_id"]
        assert self.views.resolve(token).task.malicious_goal
        controller = self.submissions.resolve(token)
        receipt = await controller.submit({
            "steps": [{
                "type": "tool",
                "mode": "override",
                "content": "payload",
                "injected_tool": "slack:get_messages",
            }]
        })
        assert receipt["success"] is True and receipt["terminal"] is True
        self.calls.append(kwargs)
        return 0


@pytest.mark.asyncio
async def test_m3_wiring_keeps_one_policy_lifetime_for_submit_and_cleans_registries(tmp_path):
    task_parent = tmp_path / "dataset" / "workflow"
    task_parent.mkdir(parents=True)
    snapshot = load_task_snapshot(write_config(task_parent), dtap_api=FAKE_DTAP_API)
    views = EpisodeRegistry()
    submissions = EpisodeSubmissionRegistry()
    runner = Runner()
    harness = SubmittingHarness(views, submissions)

    result = await run_m3_episode(
        snapshot=snapshot,
        catalog_provider=Catalog(),
        view_registry=views,
        submission_registry=submissions,
        runner=runner,
        attempts_root=tmp_path / "attempts",
        max_submissions=2,
        sandbox=object(),
        session_id=SESSION,
        adapter_url="http://adapter:18001",
        prompt="inspect and submit",
        workdir="/workspace/empty",
        time_budget_sec=60,
        harness=harness,
        candidate_validator=lambda _config, *, expected_steps: SimpleNamespace(
            canonical_steps=expected_steps
        ),
    )

    assert result.harness_return_code == 0
    assert result.runtime.status is EpisodeStatus.SUCCEEDED
    assert result.runtime.final_reward == 1.0
    assert len(harness.calls) == 1
    assert "terminal=true" in harness.calls[0]["prompt"]
    assert "H=2" in harness.calls[0]["prompt"]
    assert len(runner.calls) == 1
    assert not runner.calls[0].attempt_dir.exists()
    with pytest.raises(EpisodeAccessError):
        views.resolve(SESSION)
    with pytest.raises(PermissionError):
        submissions.resolve(SESSION)
    snapshot.assert_config_unchanged()
