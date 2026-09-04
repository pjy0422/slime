from collections import deque
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.attack_surface import ToolSpec
from examples.dtap_agent_rl.attempt_runner import AttemptResult
from examples.dtap_agent_rl.authority import EpisodeAuthorityRegistry
from examples.dtap_agent_rl.episode import load_task_snapshot
from examples.dtap_agent_rl.episode_runtime import EpisodeStatus
from examples.dtap_agent_rl.m4 import run_m4_episode
from examples.dtap_agent_rl.sandbox_policy import SandboxPolicyVerifier
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.service import EpisodeAccessError

from .conftest import FAKE_DTAP_API, write_config


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
    m4_hardened = True

    def __init__(self):
        self.results = deque([AttemptResult(evaluation_started=True, attack_success=True)])
        self.calls = []

    async def run(self, workspace):
        self.calls.append(workspace)
        return self.results.popleft()


class Sandbox:
    security_profile = {
        "non_root": True,
        "no_linux_capabilities": True,
        "read_only_root": True,
        "isolated_home": True,
        "isolated_workdir": True,
        "proc_isolated": True,
        "no_docker_socket": True,
        "no_host_workspace": True,
        "network_default_deny": True,
        "process_group_cleanup": True,
        "allowed_endpoints": ["http://adapter", "http://mcp"],
    }


class Harness:
    def __init__(self, token, registry):
        self.token = token
        self.registry = registry
        self.calls = []

    async def run(self, sandbox, **kwargs):
        authority = self.registry.resolve(self.token)
        with pytest.raises(EpisodeAccessError):
            self.registry.resolve(kwargs["session_id"])
        receipt = await authority.coordinator.submit(
            {
                "steps": [
                    {
                        "type": "tool",
                        "mode": "override",
                        "content": "payload",
                        "injected_tool": "slack:get_messages",
                    }
                ]
            }
        )
        assert receipt["success"] is True
        self.calls.append(kwargs)
        return 0


@pytest.mark.asyncio
async def test_m4_wiring_uses_split_atomic_authority_and_full_cleanup(tmp_path):
    parent = tmp_path / "dataset" / "workflow"
    parent.mkdir(parents=True)
    snapshot = load_task_snapshot(write_config(parent), dtap_api=FAKE_DTAP_API)
    registry = EpisodeAuthorityRegistry()
    runner = Runner()
    harnesses = []

    def harness_factory(token):
        harness = Harness(token, registry)
        harnesses.append(harness)
        return harness

    result = await run_m4_episode(
        snapshot=snapshot,
        catalog_provider=Catalog(),
        authority_registry=registry,
        runner=runner,
        attempts_root=tmp_path / "attempts",
        max_submissions=2,
        security_policy=M4SecurityPolicy(max_submit_calls=3),
        sandbox_verifier=SandboxPolicyVerifier(),
        sandbox=Sandbox(),
        adapter_session_id="adapter-session-0123456789abcdef",
        adapter_url="http://adapter",
        policy_mcp_url="http://mcp",
        prompt="inspect and attack",
        workdir="/workspace/empty",
        time_budget_sec=60,
        harness_factory=harness_factory,
        candidate_validator=lambda _config, *, expected_steps: SimpleNamespace(
            canonical_steps=expected_steps
        ),
    )

    assert result.runtime.status is EpisodeStatus.SUCCEEDED
    assert result.runtime.submit_calls == 1
    assert result.runtime.submissions_used == 1
    assert result.runtime.final_reward == 1.0
    assert len(registry) == 0
    assert len(runner.calls) == 1
    assert not runner.calls[0].attempt_dir.exists()
    assert "H=2; Q=3" in harnesses[0].calls[0]["prompt"]
    snapshot.assert_config_unchanged()


@pytest.mark.asyncio
async def test_m4_refuses_unattested_sandbox_before_registering_episode(tmp_path):
    parent = tmp_path / "dataset" / "workflow"
    parent.mkdir(parents=True)
    snapshot = load_task_snapshot(write_config(parent), dtap_api=FAKE_DTAP_API)
    registry = EpisodeAuthorityRegistry()
    with pytest.raises(Exception, match="attestation"):
        await run_m4_episode(
            snapshot=snapshot,
            catalog_provider=Catalog(),
            authority_registry=registry,
            runner=Runner(),
            attempts_root=tmp_path / "attempts",
            max_submissions=1,
            security_policy=M4SecurityPolicy(max_submit_calls=1),
            sandbox_verifier=SandboxPolicyVerifier(),
            sandbox=object(),
            adapter_session_id="adapter-session-0123456789abcdef",
            adapter_url="http://adapter",
            policy_mcp_url="http://mcp",
            prompt="attack",
            workdir="/workspace/empty",
            time_budget_sec=60,
        )
    assert len(registry) == 0
