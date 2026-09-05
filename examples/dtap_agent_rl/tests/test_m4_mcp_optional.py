import importlib.util
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.attack_surface import AttackSurface
from examples.dtap_agent_rl.authority import (
    EpisodeAuthority,
    EpisodeAuthorityRegistry,
    EpisodeCredentials,
)
from examples.dtap_agent_rl.mcp_server import M4EpisodeService, create_m4_mcp_server
from examples.dtap_agent_rl.policy_contract import PolicyContract
from examples.dtap_agent_rl.service import EpisodeAccessError, EpisodeView
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.task_projection import PolicyTaskSpec


class Controller:
    def __init__(self):
        self.runtime = SimpleNamespace(
            terminal=False,
            remaining_submissions=2,
            record_security_failure=lambda **_kwargs: None,
        )
        self.calls = 0

    async def submit(self, plan):
        self.calls += 1
        return {
            "accepted": True,
            "submission": self.calls,
            "success": False,
            "terminal": False,
            "remaining_submissions": 1,
        }


def _authority():
    view = EpisodeView(
        PolicyTaskSpec("task", "workflow", "goal", "normal", "indirect", None),
        AttackSurface(True, False, False, False, (), (), (), prompt_modes=("suffix",)),
    )
    return EpisodeAuthority(view, Controller(), SimpleNamespace(set=lambda: None), PolicyContract())


@pytest.mark.asyncio
async def test_m4_service_routes_one_atomic_authority_and_rejects_adapter_id():
    credentials = EpisodeCredentials.issue("adapter-session-0123456789")
    registry = EpisodeAuthorityRegistry()
    authority = _authority()
    registry.register(credentials, authority)
    service = M4EpisodeService(registry, PolicyContract(), M4SecurityPolicy(max_submit_calls=2))

    assert service.get_task_spec(credentials.mcp_bearer_token)["task_id"] == "task"
    receipt = await service.submit_attack(credentials.mcp_bearer_token, {"steps": []})
    assert receipt["accepted"] is True
    with pytest.raises(EpisodeAccessError):
        service.get_task_spec(credentials.adapter_session_id)


def test_m4_validation_endpoint_applies_read_only_resource_limits():
    credentials = EpisodeCredentials.issue("adapter-session-0123456789")
    registry = EpisodeAuthorityRegistry()
    registry.register(credentials, _authority())
    service = M4EpisodeService(
        registry,
        PolicyContract(),
        M4SecurityPolicy(max_submit_calls=2, max_content_bytes=8),
    )
    result = service.validate_attack_step(
        credentials.mcp_bearer_token,
        {"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "x" * 100},
    )
    assert result == {
        "valid": False,
        "errors": [
            {"code": "INVALID_SHAPE", "path": "$", "message": "step exceeds public limits"}
        ],
    }


def test_m4_surface_discloses_whole_plan_resource_limits():
    credentials = EpisodeCredentials.issue("adapter-session-0123456789")
    registry = EpisodeAuthorityRegistry()
    registry.register(credentials, _authority())
    policy = M4SecurityPolicy(
        max_submit_calls=3,
        max_steps_per_plan=7,
        max_placement_actions=5,
    )
    surface = M4EpisodeService(registry, PolicyContract(), policy).get_attack_surface(
        credentials.mcp_bearer_token
    )

    assert surface["policy_limits"] == {
        "max_steps_per_plan": 7,
        "max_placement_actions": 5,
        "max_apply_attack_step_calls": 5,
        "max_submit_calls": 3,
    }


@pytest.mark.asyncio
async def test_m4_fastmcp_surface_remains_exactly_four_tools():
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("fastmcp not installed")
    from fastmcp import Client

    server = create_m4_mcp_server(
        EpisodeAuthorityRegistry(),
        security_policy=M4SecurityPolicy(max_submit_calls=2),
    )
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}
    assert names == {
        "get_task_spec",
        "get_attack_surface",
        "validate_attack_step",
        "submit_attack",
    }
