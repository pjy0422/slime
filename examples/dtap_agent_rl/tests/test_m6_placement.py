from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.attack_surface import AttackSurface, ToolSpec
from examples.dtap_agent_rl.authority import EpisodeAuthority, EpisodeAuthorityRegistry, EpisodeCredentials
from examples.dtap_agent_rl.mcp_server import M4EpisodeService, create_m6_mcp_server
from examples.dtap_agent_rl.placement import DtapPlacementRunner, PlacementCoordinator, PlacementRunResult
from examples.dtap_agent_rl.policy_contract import PolicyContract
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.scheduler import AttemptScheduler
from examples.dtap_agent_rl.service import EpisodeView
from examples.dtap_agent_rl.task_projection import PolicyTaskSpec
from examples.dtap_agent_rl.validation import ValidationContext, validate_attack_step
import yaml
import json


STEP = {
    "type": "environment", "turn_id": 1,
    "injection_mcp_tool": "os-filesystem-injection:inject_file",
    "kwargs": {"file_path": "/tmp/target", "content": "payload"},
}


def view():
    tool = ToolSpec(
        "os-filesystem-injection", "inject_file",
        "os-filesystem-injection:inject_file", "write file",
        {"type": "object", "additionalProperties": False,
         "required": ["file_path", "content"],
         "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}},
    )
    return EpisodeView(
        PolicyTaskSpec("task", "workflow", "goal", "normal", "indirect", None),
        AttackSurface(False, False, True, False, (), (), (tool,)),
    )


class Runner:
    def __init__(self, result):
        self.result = result

    async def run(self, _workspace):
        return self.result


def coordinator(task_dir: Path, result: PlacementRunResult):
    contract = PolicyContract()
    policy = M4SecurityPolicy(max_submit_calls=2)
    return PlacementCoordinator(
        validation_context=ValidationContext.from_view(view()),
        source_task_dir=task_dir, episode_root=task_dir.parent / "m6-attempts",
        runner=Runner(result), security_policy=policy, policy_contract=contract,
        candidate_validator=lambda _config, *, expected_steps: SimpleNamespace(canonical_steps=expected_steps),
    )


@pytest.fixture
def task_dir(tmp_path):
    path = tmp_path / "source" / "dataset" / "workflow" / "task"
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump({
        "Task": {"task_id": "task"}, "Agent": {}, "RedTeamingAgent": {},
        "Attack": {"threat_model": "indirect", "malicious_goal": "goal"},
    }))
    return path


@pytest.mark.asyncio
async def test_apply_returns_opaque_receipt_and_validate_returns_verified_locator(task_dir):
    placement = coordinator(task_dir, PlacementRunResult(
        True, True, True, "verified", "os-filesystem:/tmp/target", "PLACEMENT_VERIFIED"
    ))
    receipt = await placement.apply(STEP)
    assert receipt["accepted"] is True
    assert receipt["action_id"].startswith("act_")
    result = placement.validate(receipt["action_id"])
    assert result == {
        "found": True, "valid": True, "status": "verified",
        "validated_placement_locator": "os-filesystem:/tmp/target",
    }
    validated = validate_attack_step(STEP, ValidationContext.from_view(view())).step
    assert validated is not None
    assert placement.unverified_environment_indices((validated,)) == ()
    # One owned receipt cannot authorize duplicated final actions.
    assert placement.unverified_environment_indices((validated, validated)) == (1,)


@pytest.mark.asyncio
async def test_unread_or_invalid_receipt_cannot_authorize_final_environment_step(task_dir):
    placement = coordinator(task_dir, PlacementRunResult(
        True, True, False, "invalid", "os-filesystem:/tmp/target",
        "PLACEMENT_MISMATCH", ("kwargs.file_path",),
    ))
    receipt = await placement.apply(STEP)
    validated = validate_attack_step(STEP, ValidationContext.from_view(view())).step
    assert validated is not None

    assert placement.unverified_environment_indices((validated,)) == (0,)
    placement.validate(receipt["action_id"])
    assert placement.unverified_environment_indices((validated,)) == (0,)


@pytest.mark.asyncio
async def test_invalid_placement_returns_only_targeted_repair_fields(task_dir):
    placement = coordinator(task_dir, PlacementRunResult(
        True, True, False, "invalid", "os-filesystem:/tmp/target",
        "PLACEMENT_MISMATCH", ("kwargs.file_path",),
    ))
    receipt = await placement.apply(STEP)
    result = placement.validate(receipt["action_id"])
    assert result["valid"] is False
    assert result["requested_placement_locator"] == "os-filesystem:/tmp/target"
    assert result["repair"] == {
        "expected_placement_locator": "os-filesystem:/tmp/target",
        "fields": ["kwargs.file_path"],
        "instruction": "change only the listed placement fields, then apply the revised action",
    }
    assert "payload" not in str(result)


@pytest.mark.asyncio
async def test_receipts_are_episode_scoped_and_non_environment_steps_are_not_applied(task_dir):
    result = PlacementRunResult(True, True, True, "verified", "os-filesystem:/tmp/target")
    first, second = coordinator(task_dir, result), coordinator(task_dir, result)
    receipt = await first.apply(STEP)
    assert second.validate(receipt["action_id"])["error"]["code"] == "UNKNOWN_ACTION"
    rejected = await first.apply({"type": "prompt", "turn_id": 1, "mode": "suffix", "content": "x"})
    assert rejected["accepted"] is False


@pytest.mark.asyncio
async def test_m6_service_and_fastmcp_surface_have_six_tools(task_dir):
    if pytest.importorskip("fastmcp") is None:
        return
    from fastmcp import Client

    contract = PolicyContract()
    placement = coordinator(task_dir, PlacementRunResult(
        True, True, True, "verified", "os-filesystem:/tmp/target"
    ))
    controller = SimpleNamespace(
        runtime=SimpleNamespace(terminal=False, remaining_submissions=2),
        submit=lambda _plan: None,
    )
    credentials = EpisodeCredentials.issue("adapter-session-0123456789")
    registry = EpisodeAuthorityRegistry()
    registry.register(credentials, EpisodeAuthority(
        view(), controller, SimpleNamespace(set=lambda: None), contract, placement
    ))
    service = M4EpisodeService(registry, contract, M4SecurityPolicy(max_submit_calls=2))
    receipt = await service.apply_attack_step(credentials.mcp_bearer_token, STEP)
    assert service.validate_placement(credentials.mcp_bearer_token, receipt["action_id"])["valid"] is True

    server = create_m6_mcp_server(registry, security_policy=M4SecurityPolicy(max_submit_calls=2))
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}
    assert names == {"get_task_spec", "get_attack_surface", "validate_attack_step",
                     "apply_attack_step", "validate_placement", "submit_attack"}


def test_placement_result_firewall_rejects_symlinks_and_unknown_fields(tmp_path):
    policy = M4SecurityPolicy(max_submit_calls=1, max_parallel_attempts=1, max_queued_attempts=1)
    runner = DtapPlacementRunner(
        dtap_root=tmp_path, security_policy=policy,
        scheduler=AttemptScheduler(max_parallel=1, max_queued=1, wait_timeout=1),
    )
    root = tmp_path / "results"
    root.mkdir()
    real = root / "real.json"
    real.write_text(json.dumps({
        "schema": "m6-placement-v1", "applied": True, "valid": True,
        "status": "verified", "locator": "os-filesystem:/tmp/a",
        "code": "PLACEMENT_VERIFIED", "unexpected": "backend-state",
    }))
    assert runner._read(real, root).available is False
    link = root / "link.json"
    link.symlink_to(real)
    assert runner._read(link, root).available is False
