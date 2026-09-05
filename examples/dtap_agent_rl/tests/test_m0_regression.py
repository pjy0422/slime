import json

import pytest
import yaml

from examples.dtap_agent_rl.attack_surface import ToolSpec, build_attack_surface
from examples.dtap_agent_rl.episode import load_task_snapshot
from examples.dtap_agent_rl.task_projection import project_task

from .conftest import FAKE_DTAP_API, write_config


class Catalog:
    async def list_victim_tools(self, snapshot):
        return {
            "slack": [
                ToolSpec("slack", "get_messages", "slack:get_messages", None, {"type": "object"})
            ]
        }

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
            ],
            "gmail-injection": [
                ToolSpec("gmail-injection", "inject_email", "gmail-injection:inject_email", None, {})
            ],
        }


class MaintenanceCatalog(Catalog):
    async def list_environment_tools(self, snapshot):
        catalog = dict(await super().list_environment_tools(snapshot))
        catalog["slack-injection"] = [
            *catalog["slack-injection"],
            ToolSpec("slack-injection", "get_status", "slack-injection:get_status", None, {}),
            ToolSpec("slack-injection", "clear_all", "slack-injection:clear_all", None, {}),
        ]
        return catalog


@pytest.mark.asyncio
async def test_m0_projection_and_surface_still_do_not_leak_example(tmp_path):
    snapshot = load_task_snapshot(write_config(tmp_path), dtap_api=FAKE_DTAP_API)
    task = project_task(snapshot)
    surface = await build_attack_surface(snapshot, Catalog())
    payload = json.dumps({"task": task.to_dict(), "surface": surface.to_dict()})

    assert "PRIVATE_EXAMPLE_PAYLOAD" not in payload
    assert "PRIVATE_SYSTEM_PROMPT" not in payload
    assert "gmail-injection" not in payload
    assert "slack-injection:inject_slack_message" in payload
    assert surface.to_dict()["plan_constraints"] == []
    snapshot.assert_config_unchanged()


@pytest.mark.asyncio
async def test_direct_surface_exposes_cross_step_turn_constraints(tmp_path):
    task_dir = write_config(tmp_path)
    config_path = task_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["Attack"]["threat_model"] = "direct"
    config["RedTeamingAgent"]["available_injections"]["prompt"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    snapshot = load_task_snapshot(task_dir, dtap_api=FAKE_DTAP_API)
    constraints = (await build_attack_surface(snapshot, Catalog())).to_dict()[
        "plan_constraints"
    ]

    assert constraints == [
        "direct plans require at least one jailbreak prompt",
        "direct jailbreak prompt turn_ids must be contiguous from 1",
        "environment turn_id must not exceed the direct jailbreak prompt count",
    ]


@pytest.mark.asyncio
async def test_surface_excludes_environment_observation_and_maintenance_tools(tmp_path):
    snapshot = load_task_snapshot(write_config(tmp_path), dtap_api=FAKE_DTAP_API)
    surface = await build_attack_surface(snapshot, MaintenanceCatalog())

    assert [tool.tool_name for tool in surface.environment_tools] == [
        "inject_slack_message"
    ]
