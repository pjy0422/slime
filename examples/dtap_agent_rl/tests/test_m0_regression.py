import json

import pytest

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
    snapshot.assert_config_unchanged()
