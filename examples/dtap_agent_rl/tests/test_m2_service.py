from examples.dtap_agent_rl.attack_surface import AttackSurface, ToolSpec
from examples.dtap_agent_rl.mcp_server import ReadOnlyEpisodeService
from examples.dtap_agent_rl.service import EpisodeRegistry, EpisodeView
from examples.dtap_agent_rl.task_projection import PolicyTaskSpec

TOKEN = "episode-m2-0123456789abcdef"


def view():
    return EpisodeView(
        task=PolicyTaskSpec(
            task_id="task",
            domain="workflow",
            malicious_goal="goal",
            task_instruction="normal task",
            threat_model="indirect",
        ),
        attack_surface=AttackSurface(
            prompt_enabled=True,
            tool_enabled=True,
            environment_enabled=True,
            skill_enabled=False,
            skill_modes=(),
            victim_tools=(ToolSpec("slack", "read", "slack:read", None, {"type": "object"}),),
            environment_tools=(
                ToolSpec(
                    "slack-injection",
                    "inject",
                    "slack-injection:inject",
                    None,
                    {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                ),
            ),
            prompt_modes=("suffix", "override"),
            tool_modes=("suffix", "override"),
        ),
    )


def test_service_exposes_schema_and_validates_without_registry_mutation():
    registry = EpisodeRegistry()
    registry.register(TOKEN, view())
    service = ReadOnlyEpisodeService(registry)
    surface = service.get_attack_surface(TOKEN)
    assert "candidate_step_schema" in surface
    before = len(registry)
    good = service.validate_attack_step(
        TOKEN,
        {
            "type": "environment",
            "turn_id": 1,
            "injection_mcp_tool": "slack-injection:inject",
            "kwargs": {"message": "x"},
        },
    )
    bad = service.validate_attack_step(
        TOKEN,
        {
            "type": "environment",
            "turn_id": 1,
            "injection_mcp_tool": "slack-injection:inject",
            "kwargs": {},
        },
    )
    assert good["valid"] is True
    assert bad["valid"] is False and bad["errors"][0]["code"] == "SCHEMA_MISMATCH"
    assert len(registry) == before
