from pathlib import Path


PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m6-openclaw-deepseek.patch"
)


def test_openclaw_patch_is_isolated_and_keeps_secrets_out_of_config():
    text = PATCH.read_text(encoding="utf-8")

    assert "openclaw_config: Dict[str, Any] = {}" in text
    assert '"id": "ANTHROPIC_API_KEY"' in text
    assert '"authHeader": True' in text
    assert "ANTHROPIC_API_KEY'," not in text
    assert "_load_openclaw_config()" not in "\n".join(
        line[1:] for line in text.splitlines() if line.startswith("+")
    )


def test_openclaw_patch_supports_current_cli_envelope_and_trajectory_fallback():
    text = PATCH.read_text(encoding="utf-8")

    assert '+            "--json",' in text
    assert "self._last_cli_payload = result" in text
    assert '"type": "prompt.submitted"' in text
    assert '"type": "model.completed"' in text
    assert '"type": "session.ended"' in text


def test_openclaw_patch_forwards_the_shared_provider_to_medical_aux_models():
    text = PATCH.read_text(encoding="utf-8")

    assert text.count("OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}") == 2
    assert 'OPENCLAW_MCP_TOOL_TIMEOUT_SECONDS' in text
    assert "timeout=timeout_seconds" in text
