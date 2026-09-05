from pathlib import Path


PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m6-openclaw-deepseek.patch"
)
STABILIZATION_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "p0-p2-linux-stabilization.patch"
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


def test_stabilization_patch_records_only_redacted_correlated_mcp_events():
    text = STABILIZATION_PATCH.read_text(encoding="utf-8")

    assert "class MCPEventSink" in text
    assert '"dtap-openclaw-mcp-event"' in text
    assert "_redacted_shape(arguments)" in text
    assert 'DTAP_EVALUATION_EPISODE_ID' in text
    assert "raw sensitive prompt payload" in text
    assert "assert all(secret not in raw" in text


def test_stabilization_patch_covers_linux_mutators_and_disables_missing_sources():
    text = STABILIZATION_PATCH.read_text(encoding="utf-8")

    assert "SUPPORTED_PLACEMENT_TOOLS" in text
    assert "test_every_supported_mutator_has_positive_and_mismatch_contract" in text
    assert "# Public release has no" in text
    assert '"windows-injection", SUPPORTED_PLACEMENT_TOOLS' in text
    assert '"macos-injection", SUPPORTED_PLACEMENT_TOOLS' in text
