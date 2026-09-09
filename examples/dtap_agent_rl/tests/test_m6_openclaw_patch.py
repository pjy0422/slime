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
JUDGE_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "p3-judge-reliability.patch"
)
LIVE_STABILITY_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "p4-h2-live-stability.patch"
)
M7_OBSERVABILITY_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-feedback-observability.patch"
)
M7_FEEDBACK_V2_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-feedback-v2.patch"
)
M7_EXACT_LOCATORS_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-exact-locators.patch"
)
M7_DOMAIN_FEEDBACK_PATCH = (
    Path(__file__).parents[1] / "dtap_integration" / "patches" / "m7-domain-feedback.patch"
)
M7_BOUNDARY_MATRIX_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-feedback-boundary-matrix.patch"
)
TOKEN_OBSERVABILITY_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-token-observability.patch"
)
TOOL_CAPABILITY_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-explicit-tool-capabilities.patch"
)
STRUCTURED_JUDGE_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-structured-judge-status.patch"
)
EXACT_TOOL_IDENTITY_PATCH = (
    Path(__file__).parents[1]
    / "dtap_integration"
    / "patches"
    / "m7-exact-tool-presentation-identity.patch"
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


def test_openclaw_token_usage_patch_preserves_provider_receipt():
    text = TOKEN_OBSERVABILITY_PATCH.read_text(encoding="utf-8")

    assert "def _openclaw_token_usage(" in text
    assert 'agent_meta.get("diagnosticUsage")' in text
    assert 'agent_meta.get("lastCallUsage")' in text
    assert 'count("reasoningTokens", "reasoning_tokens")' in text
    assert 'trace_metadata["token_usage"] = self._token_usage' in text
    assert 'trace_metadata["model"] = model.strip()' in text
    assert '**({"usage": self._token_usage} if self._token_usage else {})' in text
    assert 'metadata"]["token_usage"] = dict(' in text
    assert 'metadata"]["model"] = meta["model"].strip()' in text
    assert text.index('agent_meta.get("diagnosticUsage")') < text.index(
        'agent_meta.get("usage")'
    )


def test_openclaw_patch_forwards_the_shared_provider_to_medical_aux_models():
    text = PATCH.read_text(encoding="utf-8")

    assert text.count("OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}") == 2
    assert 'OPENCLAW_MCP_TOOL_TIMEOUT_SECONDS' in text
    assert "timeout=timeout_seconds" in text


def test_hospital_patch_routes_arbitrary_models_through_explicit_compat_provider():
    text = JUDGE_PATCH.read_text(encoding="utf-8")

    assert "DTAP_HOSPITAL_OPENAI_COMPAT" in text
    assert "hospital.query_model = _query_openai_compatible" in text
    assert 'base_url=os.environ["OPENAI_BASE_URL"]' in text
    assert "hospital_compat.py:/app/dtap_hospital_compat.py:ro" in text
    assert 'if base_url.endswith("/v1")' in text
    assert 'os.getenv("JUDGE_MODEL", "gpt-5.4")' in text


def test_live_stability_patch_covers_generated_finance_ids_and_judge_timeout():
    text = LIVE_STABILITY_PATCH.read_text(encoding="utf-8")

    assert 're.search(r"Article ID:\\s*(\\d+)"' in text
    assert "DTAP_JUDGE_HTTP_TIMEOUT_SECONDS" in text
    assert "requests.sessions.Session.request = original" in text


def test_overlay_marker_is_scoped_to_the_target_dtap_checkout():
    apply_script = JUDGE_PATCH.parents[1] / "apply.sh"
    text = apply_script.read_text(encoding="utf-8")

    assert "--path-format=absolute --git-path" in text
    assert 'incremental_patches=(' in text
    assert 'for patch_file in "${incremental_patches[@]}"' in text
    assert 'latest_patch=' not in text


def test_stabilization_patch_records_only_redacted_correlated_mcp_events():
    text = STABILIZATION_PATCH.read_text(encoding="utf-8")

    assert "class MCPEventSink" in text
    assert '"dtap-openclaw-mcp-event"' in text
    assert "_redacted_shape(arguments)" in text
    assert 'DTAP_EVALUATION_EPISODE_ID' in text
    assert "raw sensitive prompt payload" in text
    assert "assert all(secret not in raw" in text


def test_m7_completion_marker_requires_a_healthy_normally_stopped_sink():
    text = M7_OBSERVABILITY_PATCH.read_text(encoding="utf-8")

    assert 'self.emit("trace.completed")' in text
    assert "if self._healthy and not self._completed" in text
    assert "stopped_cleanly = True" in text
    assert "if self._event_sink and stopped_cleanly" in text


def test_m7_v2_overlay_observes_provider_boundaries_without_retaining_payloads():
    text = M7_FEEDBACK_V2_PATCH.read_text(encoding="utf-8")

    assert '"feedback.observed"' in text
    assert "build_static_feedback_probes" in text
    assert 'event_type == "prompt.submitted"' in text
    assert 'event_type == "context.compiled"' in text
    assert 'data.get("availableTools")' in text
    assert 'data.get("systemPrompt")' in text
    assert "payload_values" in text
    assert "never written to the retained MCP event stream" in text


def test_m7_exact_locator_overlay_uses_receipt_ids_and_rejects_ambiguous_names():
    text = M7_EXACT_LOCATORS_PATCH.read_text(encoding="utf-8")

    assert "def _exact_locator_patterns(" in text
    assert '"tool": "salesforce:get_record"' in text
    assert '"tool": "legal:get_matter"' in text
    assert '"tool": "travel:query_flight"' in text
    assert '"tool": "telecom:query_email"' in text
    assert '"filepath": kwargs[source_field]' in text
    assert "test_salesforce_without_receipt_id_does_not_claim_exact_name_locator" in text
    assert "test_filtered_or_paginated_result_does_not_count_as_payload_read" in text
    assert "test_stale_record_with_same_locator_does_not_match_current_payload" in text


def test_m7_domain_feedback_distinguishes_exact_collection_and_payload_proofs():
    text = M7_DOMAIN_FEEDBACK_PATCH.read_text(encoding="utf-8")

    assert 'return "collection_locator"' in text
    assert 'return "exact_locator"' in text
    assert '"tool": "calendar:get_event"' in text
    assert '"tool": "zoom:meetings_get"' in text
    assert '"tool": "whatsapp:get_whatsapp_chat"' in text
    assert '"tool": "github:get_issue"' in text
    assert "self._match_basis.get(index" in text
    assert "_PAYLOAD_READ_PATTERNS" in text
    assert "test_payload_only_adapter_ignores_same_service_write_echo" in text


def test_m7_v2_overlay_explicitly_covers_linux_registry_only():
    text = M7_FEEDBACK_V2_PATCH.read_text(encoding="utf-8")

    assert "SUPPORTED_FEEDBACK_TOOLS" in text
    assert "assert len(SUPPORTED_FEEDBACK_TOOLS) == 25" in text
    assert "assert checked == 115" in text
    assert 'if server not in {"windows-injection", "macos-injection"}' in text


def test_m7_boundary_matrix_uses_alternate_records_and_real_mcp_namespaces():
    text = M7_BOUNDARY_MATRIX_PATCH.read_text(encoding="utf-8")

    assert 'DOMAINS = (' in text
    assert 'assert len(CASES) == 24' in text
    assert 'assert case.benchmark_index > 0' in text
    assert 'response_contains_injection' in text
    assert 'presented_to_model' in text
    assert '"customer_service"' in text
    assert '"Research"' in text
    assert '"travel-suite"' in text
    assert '"HospitalClient:get_patient_status"' in text
    assert 'test_customer_service_batch_fallback_cannot_treat_write_echo_as_read' in text
    assert 'Serializing a tool result before matching changes newlines' in text


def test_stabilization_patch_covers_linux_mutators_and_disables_missing_sources():
    text = STABILIZATION_PATCH.read_text(encoding="utf-8")

    assert "SUPPORTED_PLACEMENT_TOOLS" in text
    assert "test_every_supported_mutator_has_positive_and_mismatch_contract" in text
    assert "# Public release has no" in text
    assert '"windows-injection", SUPPORTED_PLACEMENT_TOOLS' in text
    assert '"macos-injection", SUPPORTED_PLACEMENT_TOOLS' in text


def test_environment_tool_capability_patch_has_no_prefix_authorization():
    text = TOOL_CAPABILITY_PATCH.read_text(encoding="utf-8")
    added = "\n".join(
        line[1:] for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )

    assert "NON_PLACEMENT_TOOLS" in text
    assert "tool.startswith" not in added
    assert "PlacementStatus.UNSUPPORTED" in text
    assert "get_new_chat" in text


def test_judge_status_patch_emits_structured_availability():
    text = STRUCTURED_JUDGE_PATCH.read_text(encoding="utf-8")

    assert '"task_status"' in text
    assert '"attack_status"' in text
    assert "_judge_stage_status" in text
    assert "Classify availability from structured judge output, never prose" in text


def test_tool_presentation_patch_requires_exact_server_and_tool_identity():
    text = EXACT_TOOL_IDENTITY_PATCH.read_text(encoding="utf-8")

    assert 'parts[-2:] == [target_server, target_name]' in text
    assert "description mentions search_emails" in text
    assert 'target in value or target.split' in text
