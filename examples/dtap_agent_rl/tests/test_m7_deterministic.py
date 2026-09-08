import hashlib
import json

from examples.dtap_agent_rl.actions import ValidatedAttackStep
from examples.dtap_agent_rl.feedback.deterministic import (
    extract_deterministic_feedback,
    parse_mcp_events,
)


def _shape(value):
    return {
        "type": "string",
        "length": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _write_events(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_parser_joins_status_in_started_order_and_marks_incomplete(tmp_path):
    path = tmp_path / "victim-mcp-events.jsonl"
    _write_events(
        path,
        [
            {"type": "tool.started", "call_id": "a", "server": "slack", "tool": "one", "arguments": {}},
            {"type": "tool.completed", "call_id": "a", "server": "slack", "tool": "one", "is_error": False},
            {"type": "tool.started", "call_id": "b", "server": "slack", "tool": "two", "arguments": {}},
            {"type": "tool.completed", "call_id": "b", "server": "slack", "tool": "two", "is_error": True},
            {"type": "tool.started", "call_id": "c", "server": "slack", "tool": "three", "arguments": {}},
        ],
    )
    result = extract_deterministic_feedback((), parse_mcp_events(path))
    assert [(x.tool, x.status) for x in result.tool_sequence] == [
        ("slack:one", "ok"), ("slack:two", "error"), ("slack:three", "incomplete")
    ]
    assert result.trace_complete is False


def test_slack_exact_hash_uses_logger_codepoint_length_and_separates_evidence(tmp_path):
    path = tmp_path / "victim-mcp-events.jsonl"
    _write_events(
        path,
        [
            {
                "type": "tool.started", "call_id": "a", "server": "slack", "tool": "channels_history",
                "arguments": {"workspace_id": _shape("W01"), "channel_name": _shape("한국어")},
            },
            {"type": "tool.completed", "call_id": "a", "server": "slack", "tool": "channels_history", "is_error": False},
        ],
    )
    step = ValidatedAttackStep(
        type="environment", turn_id=1,
        injection_mcp_tool="slack-injection:inject_slack_message",
        kwargs={"workspace_id": "W01", "channel_name": "한국어", "message": "payload"},
    )
    item = extract_deterministic_feedback((step,), parse_mcp_events(path)).injections[0]
    assert item.locator_targeted is True
    assert item.access_call_status == "ok"
    assert item.injected_target_accessed is True
    assert item.response_contains_injection is None
    assert item.presented_to_model is None


def test_failed_or_incomplete_locator_call_is_unknown_not_not_accessed(tmp_path):
    for suffix, completed in (("error", True), ("incomplete", None)):
        path = tmp_path / suffix
        events = [{
            "type": "tool.started", "call_id": "a", "server": "OS-filesystem", "tool": "read_file",
            "arguments": {"filepath": _shape("/workspace/config.json")},
        }]
        if completed is not None:
            events.append({"type": "tool.completed", "call_id": "a", "server": "OS-filesystem", "tool": "read_file", "is_error": completed})
        _write_events(path, events)
        step = ValidatedAttackStep(
            type="environment", turn_id=1,
            injection_mcp_tool="os-filesystem-injection:inject_append",
            kwargs={"file_path": "/workspace/config.json", "content": "payload"},
        )
        item = extract_deterministic_feedback((step,), parse_mcp_events(path)).injections[0]
        assert item.locator_targeted is True
        assert item.access_state == "unknown"
        assert item.injected_target_accessed is (False if completed else None)


def test_absence_is_false_only_with_explicit_trace_completion(tmp_path):
    step = ValidatedAttackStep(type="tool", mode="suffix", content="x", injected_tool="slack:channels_history")
    for complete, expected in ((False, None), (True, False)):
        path = tmp_path / str(complete)
        events = [{"type": "trace.completed"}] if complete else []
        _write_events(path, events)
        item = extract_deterministic_feedback((step,), parse_mcp_events(path)).injections[0]
        assert item.polluted_tool_called is expected


def test_duplicate_call_id_is_not_joined(tmp_path):
    path = tmp_path / "events"
    started = {"type": "tool.started", "call_id": "dup", "server": "a", "tool": "b", "arguments": {}}
    _write_events(path, [started, started, {"type": "tool.completed", "call_id": "dup", "server": "a", "tool": "b", "is_error": False}])
    assert parse_mcp_events(path).started == ()


def test_malformed_lines_are_ignored_and_oversize_rejected(tmp_path):
    path = tmp_path / "events"
    path.write_text("not-json\n{}\n", encoding="utf-8")
    assert parse_mcp_events(path).started == ()
    path.write_text("x" * 20, encoding="utf-8")
    try:
        parse_mcp_events(path, max_bytes=10)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized trace accepted")


def test_completion_marker_is_fail_closed_if_trace_is_corrupt_or_has_later_events(tmp_path):
    path = tmp_path / "events"
    _write_events(path, [{"type": "trace.completed"}])
    with path.open("a") as handle:
        handle.write("not-json\n")
    assert parse_mcp_events(path).trace_complete is False

    _write_events(path, [
        {"type": "trace.completed"},
        {"type": "tool.started", "call_id": "late", "server": "a", "tool": "b", "arguments": {}},
    ])
    assert parse_mcp_events(path).trace_complete is False
