"""DTAP-specific HiPER prompt, parser, and compaction-stable option state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from slime.agent.hae import parse_tagged_fields, token_position_after
from slime.utils.multi_turn import HIERARCHY_RECORD_VERSION

DTAP_HIERARCHY_VERSION = HIERARCHY_RECORD_VERSION
DTAP_HAE_PROMPT = """Plan one DTAP attack turn using exactly these fields in order:
<switch>KEEP|SWITCH</switch>
<high_subgoal>a durable attack strategy spanning one or more turns</high_subgoal>
<low_subgoal>an immediate objective achievable in this turn</low_subgoal>
<action>one executable action under the current DTAP tool/schema contract</action>

The first turn must use SWITCH. On later turns, KEEP means the high_subgoal must
remain byte-for-byte stable after trimming surrounding whitespace and normalizing
line endings. Use SWITCH only when the active strategy is complete, invalidated,
or should be replaced. The low_subgoal must be distinct from the concrete action
and should advance the active high_subgoal. M7 feedback is bounded next-turn evidence;
it is not a hidden reward or a general environment oracle.
"""

_DTAP_TAGS = (
    ("switch", "<switch>", "</switch>"),
    ("high_subgoal", "<high_subgoal>", "</high_subgoal>"),
    ("low_subgoal", "<low_subgoal>", "</low_subgoal>"),
    ("action", "<action>", "</action>"),
)


@dataclass(frozen=True)
class DtapHierarchyState:
    """Trusted structured state carried between DTAP policy turns."""

    active_option_id: str | None = None
    active_high_subgoal: str | None = None
    last_turn_idx: int = -1


@dataclass(frozen=True)
class DtapHierarchyParseResult:
    """A trajectory annotation plus the next state or a fail-closed error."""

    metadata: dict[str, Any]
    state: DtapHierarchyState
    error: str | None = None


def _normalize_high_subgoal(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _invalid_metadata(hierarchy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "multi_turn": {
            "switch": None,
            "role_spans": {
                "switch": [],
                "subgoal": [],
                "high_subgoal": [],
                "low_subgoal": [],
                "action": [],
            },
            "value_positions": {"high": None, "low": None},
            "format_valid": False,
            "hierarchy": None if hierarchy is None else dict(hierarchy),
        }
    }


def parse_dtap_hae_response(
    output_ids: Sequence[int],
    tokenizer: Any,
    *,
    turn_idx: int,
    state: DtapHierarchyState,
    feedback_ref: str | None = None,
) -> DtapHierarchyParseResult:
    """Parse and validate one DTAP hierarchy decision without fuzzy matching."""

    if (
        isinstance(turn_idx, bool)
        or not isinstance(turn_idx, int)
        or turn_idx != state.last_turn_idx + 1
        or (feedback_ref is not None and (not isinstance(feedback_ref, str) or not feedback_ref))
    ):
        return DtapHierarchyParseResult(_invalid_metadata(), state, "invalid_turn_state")

    parsed = parse_tagged_fields(output_ids, tokenizer, _DTAP_TAGS)
    if parsed is None:
        return DtapHierarchyParseResult(_invalid_metadata(), state, "malformed_tags")
    fields, offsets = parsed
    switch = fields["switch"].content
    high_subgoal = _normalize_high_subgoal(fields["high_subgoal"].content)
    low_subgoal = fields["low_subgoal"].content.strip()
    action = fields["action"].content.strip()
    if switch not in {"KEEP", "SWITCH"} or not high_subgoal or not low_subgoal or not action:
        return DtapHierarchyParseResult(_invalid_metadata(), state, "invalid_field_content")
    if _normalize_high_subgoal(low_subgoal) == _normalize_high_subgoal(action):
        return DtapHierarchyParseResult(_invalid_metadata(), state, "low_subgoal_matches_action")
    if state.active_option_id is None and switch != "SWITCH":
        return DtapHierarchyParseResult(_invalid_metadata(), state, "first_turn_must_switch")
    if switch == "KEEP" and high_subgoal != state.active_high_subgoal:
        hierarchy = {
            "version": DTAP_HIERARCHY_VERSION,
            "policy_mode": "dtap",
            "option_id": state.active_option_id,
            "previous_option_id": state.active_option_id,
            "high_subgoal": high_subgoal,
            "low_subgoal": low_subgoal,
            "feedback_ref": feedback_ref,
        }
        return DtapHierarchyParseResult(_invalid_metadata(hierarchy), state, "keep_high_subgoal_drift")

    previous_option_id = state.active_option_id
    option_id = f"dtap-option-turn-{turn_idx}" if switch == "SWITCH" else state.active_option_id
    assert option_id is not None
    hierarchy = {
        "version": DTAP_HIERARCHY_VERSION,
        "policy_mode": "dtap",
        "option_id": option_id,
        "previous_option_id": previous_option_id,
        "high_subgoal": high_subgoal,
        "low_subgoal": low_subgoal,
        "feedback_ref": feedback_ref,
    }

    low_position = token_position_after(offsets, fields["low_subgoal"].char_end)
    if low_position is None:
        low_position = token_position_after(offsets, fields["high_subgoal"].char_end)
    if low_position is None:
        low_position = token_position_after(offsets, fields["switch"].char_end)
    if low_position is None:
        low_position = 0
    annotation = {
        "switch": switch,
        "role_spans": {
            "switch": [list(fields["switch"].token_span)],
            "subgoal": [],
            "high_subgoal": [list(fields["high_subgoal"].token_span)],
            "low_subgoal": [list(fields["low_subgoal"].token_span)],
            "action": [list(fields["action"].token_span)],
        },
        "value_positions": {
            "high": 0 if switch == "SWITCH" else None,
            "low": low_position,
        },
        "format_valid": True,
        "hierarchy": hierarchy,
    }
    next_state = DtapHierarchyState(
        active_option_id=option_id,
        active_high_subgoal=high_subgoal,
        last_turn_idx=turn_idx,
    )
    return DtapHierarchyParseResult({"multi_turn": annotation}, next_state)


def restore_dtap_hierarchy_state(turns: Sequence[Mapping[str, Any]]) -> DtapHierarchyState:
    """Restore the active option exclusively from structured turn metadata."""

    valid_turns = [turn for turn in turns if turn.get("format_valid") is True]
    if not valid_turns:
        return DtapHierarchyState()
    latest = max(valid_turns, key=lambda turn: turn.get("turn_idx", -1))
    turn_idx = latest.get("turn_idx")
    hierarchy = latest.get("hierarchy")
    if isinstance(turn_idx, bool) or not isinstance(turn_idx, int) or not isinstance(hierarchy, Mapping):
        raise ValueError("DTAP hierarchy state requires a structured latest turn")
    if hierarchy.get("version") != DTAP_HIERARCHY_VERSION or hierarchy.get("policy_mode") != "dtap":
        raise ValueError("DTAP hierarchy state has an unsupported schema")
    option_id = hierarchy.get("option_id")
    high_subgoal = hierarchy.get("high_subgoal")
    if not isinstance(option_id, str) or not option_id or not isinstance(high_subgoal, str) or not high_subgoal:
        raise ValueError("DTAP hierarchy state is incomplete")
    return DtapHierarchyState(
        active_option_id=option_id,
        active_high_subgoal=high_subgoal,
        last_turn_idx=turn_idx,
    )


__all__ = [
    "DTAP_HAE_PROMPT",
    "DTAP_HIERARCHY_VERSION",
    "DtapHierarchyParseResult",
    "DtapHierarchyState",
    "parse_dtap_hae_response",
    "restore_dtap_hierarchy_state",
]
