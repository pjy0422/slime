"""Validation for logical-turn metadata at the rollout/training boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from slime.utils.types import Sample

MULTI_TURN_METADATA_VERSION = 1
HIERARCHY_RECORD_VERSION = 1

_MULTI_TURN_KEYS = {"version", "context_revision", "turns", "dropped_turns"}
_TURN_KEYS = {
    "turn_idx",
    "response_span",
    "reward",
    "done",
    "truncated",
    "anchor_key",
    "switch",
    "role_spans",
    "value_positions",
    "format_valid",
}
_OPTIONAL_TURN_KEYS = {"hierarchy"}
_HIERARCHY_KEYS = {
    "version",
    "policy_mode",
    "option_id",
    "previous_option_id",
    "high_subgoal",
    "low_subgoal",
    "feedback_ref",
}
_ROLE_SPAN_KEYS = {"switch", "subgoal", "high_subgoal", "low_subgoal", "action"}
_VALUE_POSITION_KEYS = {"high", "low"}
_DROPPED_TURN_KEYS = {"turn_idx", "reason", "generated_tokens"}


def normalize_hierarchy_record(value: Any) -> dict[str, Any] | None:
    """Validate the optional versioned DTAP hierarchy subrecord."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _HIERARCHY_KEYS:
        raise ValueError("multi-turn hierarchy must contain the canonical fields")
    if value["version"] != HIERARCHY_RECORD_VERSION or value["policy_mode"] != "dtap":
        raise ValueError("multi-turn hierarchy has an unsupported schema")
    for field in ("option_id", "high_subgoal", "low_subgoal"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"multi-turn hierarchy {field} must be a non-empty string")
    for field in ("previous_option_id", "feedback_ref"):
        if value[field] is not None and (not isinstance(value[field], str) or not value[field]):
            raise ValueError(f"multi-turn hierarchy {field} must be a non-empty string or null")
    return dict(value)


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _validate_span(value: Any, *, response_length: int, field: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item span")
    start = _require_nonnegative_int(value[0], field=f"{field} start")
    end = _require_nonnegative_int(value[1], field=f"{field} end")
    if start >= end or end > response_length:
        raise ValueError(f"{field} lies outside the sample response")
    return start, end


def _validate_turn(
    turn: Any,
    *,
    sample: Sample,
    sample_position: int,
    seen_owners: set[tuple[int, int]],
) -> int:
    if not isinstance(turn, Mapping):
        raise ValueError(f"multi-turn sample {sample_position} contains a non-mapping turn")
    missing = _TURN_KEYS - set(turn)
    unknown = set(turn) - _TURN_KEYS - _OPTIONAL_TURN_KEYS
    if missing or unknown:
        raise ValueError(
            f"multi-turn sample {sample_position} has invalid turn fields: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    turn_idx = _require_nonnegative_int(turn["turn_idx"], field="multi-turn turn_idx")
    assert sample.rollout_id is not None
    owner = (sample.rollout_id, turn_idx)
    if owner in seen_owners:
        raise ValueError(f"logical turn has multiple owners: rollout_id={owner[0]!r}, turn_idx={owner[1]}")
    seen_owners.add(owner)

    response_start, response_end = _validate_span(
        turn["response_span"],
        response_length=sample.response_length,
        field=f"multi-turn {turn_idx} response_span",
    )
    if sample.loss_mask is None or not all(sample.loss_mask[response_start:response_end]):
        raise ValueError(f"logical turn span is not fully owned: turn_idx={turn_idx}")

    reward = turn["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        raise ValueError(f"multi-turn {turn_idx} reward must be a finite number before training")
    for field in ("done", "truncated", "format_valid"):
        if not isinstance(turn[field], bool):
            raise ValueError(f"multi-turn {turn_idx} {field} must be a boolean")
    for field in ("anchor_key", "switch"):
        if turn[field] is not None and not isinstance(turn[field], str):
            raise ValueError(f"multi-turn {turn_idx} {field} must be a string or null")

    normalize_hierarchy_record(turn.get("hierarchy"))

    role_spans = turn["role_spans"]
    if not isinstance(role_spans, Mapping) or set(role_spans) != _ROLE_SPAN_KEYS:
        raise ValueError(f"multi-turn {turn_idx} role_spans must contain the canonical roles")
    for role, spans in role_spans.items():
        if not isinstance(spans, list):
            raise ValueError(f"multi-turn {turn_idx} role_spans.{role} must be a list")
        for span in spans:
            span_start, span_end = _validate_span(
                span,
                response_length=sample.response_length,
                field=f"multi-turn {turn_idx} role_spans.{role}",
            )
            if span_start < response_start or span_end > response_end:
                raise ValueError(f"multi-turn {turn_idx} role_spans.{role} lies outside its owned response")

    value_positions = turn["value_positions"]
    if not isinstance(value_positions, Mapping) or set(value_positions) != _VALUE_POSITION_KEYS:
        raise ValueError(f"multi-turn {turn_idx} value_positions must contain high and low")
    for head, position in value_positions.items():
        if position is None:
            continue
        position = _require_nonnegative_int(position, field=f"multi-turn {turn_idx} value_positions.{head}")
        if not response_start <= position < response_end:
            raise ValueError(f"multi-turn {turn_idx} value_positions.{head} lies outside its owned response")
    return turn_idx


def validate_multi_turn_training_samples(samples: Sequence[Sample]) -> None:
    """Fail closed on malformed logical-turn metadata before DP splitting.

    Legacy samples without the reserved ``multi_turn`` namespace remain valid.
    When the namespace is present, the canonical v1 schema, response ownership,
    and exactly-one-owner identity are validated across the complete batch.
    """

    seen_owners: set[tuple[int, int]] = set()
    for sample_position, sample in enumerate(samples):
        train_metadata = sample.train_metadata
        if train_metadata is None:
            continue
        if not isinstance(train_metadata, Mapping):
            raise ValueError(f"train_metadata for sample {sample_position} must be a mapping")
        if "multi_turn" not in train_metadata:
            continue
        if sample.rollout_id is None:
            raise ValueError(f"multi-turn sample {sample_position} requires rollout_id")

        multi_turn = train_metadata["multi_turn"]
        if not isinstance(multi_turn, Mapping):
            raise ValueError(f"train_metadata.multi_turn for sample {sample_position} must be a mapping")
        missing = {"version", "context_revision", "turns"} - set(multi_turn)
        unknown = set(multi_turn) - _MULTI_TURN_KEYS
        if missing or unknown:
            raise ValueError(
                f"multi-turn sample {sample_position} has invalid namespace fields: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        version = multi_turn["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != MULTI_TURN_METADATA_VERSION:
            raise ValueError(f"unsupported multi-turn metadata version: {multi_turn['version']!r}")
        _require_nonnegative_int(
            multi_turn["context_revision"],
            field=f"multi-turn sample {sample_position} context_revision",
        )

        turns = multi_turn["turns"]
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"multi-turn sample {sample_position} turns must be a non-empty list")
        local_turns = [
            _validate_turn(
                turn,
                sample=sample,
                sample_position=sample_position,
                seen_owners=seen_owners,
            )
            for turn in turns
        ]
        if local_turns != sorted(local_turns) or len(local_turns) != len(set(local_turns)):
            raise ValueError(f"multi-turn sample {sample_position} turns must be strictly increasing")

        dropped_turns = multi_turn.get("dropped_turns", [])
        if not isinstance(dropped_turns, list):
            raise ValueError(f"multi-turn sample {sample_position} dropped_turns must be a list")
        for dropped in dropped_turns:
            if not isinstance(dropped, Mapping) or set(dropped) != _DROPPED_TURN_KEYS:
                raise ValueError(f"multi-turn sample {sample_position} contains an invalid dropped turn")
            _require_nonnegative_int(dropped["turn_idx"], field="dropped multi-turn turn_idx")
            _require_nonnegative_int(dropped["generated_tokens"], field="dropped multi-turn generated_tokens")
            if not isinstance(dropped["reason"], str) or not dropped["reason"]:
                raise ValueError("dropped multi-turn reason must be a non-empty string")


def build_training_metadata_fields(samples: Sequence[Sample]) -> dict[str, list[Any]]:
    """Validate and serialize sample identity/metadata fields for training."""

    validate_multi_turn_training_samples(samples)
    fields: dict[str, list[Any]] = {"group_indices": [sample.group_index for sample in samples]}
    if any(sample.train_metadata is not None for sample in samples):
        fields["metadata"] = [dict(sample.train_metadata or {}) for sample in samples]
    return fields


__all__ = [
    "HIERARCHY_RECORD_VERSION",
    "MULTI_TURN_METADATA_VERSION",
    "build_training_metadata_fields",
    "normalize_hierarchy_record",
    "validate_multi_turn_training_samples",
]
