"""Shared logical-turn metadata, credit packing, and token projection."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from slime.utils.multi_turn import MULTI_TURN_METADATA_VERSION, normalize_hierarchy_record


@dataclass(frozen=True)
class LogicalTurn:
    """One train-owned logical turn expressed in full response coordinates."""

    rollout_id: int
    group_index: int | None
    sample_index: int
    turn_idx: int
    response_start: int
    response_end: int
    value_position: int
    reward: float
    done: bool
    anchor_key: str | None = None
    truncated: bool = False
    switch: str | None = None
    role_spans: Mapping[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    high_value_position: int | None = None
    low_value_position: int | None = None
    format_valid: bool = True
    hierarchy: Mapping[str, object] | None = None


def collect_logical_turns(
    metadata: Sequence[object],
    rollout_ids: Sequence[int],
    response_lengths: Sequence[int],
    loss_masks: Sequence[torch.Tensor],
    group_indices: Sequence[int | None] | None = None,
) -> list[LogicalTurn]:
    """Read canonical v1 metadata and fail closed on incomplete layouts."""

    sample_count = len(response_lengths)
    if not (len(metadata) == len(rollout_ids) == len(loss_masks) == sample_count):
        raise ValueError("multi-turn fields must contain one entry per sample")
    if group_indices is not None and len(group_indices) != sample_count:
        raise ValueError("multi-turn group indices must contain one entry per sample")

    turns: list[LogicalTurn] = []
    seen: set[tuple[int, int]] = set()
    for sample_index, (sample_metadata, rollout_id, response_length, loss_mask) in enumerate(
        zip(metadata, rollout_ids, response_lengths, loss_masks, strict=True)
    ):
        group_index = None if group_indices is None else group_indices[sample_index]
        if group_index is not None and (
            isinstance(group_index, bool) or not isinstance(group_index, int) or group_index < 0
        ):
            raise ValueError(f"multi-turn sample {sample_index} group_index must be a nonnegative integer")
        if not isinstance(sample_metadata, Mapping):
            raise ValueError(f"multi-turn sample {sample_index} metadata must be a mapping")
        namespace = sample_metadata.get("multi_turn")
        if not isinstance(namespace, Mapping):
            raise ValueError(f"multi-turn sample {sample_index} is missing canonical multi_turn metadata")
        if namespace.get("version") != MULTI_TURN_METADATA_VERSION:
            raise ValueError(f"multi-turn sample {sample_index} has an unsupported metadata version")
        raw_turns = namespace.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"multi-turn sample {sample_index} must own at least one logical turn")
        if loss_mask.numel() != response_length:
            raise ValueError(f"multi-turn sample {sample_index} loss mask length is inconsistent")

        for raw_turn in raw_turns:
            if not isinstance(raw_turn, Mapping):
                raise ValueError(f"multi-turn sample {sample_index} contains a malformed turn")
            turn_idx = raw_turn.get("turn_idx")
            span = raw_turn.get("response_span")
            reward = raw_turn.get("reward")
            done = raw_turn.get("done")
            anchor_key = raw_turn.get("anchor_key")
            truncated = raw_turn.get("truncated")
            switch = raw_turn.get("switch")
            format_valid = raw_turn.get("format_valid", True)
            hierarchy = raw_turn.get("hierarchy")
            if isinstance(turn_idx, bool) or not isinstance(turn_idx, int) or turn_idx < 0:
                raise ValueError("multi-turn turn_idx must be a nonnegative integer")
            if (
                not isinstance(span, Sequence)
                or isinstance(span, (str, bytes))
                or len(span) != 2
                or any(isinstance(position, bool) or not isinstance(position, int) for position in span)
            ):
                raise ValueError(f"multi-turn turn {turn_idx} has an invalid response span")
            response_start, response_end = span
            if response_start < 0 or response_start >= response_end or response_end > response_length:
                raise ValueError(f"multi-turn turn {turn_idx} response span is outside the sample")
            if not bool(torch.all(loss_mask[response_start:response_end]).item()):
                raise ValueError(f"multi-turn turn {turn_idx} response span is not fully train-owned")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                raise ValueError(f"multi-turn turn {turn_idx} reward must be finite and numeric")
            if not isinstance(done, bool):
                raise ValueError(f"multi-turn turn {turn_idx} done must be boolean")
            if anchor_key is not None and not isinstance(anchor_key, str):
                raise ValueError(f"multi-turn turn {turn_idx} anchor_key must be a string or null")
            if not isinstance(truncated, bool):
                raise ValueError(f"multi-turn turn {turn_idx} truncated must be boolean")
            if switch is not None and not isinstance(switch, str):
                raise ValueError(f"multi-turn turn {turn_idx} switch must be a string or null")
            if not isinstance(format_valid, bool):
                raise ValueError(f"multi-turn turn {turn_idx} format_valid must be boolean")
            hierarchy = normalize_hierarchy_record(hierarchy)

            raw_role_spans = raw_turn.get("role_spans", {})
            if not isinstance(raw_role_spans, Mapping):
                raise ValueError(f"multi-turn turn {turn_idx} role_spans must be a mapping")
            role_spans: dict[str, tuple[tuple[int, int], ...]] = {}
            for role, spans in raw_role_spans.items():
                if not isinstance(role, str) or not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
                    raise ValueError(f"multi-turn turn {turn_idx} has invalid role spans")
                normalized_spans = []
                for role_span in spans:
                    if (
                        not isinstance(role_span, Sequence)
                        or isinstance(role_span, (str, bytes))
                        or len(role_span) != 2
                        or any(isinstance(position, bool) or not isinstance(position, int) for position in role_span)
                    ):
                        raise ValueError(f"multi-turn turn {turn_idx} role {role} has an invalid span")
                    span_start, span_end = role_span
                    if span_start < response_start or span_start >= span_end or span_end > response_end:
                        raise ValueError(f"multi-turn turn {turn_idx} role {role} lies outside its owned response")
                    normalized_spans.append((span_start, span_end))
                role_spans[role] = tuple(normalized_spans)

            raw_value_positions = raw_turn.get("value_positions", {})
            if not isinstance(raw_value_positions, Mapping):
                raise ValueError(f"multi-turn turn {turn_idx} value_positions must be a mapping")
            value_positions: dict[str, int | None] = {}
            for head in ("high", "low"):
                position = raw_value_positions.get(head)
                if position is not None and (
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or not response_start <= position < response_end
                ):
                    raise ValueError(f"multi-turn turn {turn_idx} {head} value position is invalid")
                value_positions[head] = position

            key = (int(rollout_id), turn_idx)
            if key in seen:
                raise ValueError(f"multi-turn logical turn has multiple owners: {key}")
            seen.add(key)
            turns.append(
                LogicalTurn(
                    rollout_id=int(rollout_id),
                    group_index=group_index,
                    sample_index=sample_index,
                    turn_idx=turn_idx,
                    response_start=response_start,
                    response_end=response_end,
                    value_position=response_start,
                    reward=float(reward),
                    done=done,
                    anchor_key=anchor_key,
                    truncated=truncated,
                    switch=switch,
                    role_spans=role_spans,
                    high_value_position=value_positions["high"],
                    low_value_position=value_positions["low"],
                    format_valid=format_valid,
                    hierarchy=hierarchy,
                )
            )

    by_rollout: dict[int, list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        by_rollout[turn.rollout_id].append(turn)
    for rollout_id, rollout_turns in by_rollout.items():
        rollout_turns.sort(key=lambda turn: turn.turn_idx)
        terminal_positions = [index for index, turn in enumerate(rollout_turns) if turn.done]
        if terminal_positions and terminal_positions != [len(rollout_turns) - 1]:
            raise ValueError(f"multi-turn rollout {rollout_id} has a non-final terminal turn")

    return turns


def pack_turn_credits(
    turns: Sequence[LogicalTurn],
    credits: Mapping[tuple[int, int], float],
    sample_count: int,
) -> list[list[float]]:
    """Pack identity-keyed scalar credits in each sample's canonical turn order."""

    turns_by_sample: dict[int, list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        turns_by_sample[turn.sample_index].append(turn)
    packed: list[list[float]] = []
    for sample_index in range(sample_count):
        sample_turns = turns_by_sample[sample_index]
        sample_credits: list[float] = []
        for turn in sample_turns:
            key = (turn.rollout_id, turn.turn_idx)
            if key not in credits:
                raise ValueError(f"missing logical-turn credit for {key}")
            credit = credits[key]
            if isinstance(credit, bool) or not isinstance(credit, (int, float)) or not math.isfinite(float(credit)):
                raise ValueError("logical-turn credit must be finite and numeric")
            sample_credits.append(float(credit))
        packed.append(sample_credits)
    return packed


def unpack_turn_credits(
    turns: Sequence[LogicalTurn],
    packed_credits: Sequence[Sequence[float]],
) -> dict[tuple[int, int], float]:
    """Validate and key sample-aligned scalar credits after DP partitioning."""

    turns_by_sample: dict[int, list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        turns_by_sample[turn.sample_index].append(turn)
    if len(packed_credits) != len(turns_by_sample):
        raise ValueError("turn credits must contain one entry per sample")

    credits: dict[tuple[int, int], float] = {}
    for sample_index, sample_credits in enumerate(packed_credits):
        sample_turns = turns_by_sample[sample_index]
        if len(sample_credits) != len(sample_turns):
            raise ValueError(f"sample {sample_index} turn credits do not match its logical turns")
        for turn, credit in zip(sample_turns, sample_credits, strict=True):
            if isinstance(credit, bool) or not isinstance(credit, (int, float)) or not math.isfinite(float(credit)):
                raise ValueError("logical-turn credit must be finite and numeric")
            credits[(turn.rollout_id, turn.turn_idx)] = float(credit)
    return credits


def project_turn_values(
    response_lengths: Sequence[int],
    turns: Sequence[LogicalTurn],
    values: Mapping[tuple[int, int], torch.Tensor],
    *,
    sparse: bool,
    reference_tensors: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Project turn scalars to full responses and return matching masks."""

    if len(response_lengths) != len(reference_tensors):
        raise ValueError("response lengths and reference tensors must align")
    projected = [
        reference.new_zeros(length) for reference, length in zip(reference_tensors, response_lengths, strict=True)
    ]
    masks = [
        reference.new_zeros(length) for reference, length in zip(reference_tensors, response_lengths, strict=True)
    ]
    for turn in turns:
        key = (turn.rollout_id, turn.turn_idx)
        if key not in values:
            raise ValueError(f"missing logical-turn value for {key}")
        if sparse:
            projected[turn.sample_index][turn.value_position] = values[key]
            masks[turn.sample_index][turn.value_position] = 1
        else:
            projected[turn.sample_index][turn.response_start : turn.response_end] = values[key]
            masks[turn.sample_index][turn.response_start : turn.response_end] = 1
    return projected, masks


def project_role_values(
    response_lengths: Sequence[int],
    turns: Sequence[LogicalTurn],
    values: Mapping[tuple[int, int], torch.Tensor],
    *,
    role: str,
    reference_tensors: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Project logical scalars onto every owned span for one semantic role."""

    if len(response_lengths) != len(reference_tensors):
        raise ValueError("response lengths and reference tensors must align")
    projected = [
        reference.new_zeros(length) for reference, length in zip(reference_tensors, response_lengths, strict=True)
    ]
    masks = [
        reference.new_zeros(length) for reference, length in zip(reference_tensors, response_lengths, strict=True)
    ]
    for turn in turns:
        key = (turn.rollout_id, turn.turn_idx)
        if key not in values:
            raise ValueError(f"missing logical-turn value for {key}")
        spans = turn.role_spans.get(role, ())
        if not spans:
            raise ValueError(f"logical turn {key} has no {role} span")
        for span_start, span_end in spans:
            projected[turn.sample_index][span_start:span_end] = values[key]
            masks[turn.sample_index][span_start:span_end] = 1
    return projected, masks


def project_head_values(
    response_lengths: Sequence[int],
    turns: Sequence[LogicalTurn],
    values: Mapping[tuple[int, int], torch.Tensor],
    *,
    head: str,
    reference_tensors: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Project logical critic targets to sparse low/high value positions."""

    if head not in {"low", "high"}:
        raise ValueError(f"unknown critic head: {head!r}")
    if len(response_lengths) != len(reference_tensors):
        raise ValueError("response lengths and reference tensors must align")
    projected = [
        reference.new_zeros(length) for reference, length in zip(reference_tensors, response_lengths, strict=True)
    ]
    masks = [
        reference.new_zeros(length) for reference, length in zip(reference_tensors, response_lengths, strict=True)
    ]
    for turn in turns:
        key = (turn.rollout_id, turn.turn_idx)
        if key not in values:
            raise ValueError(f"missing logical-turn value for {key}")
        position = turn.low_value_position if head == "low" else turn.high_value_position
        if position is None:
            raise ValueError(f"logical turn {key} has no {head} value position")
        projected[turn.sample_index][position] = values[key]
        masks[turn.sample_index][position] = 1
    return projected, masks


__all__ = [
    "LogicalTurn",
    "collect_logical_turns",
    "pack_turn_credits",
    "project_head_values",
    "project_role_values",
    "project_turn_values",
    "unpack_turn_credits",
]
