"""Pure logical-turn math and layout helpers for multi-turn PPO."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from slime.utils.multi_turn import MULTI_TURN_METADATA_VERSION


@dataclass(frozen=True)
class LogicalTurn:
    """One train-owned turn expressed in full response coordinates."""

    rollout_id: int
    group_index: int | None
    sample_index: int
    turn_idx: int
    response_start: int
    response_end: int
    value_position: int
    reward: float
    done: bool


def compute_turn_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over a single rollout's logical turns.

    The final observed turn has no separately sampled successor state, so its
    bootstrap value is zero. A terminal turn also cuts both the one-step
    bootstrap and the recursive advantage, even if more values are supplied.
    """

    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("turn rewards, values, and dones must be one-dimensional")
    if not (rewards.numel() == values.numel() == dones.numel()):
        raise ValueError("turn rewards, values, and dones must have equal lengths")
    if rewards.numel() == 0:
        raise ValueError("turn GAE requires at least one logical turn")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= lambd <= 1.0:
        raise ValueError("gamma and lambda must lie in [0, 1]")

    dones = dones.to(device=values.device, dtype=torch.bool)
    rewards = rewards.to(device=values.device, dtype=values.dtype)
    advantages = torch.zeros_like(values)
    next_advantage = values.new_zeros(())

    for turn_idx in range(values.numel() - 1, -1, -1):
        nonterminal = (~dones[turn_idx]).to(values.dtype)
        next_value = values[turn_idx + 1] if turn_idx + 1 < values.numel() else values.new_zeros(())
        delta = rewards[turn_idx] + gamma * nonterminal * next_value - values[turn_idx]
        next_advantage = delta + gamma * lambd * nonterminal * next_advantage
        advantages[turn_idx] = next_advantage

    return advantages, advantages + values


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
        raise ValueError("multi-turn PPO fields must contain one entry per sample")
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
            raise ValueError(f"multi-turn PPO sample {sample_index} metadata must be a mapping")
        namespace = sample_metadata.get("multi_turn")
        if not isinstance(namespace, Mapping):
            raise ValueError(f"multi-turn PPO sample {sample_index} is missing canonical multi_turn metadata")
        if namespace.get("version") != MULTI_TURN_METADATA_VERSION:
            raise ValueError(f"multi-turn PPO sample {sample_index} has an unsupported metadata version")
        raw_turns = namespace.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"multi-turn PPO sample {sample_index} must own at least one logical turn")
        if loss_mask.numel() != response_length:
            raise ValueError(f"multi-turn PPO sample {sample_index} loss mask length is inconsistent")

        for raw_turn in raw_turns:
            if not isinstance(raw_turn, Mapping):
                raise ValueError(f"multi-turn PPO sample {sample_index} contains a malformed turn")
            turn_idx = raw_turn.get("turn_idx")
            span = raw_turn.get("response_span")
            reward = raw_turn.get("reward")
            done = raw_turn.get("done")
            if isinstance(turn_idx, bool) or not isinstance(turn_idx, int) or turn_idx < 0:
                raise ValueError("multi-turn PPO turn_idx must be a nonnegative integer")
            if (
                not isinstance(span, Sequence)
                or isinstance(span, (str, bytes))
                or len(span) != 2
                or any(isinstance(position, bool) or not isinstance(position, int) for position in span)
            ):
                raise ValueError(f"multi-turn PPO turn {turn_idx} has an invalid response span")
            response_start, response_end = span
            if response_start < 0 or response_start >= response_end or response_end > response_length:
                raise ValueError(f"multi-turn PPO turn {turn_idx} response span is outside the sample")
            if not bool(torch.all(loss_mask[response_start:response_end]).item()):
                raise ValueError(f"multi-turn PPO turn {turn_idx} response span is not fully train-owned")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                raise ValueError(f"multi-turn PPO turn {turn_idx} reward must be finite and numeric")
            if not isinstance(done, bool):
                raise ValueError(f"multi-turn PPO turn {turn_idx} done must be boolean")

            key = (int(rollout_id), turn_idx)
            if key in seen:
                raise ValueError(f"multi-turn PPO logical turn has multiple owners: {key}")
            seen.add(key)
            turns.append(
                LogicalTurn(
                    rollout_id=int(rollout_id),
                    group_index=group_index,
                    sample_index=sample_index,
                    turn_idx=turn_idx,
                    response_start=response_start,
                    response_end=response_end,
                    # A response-aligned value at offset zero is V(s) from the
                    # logit immediately before the first generated token.
                    value_position=response_start,
                    reward=float(reward),
                    done=done,
                )
            )

    by_rollout: dict[int, list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        by_rollout[turn.rollout_id].append(turn)
    for rollout_id, rollout_turns in by_rollout.items():
        rollout_turns.sort(key=lambda turn: turn.turn_idx)
        terminal_positions = [index for index, turn in enumerate(rollout_turns) if turn.done]
        if terminal_positions and terminal_positions != [len(rollout_turns) - 1]:
            raise ValueError(f"multi-turn PPO rollout {rollout_id} has a non-final terminal turn")

    return turns


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
