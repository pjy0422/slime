"""Decomposed-credit GRPO over complete logical-turn rollout groups."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from slime.utils.advantages.mt_ppo import LogicalTurn, collect_logical_turns


def _normalized(values: Sequence[float]) -> list[float]:
    """Population z-scores, with zero credit for degenerate groups."""

    if not values:
        return []
    tensor = torch.tensor(values, dtype=torch.float64)
    centered = tensor - tensor.mean()
    variance = centered.square().mean()
    if variance == 0:
        return [0.0] * len(values)
    return (centered / variance.sqrt()).tolist()


def compute_dcgrpo_turn_credits(
    turns: Sequence[LogicalTurn],
    *,
    mode: str,
    gamma: float = 1.0,
    alpha: float = 1.0,
) -> dict[tuple[int, int], float]:
    """Compute DW or SW credit for each logical turn.

    DW normalizes discounted return-to-go within ``(group_index, turn_idx)``.
    SW separately normalizes immediate reward and successor return-to-go in
    the same comparison cohort, then computes ``I + alpha * F``.
    """

    if mode not in {"dw", "sw"}:
        raise ValueError(f"unsupported DC-GRPO mode: {mode!r}")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("DC-GRPO gamma must lie in [0, 1]")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("DC-GRPO alpha must be finite and nonnegative")
    if not turns:
        raise ValueError("DC-GRPO requires at least one logical turn")
    if any(turn.group_index is None for turn in turns):
        raise ValueError("DC-GRPO requires group_index on every sample")

    turns_by_rollout: dict[int, list[LogicalTurn]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for turn in turns:
        key = (turn.rollout_id, turn.turn_idx)
        if key in seen:
            raise ValueError(f"DC-GRPO logical turn has multiple owners: {key}")
        if not math.isfinite(turn.reward):
            raise ValueError(f"DC-GRPO logical turn {key} reward must be finite")
        seen.add(key)
        turns_by_rollout[turn.rollout_id].append(turn)

    returns: dict[tuple[int, int], float] = {}
    future_returns: dict[tuple[int, int], float] = {}
    for rollout_id, rollout_turns in turns_by_rollout.items():
        rollout_turns.sort(key=lambda turn: turn.turn_idx)
        group_indices = {turn.group_index for turn in rollout_turns}
        if len(group_indices) != 1:
            raise ValueError(f"DC-GRPO rollout {rollout_id} spans multiple comparison groups")
        terminal_positions = [index for index, turn in enumerate(rollout_turns) if turn.done]
        if terminal_positions and terminal_positions != [len(rollout_turns) - 1]:
            raise ValueError(f"DC-GRPO rollout {rollout_id} has a non-final terminal turn")

        next_return = 0.0
        for turn in reversed(rollout_turns):
            key = (rollout_id, turn.turn_idx)
            future_returns[key] = 0.0 if turn.done else next_return
            turn_return = turn.reward + gamma * future_returns[key]
            returns[key] = turn_return
            next_return = turn_return

    comparison_groups: dict[tuple[int, int], list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        assert turn.group_index is not None
        comparison_groups[(turn.group_index, turn.turn_idx)].append(turn)

    credits: dict[tuple[int, int], float] = {}
    for comparison_turns in comparison_groups.values():
        keys = [(turn.rollout_id, turn.turn_idx) for turn in comparison_turns]
        if mode == "dw":
            normalized_returns = _normalized([returns[key] for key in keys])
            credits.update(zip(keys, normalized_returns, strict=True))
            continue

        immediate = _normalized([turn.reward for turn in comparison_turns])
        future = _normalized([future_returns[key] for key in keys])
        credits.update(
            (key, immediate_credit + alpha * future_credit)
            for key, immediate_credit, future_credit in zip(keys, immediate, future, strict=True)
        )
    return credits


def precompute_dcgrpo_train_data(
    train_data: Mapping[str, object],
    *,
    mode: str,
    gamma: float = 1.0,
    alpha: float = 1.0,
) -> list[list[float]]:
    """Compute sample-aligned turn credits before data-parallel splitting."""

    required = ("metadata", "rollout_ids", "group_indices", "response_lengths", "loss_masks")
    missing = [key for key in required if key not in train_data]
    if missing:
        raise ValueError(f"DC-GRPO training data is missing required fields: {missing}")

    metadata = train_data["metadata"]
    rollout_ids = train_data["rollout_ids"]
    group_indices = train_data["group_indices"]
    response_lengths = train_data["response_lengths"]
    loss_masks = train_data["loss_masks"]
    if not all(
        isinstance(field, Sequence) for field in (metadata, rollout_ids, group_indices, response_lengths, loss_masks)
    ):
        raise ValueError("DC-GRPO training fields must be sequences")

    tensor_masks = [torch.as_tensor(mask) for mask in loss_masks]
    turns = collect_logical_turns(
        metadata,
        rollout_ids,
        response_lengths,
        tensor_masks,
        group_indices=group_indices,
    )
    credits = compute_dcgrpo_turn_credits(turns, mode=mode, gamma=gamma, alpha=alpha)

    turns_by_sample: dict[int, list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        turns_by_sample[turn.sample_index].append(turn)
    packed: list[list[float]] = []
    for sample_index in range(len(response_lengths)):
        sample_turns = turns_by_sample[sample_index]
        packed.append([credits[(turn.rollout_id, turn.turn_idx)] for turn in sample_turns])
    return packed


def unpack_dcgrpo_turn_credits(
    turns: Sequence[LogicalTurn],
    packed_credits: Sequence[Sequence[float]],
) -> dict[tuple[int, int], float]:
    """Validate and key sample-aligned credits after DP partitioning."""

    turns_by_sample: dict[int, list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        turns_by_sample[turn.sample_index].append(turn)
    if len(packed_credits) != len(turns_by_sample):
        raise ValueError("DC-GRPO turn credits must contain one entry per sample")

    credits: dict[tuple[int, int], float] = {}
    for sample_index in range(len(packed_credits)):
        sample_turns = turns_by_sample[sample_index]
        sample_credits = packed_credits[sample_index]
        if len(sample_credits) != len(sample_turns):
            raise ValueError(f"DC-GRPO sample {sample_index} turn credits do not match its logical turns")
        for turn, credit in zip(sample_turns, sample_credits, strict=True):
            if isinstance(credit, bool) or not isinstance(credit, (int, float)) or not math.isfinite(float(credit)):
                raise ValueError("DC-GRPO turn credit must be finite and numeric")
            credits[(turn.rollout_id, turn.turn_idx)] = float(credit)
    return credits


__all__ = [
    "compute_dcgrpo_turn_credits",
    "precompute_dcgrpo_train_data",
    "unpack_dcgrpo_turn_credits",
]
