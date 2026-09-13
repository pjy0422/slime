"""Group-in-Group Policy Optimization over complete logical trajectories."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from slime.utils.advantages.multi_turn import LogicalTurn, collect_logical_turns, pack_turn_credits


def relative_group_advantage(
    values: Sequence[float],
    *,
    mode: str,
    eps: float = 1e-6,
) -> list[float]:
    """Center a comparison group, optionally using sample-standard-deviation scaling."""

    if mode not in {"mean", "mean_std"}:
        raise ValueError(f"unknown GiGPO normalization: {mode!r}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("GiGPO normalization epsilon must be finite and positive")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("GiGPO group values must be numeric")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("GiGPO group values must be finite")
    if not values:
        return []

    tensor = torch.tensor(values, dtype=torch.float32)
    centered = tensor - tensor.mean()
    if mode == "mean":
        return centered.tolist()
    if tensor.numel() <= 1:
        return [0.0] * tensor.numel()
    std = tensor.std(unbiased=True)
    if not torch.isfinite(std) or std <= eps:
        return [0.0] * tensor.numel()
    return (centered / (std + eps)).tolist()


def discounted_turn_returns(
    turns: Sequence[LogicalTurn],
    *,
    gamma: float = 1.0,
) -> dict[tuple[int, int], float]:
    """Compute turn-time discounted suffix returns for complete trajectories."""

    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("GiGPO gamma must lie in [0, 1]")
    if not turns:
        raise ValueError("GiGPO requires at least one logical turn")

    turns_by_rollout: dict[int, list[LogicalTurn]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for turn in turns:
        key = (turn.rollout_id, turn.turn_idx)
        if key in seen:
            raise ValueError(f"GiGPO logical turn has multiple owners: {key}")
        if not math.isfinite(turn.reward):
            raise ValueError(f"GiGPO logical turn {key} reward must be finite")
        if turn.truncated:
            raise ValueError(f"GiGPO rollout {turn.rollout_id} contains a truncated turn")
        seen.add(key)
        turns_by_rollout[turn.rollout_id].append(turn)

    returns: dict[tuple[int, int], float] = {}
    for rollout_id, rollout_turns in turns_by_rollout.items():
        rollout_turns.sort(key=lambda turn: turn.turn_idx)
        if [turn.turn_idx for turn in rollout_turns] != list(range(len(rollout_turns))):
            raise ValueError(f"GiGPO rollout {rollout_id} does not contain a complete turn sequence")
        terminal_positions = [index for index, turn in enumerate(rollout_turns) if turn.done]
        if terminal_positions and terminal_positions != [len(rollout_turns) - 1]:
            raise ValueError(f"GiGPO rollout {rollout_id} has a non-final terminal turn")

        next_return = 0.0
        for turn in reversed(rollout_turns):
            key = (rollout_id, turn.turn_idx)
            next_return = turn.reward + gamma * (0.0 if turn.done else next_return)
            returns[key] = next_return
    return returns


def compute_gigpo_turn_credits(
    turns: Sequence[LogicalTurn],
    *,
    gamma: float = 1.0,
    step_advantage_weight: float = 1.0,
    normalization: str = "mean_std",
) -> dict[tuple[int, int], float]:
    """Compute GiGPO episode-plus-anchor relative credit for every turn."""

    if not math.isfinite(step_advantage_weight) or step_advantage_weight < 0.0:
        raise ValueError("GiGPO step advantage weight must be finite and nonnegative")
    if normalization not in {"mean", "mean_std"}:
        raise ValueError(f"unknown GiGPO normalization: {normalization!r}")
    if any(
        isinstance(turn.group_index, bool) or not isinstance(turn.group_index, int) or turn.group_index < 0
        for turn in turns
    ):
        raise ValueError("GiGPO requires group_index on every sample")
    if any(not isinstance(turn.anchor_key, str) or not turn.anchor_key for turn in turns):
        raise ValueError("GiGPO requires a non-empty anchor_key on every turn")

    returns = discounted_turn_returns(turns, gamma=gamma)
    turns_by_rollout: dict[int, list[LogicalTurn]] = defaultdict(list)
    for turn in turns:
        turns_by_rollout[turn.rollout_id].append(turn)

    episode_groups: dict[int, list[int]] = defaultdict(list)
    episode_returns: dict[int, float] = {}
    for rollout_id, rollout_turns in turns_by_rollout.items():
        group_indices = {turn.group_index for turn in rollout_turns}
        if len(group_indices) != 1:
            raise ValueError(f"GiGPO rollout {rollout_id} spans multiple episode groups")
        group_index = next(iter(group_indices))
        assert group_index is not None
        episode_groups[group_index].append(rollout_id)
        episode_returns[rollout_id] = sum(turn.reward for turn in rollout_turns)

    episode_advantages: dict[int, float] = {}
    for rollout_ids in episode_groups.values():
        advantages = relative_group_advantage(
            [episode_returns[rollout_id] for rollout_id in rollout_ids],
            mode=normalization,
        )
        episode_advantages.update(zip(rollout_ids, advantages, strict=True))

    anchor_groups: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for turn in turns:
        assert turn.group_index is not None and turn.anchor_key is not None
        anchor_groups[(turn.group_index, turn.anchor_key)].append((turn.rollout_id, turn.turn_idx))

    step_advantages: dict[tuple[int, int], float] = {}
    for keys in anchor_groups.values():
        advantages = relative_group_advantage([returns[key] for key in keys], mode=normalization)
        step_advantages.update(zip(keys, advantages, strict=True))

    return {key: episode_advantages[key[0]] + step_advantage_weight * step_advantages[key] for key in returns}


def precompute_gigpo_train_data(
    train_data: Mapping[str, object],
    *,
    gamma: float = 1.0,
    step_advantage_weight: float = 1.0,
    normalization: str = "mean_std",
) -> list[list[float]]:
    """Compute sample-aligned GiGPO credits before data-parallel splitting."""

    required = ("metadata", "rollout_ids", "group_indices", "response_lengths", "loss_masks")
    missing = [key for key in required if key not in train_data]
    if missing:
        raise ValueError(f"GiGPO training data is missing required fields: {missing}")

    metadata = train_data["metadata"]
    rollout_ids = train_data["rollout_ids"]
    group_indices = train_data["group_indices"]
    response_lengths = train_data["response_lengths"]
    loss_masks = train_data["loss_masks"]
    if not all(
        isinstance(field, Sequence) for field in (metadata, rollout_ids, group_indices, response_lengths, loss_masks)
    ):
        raise ValueError("GiGPO training fields must be sequences")

    truncated_flags = train_data.get("truncated")
    if truncated_flags is not None:
        if not isinstance(truncated_flags, Sequence) or len(truncated_flags) != len(response_lengths):
            raise ValueError("GiGPO truncated flags must contain one entry per sample")
        for sample_index, truncated in enumerate(truncated_flags):
            if not isinstance(truncated, (bool, int)) or truncated not in (0, 1):
                raise ValueError("GiGPO truncated flags must be boolean or binary")
            if bool(truncated):
                raise ValueError(f"GiGPO rollout {rollout_ids[sample_index]} was collector-truncated")

    for sample_index, sample_metadata in enumerate(metadata):
        if not isinstance(sample_metadata, Mapping):
            continue
        namespace = sample_metadata.get("multi_turn")
        if isinstance(namespace, Mapping) and namespace.get("dropped_turns"):
            raise ValueError(f"GiGPO sample {sample_index} has dropped turns and an incomplete credit history")

    turns = collect_logical_turns(
        metadata,
        rollout_ids,
        response_lengths,
        [torch.as_tensor(mask) for mask in loss_masks],
        group_indices=group_indices,
    )
    credits = compute_gigpo_turn_credits(
        turns,
        gamma=gamma,
        step_advantage_weight=step_advantage_weight,
        normalization=normalization,
    )
    return pack_turn_credits(turns, credits, len(response_lengths))


__all__ = [
    "compute_gigpo_turn_credits",
    "discounted_turn_returns",
    "precompute_gigpo_train_data",
    "relative_group_advantage",
]
