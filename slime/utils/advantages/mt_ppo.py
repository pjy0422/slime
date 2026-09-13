"""Pure logical-turn GAE for multi-turn PPO."""

from __future__ import annotations

import torch


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
