"""Reference two-head hierarchical advantage estimation.

The recurrence follows the low/high path in ``JonP07/HiPER-agent`` at
commit ``ec5982a71635242de983a34407def05ac0d2e349``. The optional
termination head and task-specific reward shaping are intentionally excluded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HAESegment:
    """Inclusive turn positions for one high-level option."""

    start: int
    end: int
    next_start: int | None

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class HAEOutput:
    """Unnormalized logical low/high advantages and critic returns."""

    low_advantages: torch.Tensor
    low_returns: torch.Tensor
    high_advantages: torch.Tensor
    high_returns: torch.Tensor
    boundary_mask: torch.Tensor
    segments: tuple[HAESegment, ...]


def build_hae_segments(switches: Sequence[bool]) -> tuple[HAESegment, ...]:
    """Split turns at turn zero and every later ``SWITCH`` decision."""

    if not switches:
        raise ValueError("HAE requires at least one logical turn")
    if any(not isinstance(switch, bool) for switch in switches):
        raise ValueError("HAE switches must be booleans")

    boundaries = [0, *(index for index, switch in enumerate(switches[1:], start=1) if switch)]
    segments = []
    for segment_index, start in enumerate(boundaries):
        next_start = boundaries[segment_index + 1] if segment_index + 1 < len(boundaries) else None
        end = len(switches) - 1 if next_start is None else next_start - 1
        segments.append(HAESegment(start=start, end=end, next_start=next_start))
    return tuple(segments)


def population_normalize(values: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize with population std, returning zeros for degenerate groups."""

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("HAE normalization requires a non-empty one-dimensional tensor")
    if not torch.isfinite(values).all():
        raise ValueError("HAE normalization values must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("HAE normalization epsilon must be finite and positive")
    centered = values - values.mean()
    std = centered.square().mean().sqrt()
    return centered / (std + epsilon)


def compute_reference_hae(
    rewards: torch.Tensor,
    low_values: torch.Tensor,
    high_values: torch.Tensor,
    dones: torch.Tensor,
    switches: Sequence[bool],
    *,
    gamma: float,
    low_lambd: float,
    high_lambd: float,
) -> HAEOutput:
    """Compute upstream-compatible low-turn and high-segment HAE.

    Low GAE resets at every segment and bootstraps its final nonterminal turn
    from the next boundary's high value. High GAE operates on discounted
    segment rewards with the SMDP factor ``gamma ** duration``.
    """

    tensors = (rewards, low_values, high_values, dones)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("HAE rewards, values, and dones must be one-dimensional")
    turn_count = rewards.numel()
    if turn_count == 0 or any(tensor.numel() != turn_count for tensor in tensors[1:]):
        raise ValueError("HAE inputs must have the same nonzero turn count")
    if len(switches) != turn_count:
        raise ValueError("HAE switches must contain one decision per turn")
    if any(not torch.isfinite(tensor).all() for tensor in (rewards, low_values, high_values)):
        raise ValueError("HAE rewards and values must be finite")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("HAE gamma must lie in [0, 1]")
    if not math.isfinite(low_lambd) or not 0.0 <= low_lambd <= 1.0:
        raise ValueError("HAE low lambda must lie in [0, 1]")
    if not math.isfinite(high_lambd) or not 0.0 <= high_lambd <= 1.0:
        raise ValueError("HAE high lambda must lie in [0, 1]")

    dones = dones.to(device=low_values.device, dtype=torch.bool)
    rewards = rewards.to(device=low_values.device, dtype=low_values.dtype)
    high_values = high_values.to(device=low_values.device, dtype=low_values.dtype)
    terminal_positions = torch.nonzero(dones, as_tuple=False).flatten().tolist()
    if terminal_positions and terminal_positions != [turn_count - 1]:
        raise ValueError("HAE only permits a terminal marker on the final turn")

    segments = build_hae_segments(switches)
    zero = low_values.new_zeros(())
    low_advantages = torch.zeros_like(low_values)
    low_returns = torch.zeros_like(low_values)

    for segment in segments:
        next_advantage = zero
        bootstrap_high = high_values[segment.next_start].detach() if segment.next_start is not None else None
        for turn_position in range(segment.end, segment.start - 1, -1):
            nonterminal = (~dones[turn_position]).to(low_values.dtype)
            if turn_position < segment.end:
                next_value = low_values[turn_position + 1]
            elif bootstrap_high is not None:
                next_value = bootstrap_high
            else:
                next_value = zero
            delta = rewards[turn_position] + gamma * nonterminal * next_value - low_values[turn_position]
            advantage = delta + gamma * low_lambd * nonterminal * next_advantage
            low_advantages[turn_position] = advantage
            low_returns[turn_position] = advantage + low_values[turn_position]
            next_advantage = advantage

    high_advantages = torch.zeros_like(high_values)
    high_returns = torch.zeros_like(high_values)
    boundary_mask = torch.zeros(turn_count, device=low_values.device, dtype=torch.bool)
    next_segment_advantage = zero
    for segment in reversed(segments):
        boundary_mask[segment.start] = True
        segment_return = zero
        discount = 1.0
        for turn_position in range(segment.start, segment.end + 1):
            segment_return = segment_return + discount * rewards[turn_position]
            discount *= gamma

        end_nonterminal = (~dones[segment.end]).to(low_values.dtype)
        bootstrap_value = high_values[segment.next_start] if segment.next_start is not None else zero
        smdp_discount = gamma**segment.duration
        delta = segment_return + smdp_discount * end_nonterminal * bootstrap_value - high_values[segment.start]
        advantage = delta + smdp_discount * high_lambd * end_nonterminal * next_segment_advantage
        high_advantages[segment.start] = advantage
        high_returns[segment.start] = advantage + high_values[segment.start]
        next_segment_advantage = advantage

    return HAEOutput(
        low_advantages=low_advantages,
        low_returns=low_returns,
        high_advantages=high_advantages,
        high_returns=high_returns,
        boundary_mask=boundary_mask,
        segments=segments,
    )


__all__ = [
    "HAEOutput",
    "HAESegment",
    "build_hae_segments",
    "compute_reference_hae",
    "population_normalize",
]
