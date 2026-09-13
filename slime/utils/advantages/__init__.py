"""Logical credit-assignment helpers."""

from .dcgrpo import compute_dcgrpo_turn_credits, precompute_dcgrpo_train_data
from .gigpo import compute_gigpo_turn_credits, discounted_turn_returns, precompute_gigpo_train_data
from .mt_ppo import compute_turn_gae
from .multi_turn import LogicalTurn, collect_logical_turns, pack_turn_credits, project_turn_values, unpack_turn_credits

__all__ = [
    "LogicalTurn",
    "collect_logical_turns",
    "compute_dcgrpo_turn_credits",
    "compute_gigpo_turn_credits",
    "compute_turn_gae",
    "discounted_turn_returns",
    "pack_turn_credits",
    "precompute_dcgrpo_train_data",
    "precompute_gigpo_train_data",
    "project_turn_values",
    "unpack_turn_credits",
]
