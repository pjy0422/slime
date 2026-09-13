"""CPU contracts for M8.1 logical-turn PPO."""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch

from slime.utils.advantages.mt_ppo import collect_logical_turns, compute_turn_gae, project_turn_values

NUM_GPUS = 0


@pytest.fixture
def loss_module(monkeypatch):
    import slime.backends.megatron_utils as megatron_utils

    missing = object()
    original_modules = {
        name: sys.modules.get(name, missing)
        for name in ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    }
    original_attributes = {name: getattr(megatron_utils, name, missing) for name in ("loss", "cp_utils")}
    sys.modules.pop("slime.backends.megatron_utils.loss", None)
    sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
    for name in ("loss", "cp_utils"):
        if hasattr(megatron_utils, name):
            delattr(megatron_utils, name)
    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
        get_data_parallel_group=lambda **_kwargs: None,
        get_data_parallel_world_size=lambda **_kwargs: 1,
        is_pipeline_last_stage=lambda: True,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        yield importlib.import_module("slime.backends.megatron_utils.loss")
    finally:
        for name, original in original_modules.items():
            sys.modules.pop(name, None)
            if original is not missing:
                sys.modules[name] = original
        for name, original in original_attributes.items():
            if original is missing:
                if hasattr(megatron_utils, name):
                    delattr(megatron_utils, name)
            else:
                setattr(megatron_utils, name, original)


def _turn(turn_idx: int, start: int, end: int, reward: float, *, done: bool = False) -> dict:
    return {
        "turn_idx": turn_idx,
        "response_span": [start, end],
        "reward": reward,
        "done": done,
        "truncated": False,
        "anchor_key": None,
        "switch": None,
        "role_spans": {
            "switch": [],
            "subgoal": [],
            "high_subgoal": [],
            "low_subgoal": [],
            "action": [],
        },
        "value_positions": {"high": None, "low": None},
        "format_valid": True,
    }


def _metadata(*turns: dict, context_revision: int = 0) -> dict:
    return {
        "multi_turn": {
            "version": 1,
            "context_revision": context_revision,
            "turns": list(turns),
        }
    }


@pytest.mark.unit
def test_turn_gae_matches_hand_calculated_reference() -> None:
    advantages, returns = compute_turn_gae(
        rewards=torch.tensor([1.0, 0.5, 2.0]),
        values=torch.tensor([0.2, 0.4, 0.1]),
        dones=torch.tensor([False, False, True]),
        gamma=0.9,
        lambd=0.8,
    )

    torch.testing.assert_close(advantages, torch.tensor([2.28176, 1.558, 1.9]))
    torch.testing.assert_close(returns, torch.tensor([2.48176, 1.958, 2.0]))


@pytest.mark.unit
def test_terminal_turn_cuts_value_bootstrap_and_gae_recurrence() -> None:
    advantages, returns = compute_turn_gae(
        rewards=torch.tensor([1.0, 10.0]),
        values=torch.tensor([0.5, 2.0]),
        dones=torch.tensor([True, False]),
        gamma=1.0,
        lambd=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([0.5, 8.0]))
    torch.testing.assert_close(returns, torch.tensor([1.0, 10.0]))


@pytest.mark.unit
def test_single_turn_gae_reduces_to_reward_minus_value() -> None:
    advantages, returns = compute_turn_gae(
        rewards=torch.tensor([3.0]),
        values=torch.tensor([0.25]),
        dones=torch.tensor([False]),
        gamma=0.95,
        lambd=0.7,
    )

    torch.testing.assert_close(advantages, torch.tensor([2.75]))
    torch.testing.assert_close(returns, torch.tensor([3.0]))


@pytest.mark.unit
def test_turn_projection_is_length_independent_and_critic_targets_are_sparse() -> None:
    metadata = [_metadata(_turn(0, 0, 2, 0.0), _turn(1, 4, 9, 1.0, done=True))]
    reference = [torch.zeros(9)]
    turns = collect_logical_turns(metadata, [11], [9], [torch.tensor([1, 1, 0, 0, 1, 1, 1, 1, 1])])
    scalars = {(11, 0): torch.tensor(-1.0), (11, 1): torch.tensor(1.0)}

    actor_values, actor_masks = project_turn_values([9], turns, scalars, sparse=False, reference_tensors=reference)
    critic_values, critic_masks = project_turn_values([9], turns, scalars, sparse=True, reference_tensors=reference)

    torch.testing.assert_close(actor_values[0], torch.tensor([-1, -1, 0, 0, 1, 1, 1, 1, 1.0]))
    torch.testing.assert_close(actor_masks[0], torch.tensor([1, 1, 0, 0, 1, 1, 1, 1, 1.0]))
    torch.testing.assert_close(critic_values[0], torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, 0.0]))
    torch.testing.assert_close(critic_masks[0], torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0.0]))


@pytest.mark.unit
def test_compacted_sibling_layout_preserves_logical_turn_identity_and_credit() -> None:
    compacted_metadata = [
        _metadata(_turn(0, 0, 2, 0.0), context_revision=0),
        _metadata(_turn(1, 1, 4, 2.0, done=True), context_revision=2),
    ]
    turns = collect_logical_turns(
        compacted_metadata,
        [7, 7],
        [2, 4],
        [torch.ones(2), torch.tensor([0, 1, 1, 1])],
    )
    assert [(turn.rollout_id, turn.turn_idx, turn.sample_index) for turn in turns] == [(7, 0, 0), (7, 1, 1)]

    advantages, returns = compute_turn_gae(
        rewards=torch.tensor([0.0, 2.0]),
        values=torch.tensor([0.25, 0.5]),
        dones=torch.tensor([False, True]),
        gamma=1.0,
        lambd=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([1.75, 1.5]))
    torch.testing.assert_close(returns, torch.tensor([2.0, 2.0]))


@pytest.mark.unit
def test_layout_rejects_a_terminal_turn_before_rollout_end() -> None:
    metadata = [_metadata(_turn(0, 0, 1, 1.0, done=True), _turn(1, 1, 2, 1.0))]

    with pytest.raises(ValueError, match="non-final terminal"):
        collect_logical_turns(metadata, [3], [2], [torch.ones(2)])


@pytest.mark.unit
def test_training_bridge_applies_turn_kl_and_builds_sparse_critic_targets(loss_module) -> None:
    loss = loss_module
    rollout_data = {
        "values": [torch.tensor([0.2, 9.0, 9.0, 0.4, 9.0, 9.0])],
        "metadata": [_metadata(_turn(0, 0, 2, 0.0), _turn(1, 3, 6, 2.0, done=True))],
        "rollout_ids": [5],
        "total_lengths": [8],
        "response_lengths": [6],
        "loss_masks": [torch.tensor([1, 1, 0, 1, 1, 1])],
    }
    args = Namespace(kl_coef=0.5, gamma=1.0, lambd=1.0, normalize_advantages=False)

    advantages, returns = loss._compute_multi_turn_ppo_advantages(
        args,
        rollout_data,
        [torch.tensor([0.1, 0.2, 0.0, 0.3, 0.4, 0.5])],
    )

    torch.testing.assert_close(advantages[0], torch.tensor([1.05, 1.05, 0.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(returns[0], torch.tensor([1.25, 0.0, 0.0, 1.4, 0.0, 0.0]))
    torch.testing.assert_close(rollout_data["value_masks"][0], torch.tensor([1, 0, 0, 1, 0, 0.0]))
    torch.testing.assert_close(rollout_data["value_mask_sums"], torch.tensor([2.0]))


@pytest.mark.unit
def test_training_bridge_normalizes_turns_before_variable_length_projection(monkeypatch, loss_module) -> None:
    loss = loss_module
    observed = {}

    def fake_whiten(values, mask, **kwargs):
        observed["values"] = values.clone()
        observed["mask"] = mask.clone()
        observed["group"] = kwargs["process_group"]
        return torch.tensor([-2.0, 2.0])

    monkeypatch.setattr(loss, "distributed_masked_whiten", fake_whiten)
    rollout_data = {
        "values": [torch.zeros(7)],
        "metadata": [_metadata(_turn(0, 0, 1, 1.0), _turn(1, 2, 7, 3.0, done=True))],
        "rollout_ids": [9],
        "total_lengths": [10],
        "response_lengths": [7],
        "loss_masks": [torch.tensor([1, 0, 1, 1, 1, 1, 1])],
    }
    args = Namespace(kl_coef=0.0, gamma=0.0, lambd=0.0, normalize_advantages=True)

    advantages, _ = loss._compute_multi_turn_ppo_advantages(args, rollout_data, [torch.zeros(7)])

    assert observed["values"].numel() == 2
    torch.testing.assert_close(observed["values"], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(observed["mask"], torch.ones(2))
    torch.testing.assert_close(advantages[0], torch.tensor([-2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0]))


@pytest.mark.unit
def test_value_loss_reducer_uses_sparse_turn_mask_and_turn_denominator(monkeypatch, loss_module) -> None:
    loss = loss_module

    def fake_value_loss(_args, _batch, _logits, reducer):
        reduced = reducer(torch.arange(6, dtype=torch.float32))
        return reduced, {"value_loss": reduced.detach()}

    monkeypatch.setattr(loss, "value_loss_function", fake_value_loss)
    args = Namespace(
        loss_type="value_loss",
        advantage_estimator="multi_turn_ppo",
        calculate_per_token_loss=False,
        custom_loss_function_path=None,
        recompute_loss_function=False,
        allgather_cp=False,
    )
    batch = {
        "loss_masks": [torch.ones(6)],
        "rollout_mask_sums": [torch.tensor(6.0)],
        "value_masks": [torch.tensor([1, 0, 0, 1, 0, 0.0])],
        "value_mask_sums": [torch.tensor(2.0)],
        "total_lengths": [8],
        "response_lengths": [6],
    }

    reduced, _, _ = loss.loss_function(args, batch, 1, 1, torch.zeros(1, 8, 1))

    # Sparse turn mean: (position 0 + position 3) / 2, not all-token mean.
    torch.testing.assert_close(reduced, torch.tensor(1.5))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
