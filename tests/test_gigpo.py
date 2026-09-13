"""CPU contracts for M8.3 Group-in-Group Policy Optimization."""

from __future__ import annotations

import importlib
import math
import sys
import types
from argparse import Namespace

import pytest
import torch

from slime.utils.advantages.gigpo import (
    compute_gigpo_turn_credits,
    discounted_turn_returns,
    precompute_gigpo_train_data,
    relative_group_advantage,
)
from slime.utils.advantages.multi_turn import LogicalTurn
from slime.utils.dp_schedule import partition_train_data

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


def _logical_turn(
    rollout_id: int,
    group_index: int | None,
    turn_idx: int,
    reward: float,
    anchor_key: str | None,
    *,
    done: bool = False,
    sample_index: int | None = None,
    truncated: bool = False,
) -> LogicalTurn:
    return LogicalTurn(
        rollout_id=rollout_id,
        group_index=group_index,
        sample_index=rollout_id if sample_index is None else sample_index,
        turn_idx=turn_idx,
        response_start=0,
        response_end=1,
        value_position=0,
        reward=reward,
        done=done,
        anchor_key=anchor_key,
        truncated=truncated,
    )


def _raw_turn(
    turn_idx: int,
    span: list[int],
    reward: float,
    anchor_key: str | None,
    *,
    done: bool = False,
    truncated: bool = False,
) -> dict:
    return {
        "turn_idx": turn_idx,
        "response_span": span,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "anchor_key": anchor_key,
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


def _metadata(*turns: dict, context_revision: int = 0, dropped_turns: list[dict] | None = None) -> dict:
    multi_turn = {
        "version": 1,
        "context_revision": context_revision,
        "turns": list(turns),
    }
    if dropped_turns is not None:
        multi_turn["dropped_turns"] = dropped_turns
    return {"multi_turn": multi_turn}


def _paper_fixture() -> list[LogicalTurn]:
    # M8 plan section 7.9, derived from GiGPO Eq. (3), (5), (7), and (8).
    return [
        _logical_turn(0, 0, 0, 0.0, "A"),
        _logical_turn(0, 0, 1, 1.0, "B", done=True),
        _logical_turn(1, 0, 0, 0.0, "A"),
        _logical_turn(1, 0, 1, 0.0, "C", done=True),
    ]


@pytest.mark.unit
def test_required_gigpo_numerical_fixture() -> None:
    credits = compute_gigpo_turn_credits(
        _paper_fixture(),
        gamma=0.9,
        step_advantage_weight=1.0,
        normalization="mean",
    )

    assert credits == pytest.approx({(0, 0): 0.95, (0, 1): 0.5, (1, 0): -0.95, (1, 1): -0.5})


@pytest.mark.unit
def test_discounted_turn_return_is_turn_time_suffix_return() -> None:
    turns = [
        _logical_turn(0, 0, index, reward, f"s{index}", done=index == 3) for index, reward in enumerate([0, 0, 0, 1])
    ]

    returns = discounted_turn_returns(turns, gamma=0.95)

    assert [returns[(0, index)] for index in range(4)] == pytest.approx([0.95**3, 0.95**2, 0.95, 1.0])


@pytest.mark.unit
def test_episode_and_anchor_credit_can_be_isolated_with_step_weight() -> None:
    turns = _paper_fixture()
    episode_only = compute_gigpo_turn_credits(turns, gamma=0.9, step_advantage_weight=0.0, normalization="mean")
    combined = compute_gigpo_turn_credits(turns, gamma=0.9, step_advantage_weight=2.0, normalization="mean")

    assert episode_only == pytest.approx({(0, 0): 0.5, (0, 1): 0.5, (1, 0): -0.5, (1, 1): -0.5})
    assert combined == pytest.approx({(0, 0): 1.4, (0, 1): 0.5, (1, 0): -1.4, (1, 1): -0.5})


@pytest.mark.unit
def test_mean_std_uses_sample_standard_deviation() -> None:
    assert relative_group_advantage([-1.0, 1.0], mode="mean") == pytest.approx([-1.0, 1.0])
    assert relative_group_advantage([-1.0, 1.0], mode="mean_std") == pytest.approx(
        [-1 / math.sqrt(2), 1 / math.sqrt(2)], abs=1e-6
    )
    assert relative_group_advantage([5.0], mode="mean_std") == [0.0]
    assert relative_group_advantage([5.0, 5.0], mode="mean_std") == [0.0, 0.0]


@pytest.mark.unit
def test_repeated_anchor_occurrences_are_not_deduplicated() -> None:
    turns = [
        _logical_turn(0, 1, 0, 0.0, "same"),
        _logical_turn(0, 1, 1, 2.0, "same", done=True),
        _logical_turn(1, 1, 0, 0.0, "same", done=True),
    ]

    episode_only = compute_gigpo_turn_credits(turns, normalization="mean", step_advantage_weight=0.0)
    combined = compute_gigpo_turn_credits(turns, normalization="mean", step_advantage_weight=1.0)

    step_credit = {key: combined[key] - episode_only[key] for key in combined}
    assert step_credit == pytest.approx({(0, 0): 2 / 3, (0, 1): 2 / 3, (1, 0): -4 / 3})


@pytest.mark.unit
def test_identical_anchor_keys_do_not_cross_episode_groups() -> None:
    turns = [
        _logical_turn(0, 0, 0, 1.0, "same", done=True),
        _logical_turn(1, 1, 0, 0.0, "same", done=True),
    ]

    credits = compute_gigpo_turn_credits(turns, normalization="mean")

    assert credits == {(0, 0): 0.0, (1, 0): 0.0}


@pytest.mark.unit
def test_ragged_trajectories_and_permutation_are_supported() -> None:
    turns = [
        _logical_turn(10, 3, 0, 0.0, "A"),
        _logical_turn(10, 3, 1, 2.0, "B", done=True),
        _logical_turn(11, 3, 0, 1.0, "A", done=True),
        _logical_turn(20, 4, 0, 9.0, "A", done=True),
    ]

    forward = compute_gigpo_turn_credits(turns, gamma=0.5, normalization="mean_std")
    reverse = compute_gigpo_turn_credits(list(reversed(turns)), gamma=0.5, normalization="mean_std")

    assert reverse == pytest.approx(forward)


@pytest.mark.unit
def test_precompute_reconstructs_clean_and_fork_siblings() -> None:
    unsplit = {
        "metadata": [
            _metadata(_raw_turn(0, [0, 1], 0.0, "A"), _raw_turn(1, [2, 3], 1.0, "B", done=True)),
            _metadata(_raw_turn(0, [0, 1], 0.0, "A"), _raw_turn(1, [2, 3], 0.0, "C", done=True)),
        ],
        "rollout_ids": [10, 11],
        "group_indices": [6, 6],
        "response_lengths": [3, 3],
        "loss_masks": [[1, 0, 1], [1, 0, 1]],
    }
    siblings = {
        "metadata": [
            _metadata(_raw_turn(0, [0, 1], 0.0, "A")),
            _metadata(_raw_turn(1, [1, 2], 1.0, "B", done=True), context_revision=1),
            unsplit["metadata"][1],
        ],
        "rollout_ids": [10, 10, 11],
        "group_indices": [6, 6, 6],
        "response_lengths": [1, 2, 3],
        "loss_masks": [[1], [0, 1], [1, 0, 1]],
    }

    unsplit_credits = precompute_gigpo_train_data(unsplit, gamma=0.9, normalization="mean")
    sibling_credits = precompute_gigpo_train_data(siblings, gamma=0.9, normalization="mean")
    unsplit_by_id = {
        (10, 0): unsplit_credits[0][0],
        (10, 1): unsplit_credits[0][1],
        (11, 0): unsplit_credits[1][0],
        (11, 1): unsplit_credits[1][1],
    }
    sibling_by_id = {
        (10, 0): sibling_credits[0][0],
        (10, 1): sibling_credits[1][0],
        (11, 0): sibling_credits[2][0],
        (11, 1): sibling_credits[2][1],
    }

    assert sibling_by_id == pytest.approx(unsplit_by_id)
    siblings["turn_credits"] = sibling_credits
    partitioned = partition_train_data(siblings, [2, 0])
    assert partitioned["turn_credits"] == [sibling_credits[2], sibling_credits[0]]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda turns: turns.__setitem__(0, _logical_turn(0, None, 0, 0.0, "A")), "group_index"),
        (lambda turns: turns.__setitem__(0, _logical_turn(0, 0, 0, 0.0, None)), "anchor_key"),
        (lambda turns: turns.__setitem__(0, _logical_turn(0, 0, 0, 0.0, "")), "anchor_key"),
        (lambda turns: turns.__setitem__(0, _logical_turn(0, -1, 0, 0.0, "A")), "group_index"),
        (lambda turns: turns.__setitem__(0, _logical_turn(0, 0, 0, 0.0, 123)), "anchor_key"),
        (lambda turns: turns.__setitem__(0, _logical_turn(0, 0, 0, float("nan"), "A")), "reward"),
        (lambda turns: turns.__setitem__(1, _logical_turn(0, 1, 1, 1.0, "B", done=True)), "multiple episode groups"),
        (lambda turns: turns.append(_logical_turn(0, 0, 0, 0.0, "D")), "multiple owners"),
        (lambda turns: turns.__setitem__(0, _logical_turn(0, 0, 0, 0.0, "A", truncated=True)), "truncated"),
    ],
)
def test_invalid_turn_identity_or_trajectory_fails_closed(mutation, message: str) -> None:
    turns = _paper_fixture()
    mutation(turns)

    with pytest.raises(ValueError, match=message):
        compute_gigpo_turn_credits(turns, normalization="mean")


@pytest.mark.unit
def test_missing_turn_index_and_dropped_credit_history_fail_closed() -> None:
    with pytest.raises(ValueError, match="complete turn sequence"):
        compute_gigpo_turn_credits(
            [_logical_turn(0, 0, 1, 1.0, "A", done=True)],
            normalization="mean",
        )

    train_data = {
        "metadata": [
            _metadata(
                _raw_turn(0, [0, 1], 1.0, "A", done=True),
                dropped_turns=[{"turn_idx": 1, "reason": "realigned_context", "generated_tokens": 2}],
            )
        ],
        "rollout_ids": [0],
        "group_indices": [0],
        "response_lengths": [1],
        "loss_masks": [[1]],
    }
    with pytest.raises(ValueError, match="dropped turns"):
        precompute_gigpo_train_data(train_data)

    train_data["metadata"] = [_metadata(_raw_turn(0, [0, 1], 1.0, "A", done=True))]
    train_data["truncated"] = [1]
    with pytest.raises(ValueError, match="collector-truncated"):
        precompute_gigpo_train_data(train_data)


@pytest.mark.unit
def test_partial_turn_span_fails_closed() -> None:
    train_data = {
        "metadata": [_metadata(_raw_turn(0, [0, 2], 1.0, "A", done=True))],
        "rollout_ids": [0],
        "group_indices": [0],
        "response_lengths": [2],
        "loss_masks": [[1, 0]],
    }

    with pytest.raises(ValueError, match="not fully train-owned"):
        precompute_gigpo_train_data(train_data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gamma": -0.1}, "gamma"),
        ({"gamma": float("nan")}, "gamma"),
        ({"step_advantage_weight": -0.1}, "step advantage weight"),
        ({"step_advantage_weight": float("inf")}, "step advantage weight"),
        ({"normalization": "invalid"}, "normalization"),
    ],
)
def test_invalid_gigpo_configuration_fails_closed(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_gigpo_turn_credits(_paper_fixture(), **kwargs)


@pytest.mark.unit
def test_training_bridge_projects_turn_credit_without_token_length_weighting_or_whitening(
    monkeypatch, loss_module
) -> None:
    rollout_data = {
        "log_probs": [torch.zeros(7)],
        "ref_log_probs": [torch.zeros(7)],
        "rewards": [99.0],
        "metadata": [_metadata(_raw_turn(0, [0, 1], 0.0, "A"), _raw_turn(1, [2, 7], 1.0, "B", done=True))],
        "rollout_ids": [4],
        "turn_credits": [[0.95, 0.5]],
        "total_lengths": [9],
        "response_lengths": [7],
        "loss_masks": [torch.tensor([1, 0, 1, 1, 1, 1, 1])],
    }
    args = Namespace(
        advantage_estimator="gigpo",
        use_rollout_logprobs=False,
        kl_coef=0.0,
        kl_loss_type="k1",
        custom_advantage_function_path=None,
        use_opd=False,
        normalize_advantages=True,
    )
    monkeypatch.setattr(
        loss_module,
        "distributed_masked_whiten",
        lambda *_args, **_kwargs: pytest.fail("precomputed GiGPO credit must not be normalized again"),
    )

    loss_module.compute_advantages_and_returns(args, rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor([0.95, 0, 0.5, 0.5, 0.5, 0.5, 0.5]))
    torch.testing.assert_close(rollout_data["returns"][0], rollout_data["advantages"][0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
