"""CPU contracts for M8.2 decomposed-credit GRPO."""

from __future__ import annotations

import importlib
import math
import sys
import types
from argparse import Namespace

import pytest
import torch

from slime.utils.advantages.dcgrpo import compute_dcgrpo_turn_credits, precompute_dcgrpo_train_data
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
    group_index: int,
    turn_idx: int,
    reward: float,
    *,
    done: bool = False,
    sample_index: int | None = None,
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
    )


def _raw_turn(turn_idx: int, span: list[int], reward: float, *, done: bool = False) -> dict:
    return {
        "turn_idx": turn_idx,
        "response_span": span,
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


def _credits_by_identity(train_data: dict, packed: list[list[float]]) -> dict[tuple[int, int], float]:
    return {
        (rollout_id, turn["turn_idx"]): credit
        for rollout_id, metadata, sample_credits in zip(
            train_data["rollout_ids"], train_data["metadata"], packed, strict=True
        )
        for turn, credit in zip(metadata["multi_turn"]["turns"], sample_credits, strict=True)
    }


@pytest.mark.unit
def test_dw_normalizes_discounted_return_to_go_with_default_gamma() -> None:
    turns = [
        _logical_turn(0, 7, 0, -1.0),
        _logical_turn(0, 7, 1, 0.0, done=True),
        _logical_turn(1, 7, 0, 0.0),
        _logical_turn(1, 7, 1, 0.0, done=True),
        _logical_turn(2, 7, 0, 0.0),
        _logical_turn(2, 7, 1, 1.0, done=True),
    ]

    credits = compute_dcgrpo_turn_credits(turns, mode="dw")
    symmetric_z = math.sqrt(1.5)

    assert credits[(0, 0)] == pytest.approx(-symmetric_z)
    assert credits[(1, 0)] == pytest.approx(0.0)
    assert credits[(2, 0)] == pytest.approx(symmetric_z)
    assert credits[(0, 1)] == pytest.approx(-1 / math.sqrt(2))
    assert credits[(1, 1)] == pytest.approx(-1 / math.sqrt(2))
    assert credits[(2, 1)] == pytest.approx(math.sqrt(2))


@pytest.mark.unit
def test_sw_separately_normalizes_immediate_and_future_credit_with_default_alpha() -> None:
    turns = [
        _logical_turn(0, 4, 0, -1.0),
        _logical_turn(0, 4, 1, 1.0, done=True),
        _logical_turn(1, 4, 0, 0.0),
        _logical_turn(1, 4, 1, 0.0, done=True),
        _logical_turn(2, 4, 0, 1.0),
        _logical_turn(2, 4, 1, -1.0, done=True),
    ]

    default_credits = compute_dcgrpo_turn_credits(turns, mode="sw")
    half_future = compute_dcgrpo_turn_credits(turns, mode="sw", alpha=0.5)
    symmetric_z = math.sqrt(1.5)

    assert [default_credits[(rollout_id, 0)] for rollout_id in range(3)] == pytest.approx([0.0, 0.0, 0.0])
    assert [default_credits[(rollout_id, 1)] for rollout_id in range(3)] == pytest.approx(
        [symmetric_z, 0.0, -symmetric_z]
    )
    assert [half_future[(rollout_id, 0)] for rollout_id in range(3)] == pytest.approx(
        [-0.5 * symmetric_z, 0.0, 0.5 * symmetric_z]
    )


@pytest.mark.unit
def test_gamma_changes_return_to_go_before_dw_normalization() -> None:
    turns = [
        _logical_turn(0, 2, 0, 0.0),
        _logical_turn(0, 2, 1, 0.0),
        _logical_turn(0, 2, 2, 2.0, done=True),
        _logical_turn(1, 2, 0, 0.0),
        _logical_turn(1, 2, 1, 2.0),
        _logical_turn(1, 2, 2, 0.0, done=True),
    ]

    discounted = compute_dcgrpo_turn_credits(turns, mode="dw", gamma=0.5)
    undiscounted = compute_dcgrpo_turn_credits(turns, mode="dw", gamma=1.0)

    assert [discounted[(rollout_id, 0)] for rollout_id in range(2)] == pytest.approx([-1.0, 1.0])
    assert [undiscounted[(rollout_id, 0)] for rollout_id in range(2)] == pytest.approx([0.0, 0.0])


@pytest.mark.unit
def test_final_turn_dw_sw_and_single_turn_grpo_credit_are_equivalent() -> None:
    turns = [
        _logical_turn(0, 3, 0, -1.0, done=True),
        _logical_turn(1, 3, 0, 0.0, done=True),
        _logical_turn(2, 3, 0, 1.0, done=True),
    ]

    dw = compute_dcgrpo_turn_credits(turns, mode="dw")
    sw = compute_dcgrpo_turn_credits(turns, mode="sw")
    expected = [-math.sqrt(1.5), 0.0, math.sqrt(1.5)]

    assert [dw[(rollout_id, 0)] for rollout_id in range(3)] == pytest.approx(expected)
    assert [sw[(rollout_id, 0)] for rollout_id in range(3)] == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["dw", "sw"])
def test_singleton_and_zero_variance_components_have_zero_credit(mode: str) -> None:
    singleton = [_logical_turn(0, 0, 0, 5.0, done=True)]
    tied = [_logical_turn(rollout_id, 1, 0, 2.0, done=True) for rollout_id in range(3)]

    assert compute_dcgrpo_turn_credits(singleton, mode=mode)[(0, 0)] == 0.0
    assert set(compute_dcgrpo_turn_credits(tied, mode=mode).values()) == {0.0}


@pytest.mark.unit
def test_ragged_trajectories_include_terminal_zero_in_sw_future_cohort() -> None:
    turns = [
        _logical_turn(0, 5, 0, 0.0),
        _logical_turn(0, 5, 1, 2.0, done=True),
        _logical_turn(1, 5, 0, 2.0, done=True),
    ]

    sw = compute_dcgrpo_turn_credits(turns, mode="sw")
    dw = compute_dcgrpo_turn_credits(turns, mode="dw")

    assert [sw[(rollout_id, 0)] for rollout_id in range(2)] == pytest.approx([0.0, 0.0])
    assert sw[(0, 1)] == 0.0
    assert set(dw.values()) == {0.0}


@pytest.mark.unit
def test_credit_is_invariant_to_input_permutation() -> None:
    turns = [
        _logical_turn(10, 1, 0, 0.0),
        _logical_turn(10, 1, 1, 2.0, done=True),
        _logical_turn(11, 1, 0, 3.0),
        _logical_turn(11, 1, 1, 0.0, done=True),
        _logical_turn(20, 2, 0, 9.0, done=True),
    ]

    forward = compute_dcgrpo_turn_credits(turns, mode="sw")
    reverse = compute_dcgrpo_turn_credits(list(reversed(turns)), mode="sw")

    assert reverse == pytest.approx(forward)


@pytest.mark.unit
def test_precompute_is_compaction_invariant_and_survives_dp_partition() -> None:
    unsplit = {
        "metadata": [
            _metadata(_raw_turn(0, [0, 1], 0.0), _raw_turn(1, [2, 3], 3.0, done=True)),
            _metadata(_raw_turn(0, [0, 1], 2.0), _raw_turn(1, [2, 3], 0.0, done=True)),
        ],
        "rollout_ids": [10, 11],
        "group_indices": [6, 6],
        "response_lengths": [3, 3],
        "loss_masks": [[1, 0, 1], [1, 0, 1]],
    }
    compacted = {
        "metadata": [
            _metadata(_raw_turn(0, [0, 1], 0.0)),
            _metadata(_raw_turn(1, [1, 2], 3.0, done=True), context_revision=1),
            unsplit["metadata"][1],
        ],
        "rollout_ids": [10, 10, 11],
        "group_indices": [6, 6, 6],
        "response_lengths": [1, 2, 3],
        "loss_masks": [[1], [0, 1], [1, 0, 1]],
    }

    unsplit_packed = precompute_dcgrpo_train_data(unsplit, mode="dw")
    compacted_packed = precompute_dcgrpo_train_data(compacted, mode="dw")

    assert _credits_by_identity(compacted, compacted_packed) == pytest.approx(
        _credits_by_identity(unsplit, unsplit_packed)
    )
    compacted["turn_credits"] = compacted_packed
    partitioned = partition_train_data(compacted, [1, 0])
    assert partitioned["turn_credits"] == [compacted_packed[1], compacted_packed[0]]


@pytest.mark.unit
def test_precompute_rejects_missing_group_identity() -> None:
    train_data = {
        "metadata": [_metadata(_raw_turn(0, [0, 1], 1.0, done=True))],
        "rollout_ids": [0],
        "group_indices": [None],
        "response_lengths": [1],
        "loss_masks": [[1]],
    }

    with pytest.raises(ValueError, match="requires group_index"):
        precompute_dcgrpo_train_data(train_data, mode="dw")


@pytest.mark.unit
def test_training_bridge_projects_precomputed_credit_without_second_normalization(monkeypatch, loss_module) -> None:
    loss = loss_module
    rollout_data = {
        "log_probs": [torch.zeros(6)],
        "ref_log_probs": [torch.zeros(6)],
        "rewards": [99.0],
        "metadata": [_metadata(_raw_turn(0, [0, 2], 0.0), _raw_turn(1, [3, 6], 1.0, done=True))],
        "rollout_ids": [4],
        "turn_credits": [[-1.0, 2.0]],
        "total_lengths": [8],
        "response_lengths": [6],
        "loss_masks": [torch.tensor([1, 1, 0, 1, 1, 1])],
    }
    args = Namespace(
        advantage_estimator="dcgrpo",
        use_rollout_logprobs=False,
        kl_coef=0.0,
        kl_loss_type="k1",
        custom_advantage_function_path=None,
        use_opd=False,
        normalize_advantages=True,
    )
    monkeypatch.setattr(
        loss,
        "distributed_masked_whiten",
        lambda *_args, **_kwargs: pytest.fail("precomputed DC-GRPO credit must not be normalized again"),
    )

    loss.compute_advantages_and_returns(args, rollout_data)

    torch.testing.assert_close(rollout_data["advantages"][0], torch.tensor([-1, -1, 0, 2, 2, 2.0]))
    torch.testing.assert_close(rollout_data["returns"][0], rollout_data["advantages"][0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "invalid"}, "unsupported"),
        ({"mode": "dw", "gamma": -0.1}, "gamma"),
        ({"mode": "sw", "alpha": -0.1}, "alpha"),
    ],
)
def test_invalid_dcgrpo_configuration_fails_closed(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_dcgrpo_turn_credits([_logical_turn(0, 0, 0, 1.0, done=True)], **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("turns", "message"),
    [
        (
            [_logical_turn(0, 0, 0, 1.0), _logical_turn(0, 0, 0, 2.0, done=True)],
            "multiple owners",
        ),
        (
            [_logical_turn(0, 0, 0, 1.0, done=True), _logical_turn(0, 0, 1, 2.0)],
            "non-final terminal",
        ),
    ],
)
def test_malformed_logical_trajectory_fails_closed(turns: list[LogicalTurn], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_dcgrpo_turn_credits(turns, mode="dw")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
