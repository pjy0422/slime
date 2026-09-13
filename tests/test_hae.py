"""CPU contracts for M8.4b reference HiPER/HAE."""

from __future__ import annotations

import importlib
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from slime.agent.hae import parse_reference_hae_response
from slime.utils.advantages.hae import build_hae_segments, compute_reference_hae, population_normalize

NUM_GPUS = 0


@pytest.fixture
def loss_module(monkeypatch):
    import slime.backends.megatron_utils as megatron_utils

    missing = object()
    names = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    originals = {name: sys.modules.get(name, missing) for name in names}
    original_attrs = {name: getattr(megatron_utils, name, missing) for name in ("loss", "cp_utils")}
    for name in names:
        sys.modules.pop(name, None)
    for name in ("loss", "cp_utils"):
        if hasattr(megatron_utils, name):
            delattr(megatron_utils, name)
    mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
        get_data_parallel_group=lambda **_kwargs: None,
        get_data_parallel_world_size=lambda **_kwargs: 1,
        is_pipeline_last_stage=lambda: True,
    )
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    try:
        yield importlib.import_module("slime.backends.megatron_utils.loss")
    finally:
        for name, original in originals.items():
            sys.modules.pop(name, None)
            if original is not missing:
                sys.modules[name] = original
        for name, original in original_attrs.items():
            if original is missing:
                if hasattr(megatron_utils, name):
                    delattr(megatron_utils, name)
            else:
                setattr(megatron_utils, name, original)


def _turn(turn_idx: int, start: int, *, switch: str, reward: float, done: bool) -> dict:
    boundary = turn_idx == 0 or switch == "SWITCH"
    return {
        "turn_idx": turn_idx,
        "response_span": [start, start + 8],
        "reward": reward,
        "done": done,
        "truncated": False,
        "anchor_key": None,
        "switch": switch,
        "role_spans": {
            "switch": [[start, start + 1]],
            "subgoal": [[start + 1, start + 4]],
            "high_subgoal": [[start + 1, start + 4]],
            "low_subgoal": [],
            "action": [[start + 5, start + 8]],
        },
        "value_positions": {"high": start if boundary else None, "low": start + 4},
        "format_valid": True,
    }


def _metadata(*turns: dict) -> dict:
    return {"multi_turn": {"version": 1, "context_revision": 0, "turns": list(turns)}}


@pytest.mark.unit
def test_reference_fixture_matches_upstream_low_and_high_recurrences() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "hae_reference_v1.json").read_text())
    inputs = fixture["inputs"]
    result = compute_reference_hae(
        torch.tensor(inputs["rewards"]),
        torch.tensor(inputs["low_values"]),
        torch.tensor(inputs["high_values"]),
        torch.tensor(inputs["dones"]),
        inputs["switches"],
        gamma=inputs["gamma"],
        low_lambd=inputs["low_lambd"],
        high_lambd=inputs["high_lambd"],
    )
    expected = fixture["expected"]
    assert [[segment.start, segment.end, segment.next_start] for segment in result.segments] == expected["segments"]
    assert result.boundary_mask.tolist() == expected["boundary_mask"]
    for field in ("low_advantages", "low_returns", "high_advantages", "high_returns"):
        torch.testing.assert_close(getattr(result, field), torch.tensor(expected[field]), atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_segments_start_at_zero_and_switch_turns() -> None:
    assert [(s.start, s.end, s.next_start) for s in build_hae_segments([False, False, True, False, True])] == [
        (0, 1, 2),
        (2, 3, 4),
        (4, 4, None),
    ]


@pytest.mark.unit
def test_population_normalization_uses_population_std_and_handles_singletons() -> None:
    torch.testing.assert_close(population_normalize(torch.tensor([1.0, 3.0])), torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(population_normalize(torch.tensor([7.0])), torch.tensor([0.0]))


class _PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[index] for index in ids)


@pytest.mark.unit
def test_reference_parser_includes_tags_and_places_values() -> None:
    pieces = [
        "<switch>",
        "SWITCH",
        "</switch>",
        "<subgoal>",
        "inspect",
        "</subgoal>",
        "<action>",
        "open",
        "</action>",
    ]
    parsed = parse_reference_hae_response(range(len(pieces)), _PieceTokenizer(pieces), turn_idx=2)["multi_turn"]
    assert parsed["format_valid"] is True
    assert parsed["switch"] == "SWITCH"
    assert parsed["role_spans"]["switch"] == [[0, 3]]
    assert parsed["role_spans"]["subgoal"] == [[3, 6]]
    assert parsed["role_spans"]["action"] == [[6, 9]]
    assert parsed["value_positions"] == {"high": 0, "low": 6}

    pieces[1] = "KEEP"
    kept = parse_reference_hae_response(range(len(pieces)), _PieceTokenizer(pieces), turn_idx=2)["multi_turn"]
    assert kept["value_positions"]["high"] is None


@pytest.mark.unit
def test_reference_parser_fails_closed_on_duplicate_or_bad_switch_tags() -> None:
    pieces = ["<switch>", "MAYBE", "</switch>", "<subgoal>x</subgoal><action>y</action>"]
    parsed = parse_reference_hae_response(range(len(pieces)), _PieceTokenizer(pieces), turn_idx=0)["multi_turn"]
    assert parsed["format_valid"] is False
    assert parsed["switch"] is None


@pytest.mark.unit
def test_training_bridge_projects_semantic_actor_spans_and_sparse_targets(loss_module) -> None:
    values = torch.zeros(16, 2)
    values[4, 0] = 0.2
    values[12, 0] = 0.5
    values[0, 1] = 1.0
    values[8, 1] = 1.5
    rollout_data = {
        "values": [values],
        "metadata": [
            _metadata(
                _turn(0, 0, switch="KEEP", reward=1.0, done=False), _turn(1, 8, switch="SWITCH", reward=3.0, done=True)
            )
        ],
        "rollout_ids": [5],
        "total_lengths": [20],
        "response_lengths": [16],
        "loss_masks": [torch.ones(16)],
    }
    args = Namespace(kl_coef=0.0, gamma=1.0, lambd=0.0, hae_high_lambd=0.0, normalize_advantages=False)

    advantages, _ = loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(16)])

    expected = torch.zeros(16)
    expected[1:4] = 1.5
    expected[5:8] = 2.3
    expected[9:12] = 1.5
    expected[13:16] = 2.5
    torch.testing.assert_close(advantages[0], expected)
    torch.testing.assert_close(
        rollout_data["low_returns"][0], torch.tensor([0, 0, 0, 0, 2.5, 0, 0, 0, 0, 0, 0, 0, 3.0, 0, 0, 0.0])
    )
    torch.testing.assert_close(
        rollout_data["high_returns"][0], torch.tensor([2.5, 0, 0, 0, 0, 0, 0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0.0])
    )
    assert rollout_data["low_value_masks"][0].nonzero().flatten().tolist() == [4, 12]
    assert rollout_data["high_value_masks"][0].nonzero().flatten().tolist() == [0, 8]
    assert rollout_data["low_value_mask_sums"].tolist() == [2.0]
    assert rollout_data["high_value_mask_sums"].tolist() == [2.0]


@pytest.mark.unit
def test_compute_bridge_builds_scalar_zero_kl_for_two_head_values(loss_module) -> None:
    values = torch.zeros(8, 2)
    rollout_data = {
        "values": [values],
        "metadata": [_metadata(_turn(0, 0, switch="KEEP", reward=1.0, done=True))],
        "rollout_ids": [4],
        "total_lengths": [10],
        "response_lengths": [8],
        "loss_masks": [torch.ones(8)],
        "rewards": [1.0],
        "log_probs": None,
        "rollout_log_probs": None,
        "ref_log_probs": None,
    }
    args = Namespace(
        advantage_estimator="hae",
        use_rollout_logprobs=False,
        kl_coef=0.0,
        custom_advantage_function_path=None,
        gamma=1.0,
        lambd=1.0,
        hae_high_lambd=1.0,
        normalize_advantages=False,
        use_opd=False,
    )
    loss_module.compute_advantages_and_returns(args, rollout_data)
    assert rollout_data["kl"][0].shape == (8,)
    assert rollout_data["advantages"][0].shape == (8,)


@pytest.mark.unit
def test_training_bridge_rejects_truncated_or_non_boundary_high_value(loss_module) -> None:
    truncated = _turn(0, 0, switch="KEEP", reward=1.0, done=False)
    truncated["truncated"] = True
    rollout_data = {
        "values": [torch.zeros(8, 2)],
        "metadata": [_metadata(truncated)],
        "rollout_ids": [8],
        "total_lengths": [10],
        "response_lengths": [8],
        "loss_masks": [torch.ones(8)],
    }
    args = Namespace(kl_coef=0.0, gamma=1.0, lambd=1.0, hae_high_lambd=1.0, normalize_advantages=False)
    with pytest.raises(ValueError, match="truncated"):
        loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(8)])

    first = _turn(0, 0, switch="KEEP", reward=0.0, done=False)
    second = _turn(1, 8, switch="KEEP", reward=1.0, done=True)
    second["value_positions"]["high"] = 8
    rollout_data.update(
        values=[torch.zeros(16, 2)],
        metadata=[_metadata(first, second)],
        total_lengths=[20],
        response_lengths=[16],
        loss_masks=[torch.ones(16)],
    )
    with pytest.raises(ValueError, match="non-boundary"):
        loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(16)])


@pytest.mark.unit
def test_compacted_sibling_samples_reconstruct_one_hae_trajectory(loss_module) -> None:
    first = _turn(0, 0, switch="KEEP", reward=1.0, done=False)
    second = _turn(1, 0, switch="SWITCH", reward=3.0, done=True)
    first_values = torch.zeros(8, 2)
    second_values = torch.zeros(8, 2)
    first_values[4, 0], first_values[0, 1] = 0.2, 1.0
    second_values[4, 0], second_values[0, 1] = 0.5, 1.5
    rollout_data = {
        "values": [first_values, second_values],
        "metadata": [_metadata(first), _metadata(second)],
        "rollout_ids": [5, 5],
        "total_lengths": [10, 12],
        "response_lengths": [8, 8],
        "loss_masks": [torch.ones(8), torch.ones(8)],
    }
    args = Namespace(kl_coef=0.0, gamma=1.0, lambd=0.0, hae_high_lambd=0.0, normalize_advantages=False)

    advantages, _ = loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(8), torch.zeros(8)])

    torch.testing.assert_close(advantages[0][5:8], torch.full((3,), 2.3))
    torch.testing.assert_close(advantages[1][5:8], torch.full((3,), 2.5))
    torch.testing.assert_close(advantages[0][1:4], torch.full((3,), 1.5))
    torch.testing.assert_close(advantages[1][1:4], torch.full((3,), 1.5))
    assert rollout_data["low_value_mask_sums"].tolist() == [2.0, 2.0]
    assert rollout_data["high_value_mask_sums"].tolist() == [2.0, 2.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
