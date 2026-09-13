"""CPU contracts for M8.5 DTAP-specific hierarchy and HAE projection."""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from copy import deepcopy

import pytest
import torch

from examples.dtap_agent_rl.hierarchy import (
    DTAP_HAE_PROMPT,
    DtapHierarchyState,
    parse_dtap_hae_response,
    restore_dtap_hierarchy_state,
)
from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.utils.types import Sample

NUM_GPUS = 0


class _PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def batch_decode(self, rows, **_kwargs):
        return ["".join(self.pieces[index] for index in row) for row in rows]


def _pieces(switch: str, high: str, low: str, action: str) -> list[str]:
    return [
        "<switch>",
        switch,
        "</switch>",
        "<high_subgoal>",
        high,
        "</high_subgoal>",
        "<low_subgoal>",
        low,
        "</low_subgoal>",
        "<action>",
        action,
        "</action>",
    ]


def _parse(
    turn_idx: int,
    state: DtapHierarchyState,
    *,
    switch="SWITCH",
    high="map attack surface",
    low="inspect tools",
    action="get_attack_surface",
    feedback_ref=None,
):
    pieces = _pieces(switch, high, low, action)
    return parse_dtap_hae_response(
        range(len(pieces)),
        _PieceTokenizer(pieces),
        turn_idx=turn_idx,
        state=state,
        feedback_ref=feedback_ref,
    )


def _owned_turn(result, turn_idx: int, start: int, reward: float, done: bool) -> dict:
    annotation = deepcopy(result.metadata["multi_turn"])
    annotation["turn_idx"] = turn_idx
    annotation["response_span"] = [start, start + 12]
    annotation["reward"] = reward
    annotation["done"] = done
    annotation["truncated"] = False
    annotation["anchor_key"] = None
    annotation["role_spans"] = {
        role: [[span_start + start, span_end + start] for span_start, span_end in spans]
        for role, spans in annotation["role_spans"].items()
    }
    annotation["value_positions"] = {
        head: None if position is None else position + start
        for head, position in annotation["value_positions"].items()
    }
    return annotation


def _metadata(*turns: dict) -> dict:
    return {"multi_turn": {"version": 1, "context_revision": 0, "turns": list(turns)}}


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


@pytest.mark.unit
def test_prompt_explains_dtap_time_horizons_and_feedback_boundary() -> None:
    assert "first turn must use SWITCH" in DTAP_HAE_PROMPT
    assert "durable attack strategy" in DTAP_HAE_PROMPT
    assert "immediate objective achievable in this turn" in DTAP_HAE_PROMPT
    assert "current DTAP tool/schema contract" in DTAP_HAE_PROMPT
    assert "bounded next-turn evidence" in DTAP_HAE_PROMPT


@pytest.mark.unit
def test_parser_extracts_four_distinct_tag_inclusive_roles_and_state() -> None:
    result = _parse(0, DtapHierarchyState(), feedback_ref="feedback/attempt-0")

    assert result.error is None
    annotation = result.metadata["multi_turn"]
    assert annotation["switch"] == "SWITCH"
    assert annotation["role_spans"] == {
        "switch": [[0, 3]],
        "subgoal": [],
        "high_subgoal": [[3, 6]],
        "low_subgoal": [[6, 9]],
        "action": [[9, 12]],
    }
    assert annotation["value_positions"] == {"high": 0, "low": 9}
    assert annotation["hierarchy"]["option_id"] == "dtap-option-turn-0"
    assert annotation["hierarchy"]["previous_option_id"] is None
    assert annotation["hierarchy"]["feedback_ref"] == "feedback/attempt-0"
    assert result.state.active_high_subgoal == "map attack surface"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("pieces", "error"),
    [
        (["<switch>", "SWITCH", "</switch>"], "malformed_tags"),
        (_pieces("SWITCH", "goal", "inspect", "call") + ["<action>again</action>"], "malformed_tags"),
        (_pieces("MAYBE", "goal", "inspect", "call"), "invalid_field_content"),
        (_pieces("SWITCH", "goal", "same", "same"), "low_subgoal_matches_action"),
    ],
)
def test_parser_fails_closed_on_malformed_or_ambiguous_output(pieces, error) -> None:
    result = parse_dtap_hae_response(
        range(len(pieces)), _PieceTokenizer(pieces), turn_idx=0, state=DtapHierarchyState()
    )

    assert result.error == error
    assert result.metadata["multi_turn"]["format_valid"] is False
    assert result.state == DtapHierarchyState()


@pytest.mark.unit
def test_keep_preserves_option_identity_and_detects_exact_normalized_drift() -> None:
    first = _parse(0, DtapHierarchyState(), high="  durable\r\ngoal  ")
    kept = _parse(
        1, first.state, switch="KEEP", high="durable\ngoal", low="refine payload", action="validate_attack_step"
    )
    drifted = _parse(2, kept.state, switch="KEEP", high="different goal")

    assert kept.error is None
    assert kept.state.active_option_id == first.state.active_option_id
    assert kept.metadata["multi_turn"]["hierarchy"]["previous_option_id"] == first.state.active_option_id
    assert drifted.error == "keep_high_subgoal_drift"
    assert drifted.state == kept.state
    assert drifted.metadata["multi_turn"]["format_valid"] is False

    valid_turn = _owned_turn(kept, 1, 0, 0.0, False)
    invalid_turn = _owned_turn(drifted, 2, 12, 0.0, False)
    assert restore_dtap_hierarchy_state([valid_turn, invalid_turn]) == kept.state


@pytest.mark.unit
def test_switch_creates_one_new_stable_option_boundary() -> None:
    first = _parse(0, DtapHierarchyState())
    kept = _parse(1, first.state, switch="KEEP", low="choose payload", action="validate_attack_step")
    switched = _parse(
        2, kept.state, switch="SWITCH", high="place validated payload", low="apply payload", action="apply_attack_step"
    )

    assert switched.error is None
    assert switched.state.active_option_id == "dtap-option-turn-2"
    assert switched.metadata["multi_turn"]["hierarchy"]["previous_option_id"] == "dtap-option-turn-0"
    assert switched.metadata["multi_turn"]["value_positions"]["high"] == 0
    assert kept.metadata["multi_turn"]["value_positions"]["high"] is None


@pytest.mark.unit
def test_first_turn_keep_is_rejected() -> None:
    result = _parse(0, DtapHierarchyState(), switch="KEEP")
    assert result.error == "first_turn_must_switch"


@pytest.mark.unit
def test_compacted_siblings_restore_option_state_and_feedback_without_rewriting_prior_turn() -> None:
    manager = TrajectoryManager(fork_threshold_tokens=0)
    first = _parse(0, DtapHierarchyState())
    second = _parse(
        1,
        first.state,
        switch="KEEP",
        low="repair placement",
        action="validate_placement",
        feedback_ref="feedback/attempt-1",
    )
    first_metadata = deepcopy(first.metadata)
    second.metadata["multi_turn"]["context_revision"] = 2
    manager.record_turn(
        "dtap-hierarchy",
        turn=TurnRecord(prompt_ids=[100], output_ids=list(range(12)), finish_reason="stop"),
        prompt_messages=[{"role": "user", "content": "first"}],
        response_message={"role": "assistant", "content": "first response"},
        metadata=first.metadata,
    )
    manager.record_turn(
        "dtap-hierarchy",
        turn=TurnRecord(prompt_ids=[200], output_ids=list(range(12, 24)), finish_reason="stop"),
        prompt_messages=[{"role": "user", "content": "compacted summary"}],
        response_message={"role": "assistant", "content": "second response"},
        metadata=second.metadata,
    )

    samples = manager.get_trajectory("dtap-hierarchy", base_sample=Sample(index=7, rollout_id=19), reward=1.0)
    turns = sorted(
        [turn for sample in samples for turn in sample.train_metadata["multi_turn"]["turns"]],
        key=lambda turn: turn["turn_idx"],
    )

    assert len(samples) == 2
    assert turns[0]["hierarchy"] == first_metadata["multi_turn"]["hierarchy"]
    assert turns[0]["hierarchy"]["feedback_ref"] is None
    assert turns[1]["hierarchy"]["feedback_ref"] == "feedback/attempt-1"
    assert restore_dtap_hierarchy_state(turns) == second.state


def _dtap_rollout_data():
    first = _parse(0, DtapHierarchyState())
    second = _parse(
        1,
        first.state,
        switch="KEEP",
        low="refine payload",
        action="validate_attack_step",
    )
    values = torch.zeros(24, 2)
    values[9, 0] = 0.2
    values[21, 0] = 0.5
    values[0, 1] = 1.0
    return {
        "values": [values],
        "metadata": [_metadata(_owned_turn(first, 0, 0, 1.0, False), _owned_turn(second, 1, 12, 3.0, True))],
        "rollout_ids": [41],
        "total_lengths": [28],
        "response_lengths": [24],
        "loss_masks": [torch.ones(24)],
    }


@pytest.mark.unit
def test_dtap_credit_targets_low_subgoal_and_action_and_boundary_high_only(loss_module) -> None:
    rollout_data = _dtap_rollout_data()
    args = Namespace(
        kl_coef=0.0,
        gamma=1.0,
        lambd=0.0,
        hae_high_lambd=0.0,
        normalize_advantages=False,
        hae_policy_mode="dtap",
        hae_dtap_switch_credit=False,
    )

    advantages, _ = loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(24)])

    expected = torch.zeros(24)
    expected[3:6] = 3.0
    expected[6:9] = 1.3
    expected[9:12] = 1.3
    expected[18:21] = 2.5
    expected[21:24] = 2.5
    torch.testing.assert_close(advantages[0], expected)
    assert advantages[0][0:3].count_nonzero() == 0
    assert advantages[0][15:18].count_nonzero() == 0


@pytest.mark.unit
def test_dtap_switch_credit_requires_explicit_adaptation(loss_module) -> None:
    rollout_data = _dtap_rollout_data()
    args = Namespace(
        kl_coef=0.0,
        gamma=1.0,
        lambd=0.0,
        hae_high_lambd=0.0,
        normalize_advantages=False,
        hae_policy_mode="dtap",
        hae_dtap_switch_credit=True,
    )

    advantages, _ = loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(24)])

    torch.testing.assert_close(advantages[0][0:3], torch.full((3,), 3.0))
    assert advantages[0][12:15].count_nonzero() == 0


@pytest.mark.unit
def test_reference_and_dtap_modes_reject_each_others_metadata(loss_module) -> None:
    rollout_data = _dtap_rollout_data()
    args = Namespace(
        kl_coef=0.0,
        gamma=1.0,
        lambd=0.0,
        hae_high_lambd=0.0,
        normalize_advantages=False,
        hae_policy_mode="reference",
        hae_dtap_switch_credit=False,
    )

    with pytest.raises(ValueError, match="must not carry DTAP hierarchy"):
        loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(24)])


@pytest.mark.unit
def test_dtap_training_rejects_overlapping_semantic_spans(loss_module) -> None:
    rollout_data = _dtap_rollout_data()
    turn = rollout_data["metadata"][0]["multi_turn"]["turns"][0]
    turn["role_spans"]["low_subgoal"] = [[6, 10]]
    args = Namespace(
        kl_coef=0.0,
        gamma=1.0,
        lambd=0.0,
        hae_high_lambd=0.0,
        normalize_advantages=False,
        hae_policy_mode="dtap",
        hae_dtap_switch_credit=False,
    )

    with pytest.raises(ValueError, match="overlapping or out-of-order"):
        loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(24)])


@pytest.mark.unit
def test_dtap_training_rejects_reused_option_identity(loss_module) -> None:
    first = _parse(0, DtapHierarchyState())
    switched = _parse(
        1,
        first.state,
        switch="SWITCH",
        high="place validated payload",
        low="apply payload",
        action="apply_attack_step",
    )
    third = _parse(
        2,
        switched.state,
        switch="SWITCH",
        high="verify placement",
        low="read receipt",
        action="validate_placement",
    )
    turns = [
        _owned_turn(first, 0, 0, 0.0, False),
        _owned_turn(switched, 1, 12, 0.0, False),
        _owned_turn(third, 2, 24, 1.0, True),
    ]
    turns[2]["hierarchy"]["option_id"] = turns[0]["hierarchy"]["option_id"]
    values = torch.zeros(36, 2)
    rollout_data = {
        "values": [values],
        "metadata": [_metadata(*turns)],
        "rollout_ids": [42],
        "total_lengths": [40],
        "response_lengths": [36],
        "loss_masks": [torch.ones(36)],
    }
    args = Namespace(
        kl_coef=0.0,
        gamma=1.0,
        lambd=0.0,
        hae_high_lambd=0.0,
        normalize_advantages=False,
        hae_policy_mode="dtap",
        hae_dtap_switch_credit=False,
    )

    with pytest.raises(ValueError, match="must create a new option identity"):
        loss_module._compute_hae_advantages(args, rollout_data, [torch.zeros(36)])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
