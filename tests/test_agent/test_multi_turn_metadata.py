"""CPU boundary tests for logical-turn trajectory metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

NUM_GPUS = 0

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.agent.trajectory import MULTI_TURN_METADATA_VERSION, TrajectoryManager, TurnRecord
from slime.utils.types import Sample


def _record(
    manager: TrajectoryManager,
    sid: str,
    *,
    prompt_ids: list[int],
    output_ids: list[int],
    prompt_messages: list[dict],
    response_content: str,
    finish_reason: str = "stop",
    metadata: dict | None = None,
) -> None:
    manager.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            finish_reason=finish_reason,
            output_log_probs=[-0.1] * len(output_ids),
        ),
        prompt_messages=prompt_messages,
        response_message={"role": "assistant", "content": response_content},
        metadata=metadata,
    )


def _turns(samples: list[Sample]) -> list[dict]:
    return sorted(
        [turn for sample in samples for turn in sample.train_metadata["multi_turn"]["turns"]],
        key=lambda turn: turn["turn_idx"],
    )


def test_clean_turns_emit_response_relative_metadata_and_terminal_reward() -> None:
    manager = TrajectoryManager()
    user = {"role": "user", "content": "u"}
    assistant = {"role": "assistant", "content": "a0"}
    tool = {"role": "tool", "content": "result"}

    _record(
        manager,
        "clean",
        prompt_ids=[10, 11],
        output_ids=[20, 21, 22],
        prompt_messages=[user],
        response_content="a0",
        metadata={
            "multi_turn": {
                "anchor_key": "state-0",
                "switch": "KEEP",
                "role_spans": {"switch": [[0, 1]], "action": [[1, 3]]},
                "value_positions": {"high": 0, "low": 1},
            }
        },
    )
    _record(
        manager,
        "clean",
        prompt_ids=[10, 11, 20, 21, 22, 30],
        output_ids=[40, 41],
        prompt_messages=[user, assistant, tool],
        response_content="a1",
    )

    samples = manager.get_trajectory(
        "clean",
        base_sample=Sample(index=7, group_index=3, prompt="", train_metadata={"source": "fixture"}),
        reward=2.5,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.reward == 2.5  # legacy sample-level reward remains unchanged
    assert sample.train_metadata["source"] == "fixture"
    assert sample.train_metadata["multi_turn"]["version"] == MULTI_TURN_METADATA_VERSION
    assert sample.train_metadata["multi_turn"]["context_revision"] == 0
    turns = _turns(samples)
    assert [turn["turn_idx"] for turn in turns] == [0, 1]
    assert [turn["response_span"] for turn in turns] == [[0, 3], [4, 6]]
    assert turns[0]["role_spans"]["switch"] == [[0, 1]]
    assert turns[0]["role_spans"]["action"] == [[1, 3]]
    assert turns[0]["value_positions"] == {"high": 0, "low": 1}
    assert [turn["reward"] for turn in turns] == [0.0, 2.5]
    assert [turn["done"] for turn in turns] == [False, True]


def test_explicit_rewards_are_preserved_and_partial_annotations_fail() -> None:
    user = {"role": "user", "content": "u"}
    assistant = {"role": "assistant", "content": "a0"}
    tool = {"role": "tool", "content": "result"}

    manager = TrajectoryManager()
    _record(
        manager,
        "explicit",
        prompt_ids=[1],
        output_ids=[2],
        prompt_messages=[user],
        response_content="a0",
        metadata={"multi_turn": {"reward": 0.25}},
    )
    _record(
        manager,
        "explicit",
        prompt_ids=[1, 2, 3],
        output_ids=[4],
        prompt_messages=[user, assistant, tool],
        response_content="a1",
        metadata={"multi_turn": {"reward": 0.75, "done": True}},
    )
    explicit = manager.get_trajectory("explicit", base_sample=Sample(index=1), reward=99.0)
    assert [turn["reward"] for turn in _turns(explicit)] == [0.25, 0.75]
    assert [turn["done"] for turn in _turns(explicit)] == [False, True]

    manager = TrajectoryManager()
    _record(
        manager,
        "partial",
        prompt_ids=[1],
        output_ids=[2],
        prompt_messages=[user],
        response_content="a0",
        metadata={"multi_turn": {"reward": 0.25}},
    )
    _record(
        manager,
        "partial",
        prompt_ids=[1, 2, 3],
        output_ids=[4],
        prompt_messages=[user, assistant, tool],
        response_content="a1",
    )
    with pytest.raises(ValueError, match="explicit for every owned turn"):
        manager.get_trajectory("partial", base_sample=Sample(index=2), reward=1.0)
    assert manager.has_session("partial") is False


def test_fork_and_sibling_paths_have_exactly_one_owner_per_emitted_turn() -> None:
    manager = TrajectoryManager(fork_threshold_tokens=1)
    user = {"role": "user", "content": "u"}
    assistant = {"role": "assistant", "content": "call"}
    tool_a = {"role": "tool", "content": "a"}
    tool_b = {"role": "tool", "content": "b"}

    _record(
        manager,
        "siblings",
        prompt_ids=[1, 2],
        output_ids=[3],
        prompt_messages=[user],
        response_content="call",
        finish_reason="tool_calls",
    )
    _record(
        manager,
        "siblings",
        prompt_ids=[99, 3, 4],
        output_ids=[5],
        prompt_messages=[user, assistant, tool_a],
        response_content="a",
    )
    _record(
        manager,
        "siblings",
        prompt_ids=[98, 3, 6],
        output_ids=[7],
        prompt_messages=[user, assistant, tool_b],
        response_content="b",
    )

    samples = manager.get_trajectory("siblings", base_sample=Sample(index=8), reward=1.0)
    assert len(samples) == 3
    assert {sample.rollout_id for sample in samples} == {8}
    identities = [(sample.rollout_id, turn["turn_idx"]) for sample in samples for turn in _turns([sample])]
    assert sorted(identities) == [(8, 0), (8, 1), (8, 2)]
    assert len(identities) == len(set(identities))


def test_realign_masks_rewritten_response_and_tracks_context_revision() -> None:
    manager = TrajectoryManager()
    user = {"role": "user", "content": "u"}
    assistant_0 = {"role": "assistant", "content": "a0"}
    assistant_1 = {"role": "assistant", "content": "a1"}
    tool_0 = {"role": "tool", "content": "t0"}
    tool_1 = {"role": "tool", "content": "t1"}

    _record(
        manager,
        "realign",
        prompt_ids=[10, 11],
        output_ids=[20, 21],
        prompt_messages=[user],
        response_content="a0",
    )
    _record(
        manager,
        "realign",
        prompt_ids=[10, 11, 20, 99, 30],
        output_ids=[40, 41],
        prompt_messages=[user, assistant_0, tool_0],
        response_content="a1",
    )
    _record(
        manager,
        "realign",
        prompt_ids=[10, 11, 20, 99, 30, 40, 98, 50],
        output_ids=[60],
        prompt_messages=[user, assistant_0, tool_0, assistant_1, tool_1],
        response_content="a2",
    )

    samples = manager.get_trajectory("realign", base_sample=Sample(index=4), reward=3.0)
    assert len(samples) == 1
    sample = samples[0]
    multi_turn = sample.train_metadata["multi_turn"]
    assert multi_turn["context_revision"] == 2
    assert [turn["turn_idx"] for turn in multi_turn["turns"]] == [2]
    assert [item["turn_idx"] for item in multi_turn["dropped_turns"]] == [0, 1]
    assert {item["reason"] for item in multi_turn["dropped_turns"]} == {"realigned_context"}
    response_start = len(sample.tokens) - sample.response_length
    assert sample.loss_mask == [0] * (8 - response_start) + [1]


def test_partial_generated_turn_is_not_owned_or_rewarded() -> None:
    manager = TrajectoryManager()
    user = {"role": "user", "content": "u"}
    assistant = {"role": "assistant", "content": "a0"}
    tool = {"role": "tool", "content": "t"}
    _record(
        manager,
        "partial",
        prompt_ids=[1],
        output_ids=[2, 3],
        prompt_messages=[user],
        response_content="a0",
    )
    _record(
        manager,
        "partial",
        prompt_ids=[1, 2, 3, 4],
        output_ids=[5, 6],
        prompt_messages=[user, assistant, tool],
        response_content="partial",
        finish_reason="length",
    )

    samples = manager.get_trajectory("partial", base_sample=Sample(index=5), reward=9.0)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.status is Sample.Status.TRUNCATED
    assert sample.loss_mask == [1, 1, 0, 0, 0]
    multi_turn = sample.train_metadata["multi_turn"]
    assert [turn["turn_idx"] for turn in multi_turn["turns"]] == [0]
    assert multi_turn["turns"][0]["reward"] == 0.0
    assert multi_turn["turns"][0]["truncated"] is True
    assert multi_turn["turns"][0]["done"] is False
    assert multi_turn["dropped_turns"] == [{"turn_idx": 1, "reason": "partial_turn", "generated_tokens": 2}]


def test_assistant_rewrite_keeps_diagnostic_without_recovering_from_prompt_text() -> None:
    manager = TrajectoryManager()
    user = {"role": "user", "content": "u"}
    original_assistant = {"role": "assistant", "content": "original"}
    rewritten_assistant = {"role": "assistant", "content": "rewritten"}
    tool = {"role": "tool", "content": "t"}
    _record(
        manager,
        "rewrite",
        prompt_ids=[1],
        output_ids=[2, 3],
        prompt_messages=[user],
        response_content=original_assistant["content"],
    )
    _record(
        manager,
        "rewrite",
        prompt_ids=[1, 20, 30],
        output_ids=[4],
        prompt_messages=[user, rewritten_assistant, tool],
        response_content="next",
    )

    samples = manager.get_trajectory("rewrite", base_sample=Sample(index=9), reward=1.0)
    assert len(samples) == 1
    multi_turn = samples[0].train_metadata["multi_turn"]
    assert [turn["turn_idx"] for turn in multi_turn["turns"]] == [1]
    assert multi_turn["dropped_turns"] == [{"turn_idx": 0, "reason": "assistant_rewrite", "generated_tokens": 2}]


def test_only_partial_turn_produces_no_training_sample() -> None:
    manager = TrajectoryManager()
    _record(
        manager,
        "only-partial",
        prompt_ids=[1],
        output_ids=[2, 3],
        prompt_messages=[{"role": "user", "content": "u"}],
        response_content="partial",
        finish_reason="length",
    )
    assert manager.get_trajectory("only-partial", base_sample=Sample(index=10), reward=1.0) == []


def test_max_token_cutoff_removes_partial_turn_ownership() -> None:
    manager = TrajectoryManager()
    user = {"role": "user", "content": "u"}
    assistant = {"role": "assistant", "content": "a0"}
    tool = {"role": "tool", "content": "t"}
    _record(
        manager,
        "cutoff",
        prompt_ids=[10, 11],
        output_ids=[20, 21],
        prompt_messages=[user],
        response_content="a0",
    )
    _record(
        manager,
        "cutoff",
        prompt_ids=[10, 11, 20, 21, 30],
        output_ids=[40, 41, 42],
        prompt_messages=[user, assistant, tool],
        response_content="a1",
    )

    samples = manager.get_trajectory(
        "cutoff",
        base_sample=Sample(index=6),
        reward=4.0,
        max_sample_tokens=7,
    )
    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [10, 11, 20, 21, 30, 40, 41]
    assert sample.loss_mask == [1, 1, 0, 0, 0]
    turn = sample.train_metadata["multi_turn"]["turns"][0]
    assert turn["turn_idx"] == 0
    assert turn["truncated"] is True
    assert turn["reward"] == 0.0
    assert sample.train_metadata["multi_turn"]["dropped_turns"] == [
        {"turn_idx": 1, "reason": "max_sample_tokens", "generated_tokens": 2}
    ]


@pytest.mark.parametrize(
    ("multi_turn", "match"),
    [
        ({"version": 2}, "unsupported"),
        ({"reward": float("nan")}, "finite"),
        ({"role_spans": {"action": [[0, 2]]}}, "outside"),
        ({"value_positions": {"low": 1}}, "outside"),
        ({"unknown": True}, "unknown fields"),
    ],
)
def test_turn_annotation_validation(multi_turn: dict, match: str) -> None:
    manager = TrajectoryManager()
    with pytest.raises(ValueError, match=match):
        _record(
            manager,
            "invalid",
            prompt_ids=[1],
            output_ids=[2],
            prompt_messages=[{"role": "user", "content": "u"}],
            response_content="a",
            metadata={"multi_turn": multi_turn},
        )


def test_reserved_base_metadata_namespace_is_rejected() -> None:
    manager = TrajectoryManager()
    _record(
        manager,
        "reserved",
        prompt_ids=[1],
        output_ids=[2],
        prompt_messages=[{"role": "user", "content": "u"}],
        response_content="a",
    )
    with pytest.raises(ValueError, match="reserved"):
        manager.get_trajectory(
            "reserved",
            base_sample=Sample(index=1, train_metadata={"multi_turn": {"version": 0}}),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
