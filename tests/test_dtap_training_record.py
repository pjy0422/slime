"""CPU tests for resumable DTAP rollout/training records."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# isort: off
from examples.dtap_agent_rl.training_record import (
    TrainingRecordContext,
    TrainingRecordStore,
    build_record,
    classify_eligibility,
    restore_samples,
    task_reference,
    validate_artifact_references,
)
from slime.utils.types import Sample

# isort: on

NUM_GPUS = 0


def context() -> TrainingRecordContext:
    return TrainingRecordContext(
        training_run_id="m8-cpu-contract",
        task_ref="finance/malicious/indirect/action_reversal/1",
        public_episode_id="m4-public",
        seed=17,
        tuning_trial_id="planned-runtime",
        runtime_setup_digest="sha256:runtime",
        feedback_mode="final_deterministic",
        hierarchy_mode="dtap",
    )


def training_sample(reward: float = 0.0) -> Sample:
    return Sample(
        group_index=2,
        index=3,
        rollout_id=4,
        session_id="session",
        tokens=[10, 11, 12],
        response_length=2,
        reward=reward,
        loss_mask=[1, 0],
        rollout_log_probs=[-0.1, 0.0],
        metadata={"sample_id": "sample", "task_dir": "/secret/path", "SECRET": "credential-canary"},
        train_metadata={"multi_turn": {"version": 1, "context_revision": 0, "turns": [], "dropped_turns": []}},
        status=Sample.Status.COMPLETED,
    )


def test_reward_zero_attack_miss_is_trainable_but_infrastructure_is_not() -> None:
    assert classify_eligibility("exhausted") == (True, "attack_outcome")
    assert classify_eligibility("infra_error") == (False, "infra_error")
    assert classify_eligibility("exhausted", "unsupported_placement") == (False, "unsupported_placement")


def test_atomic_record_round_trip_restores_trainable_sample_without_prompt_text(tmp_path) -> None:
    store = TrainingRecordStore(tmp_path / "records")
    sample = training_sample(0.0)
    record_id = store.record_id(context(), sample)
    record = build_record(
        record_id=record_id,
        context=context(),
        status="exhausted",
        failure_class=None,
        samples=[sample],
        reproduction={"sampling_seed": 17, "temperature": 1.0},
    )

    target = store.write(record)
    loaded = store.load(record_id)
    restored = restore_samples(loaded)

    assert target.parent.name == "eligible"
    assert restored[0].reward == 0.0
    assert restored[0].tokens == [10, 11, 12]
    assert restored[0].loss_mask == [1, 0]
    serialized = json.dumps(loaded)
    assert "/secret/path" not in serialized
    assert "credential-canary" not in serialized
    assert '"prompt"' not in serialized and '"response"' not in serialized
    assert '"session_id"' not in serialized


def test_excluded_record_is_observable_but_has_no_training_payload(tmp_path) -> None:
    store = TrainingRecordStore(tmp_path)
    sample = training_sample()
    record_id = store.record_id(context(), sample)
    record = build_record(
        record_id=record_id,
        context=context(),
        status="infra_error",
        failure_class="judge_result",
        samples=[sample],
    )
    path = store.write(record)
    loaded = store.load(record_id)

    assert path.parent.name == "excluded"
    assert loaded["samples"] == []
    assert loaded["outcome"]["eligible"] is False
    assert restore_samples(loaded) == []


def test_parallel_idempotent_writes_do_not_overwrite(tmp_path) -> None:
    store = TrainingRecordStore(tmp_path)
    sample = training_sample(1.0)
    record_id = store.record_id(context(), sample)
    record = build_record(
        record_id=record_id,
        context=context(),
        status="succeeded",
        failure_class=None,
        samples=[sample],
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: store.write(record), range(32)))
    assert len(set(paths)) == 1
    assert store.load(record_id)["payload_sha256"] == record["payload_sha256"]


def test_collision_and_corruption_fail_closed(tmp_path) -> None:
    store = TrainingRecordStore(tmp_path)
    sample = training_sample()
    record_id = store.record_id(context(), sample)
    record = build_record(
        record_id=record_id,
        context=context(),
        status="exhausted",
        failure_class=None,
        samples=[sample],
    )
    path = store.write(record)
    changed = build_record(
        record_id=record_id,
        context=context(),
        status="succeeded",
        failure_class=None,
        samples=[sample],
        reproduction={"sampling_seed": 99},
    )
    with pytest.raises(ValueError, match="collision"):
        store.write(changed)
    payload = json.loads(path.read_text())
    payload["outcome"]["status"] = "succeeded"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        store.load(record_id)


def test_task_reference_never_persists_an_absolute_host_path() -> None:
    value = task_reference({"task_dir": "/home/user/private/dataset/task"})
    assert value.startswith("task-sha256:")
    assert "/home/" not in value


def test_artifact_references_are_content_free_and_worker_scoped() -> None:
    value = validate_artifact_references(
        [
            {
                "kind": "mcp_trajectory",
                "ref": "workers/worker-7/episode/mcp.jsonl",
                "sha256": "c" * 64,
                "size_bytes": 17,
            }
        ],
        worker_id="worker-7",
    )
    assert value[0]["ref"] == "workers/worker-7/episode/mcp.jsonl"
    with pytest.raises(ValueError, match="namespace"):
        validate_artifact_references(
            [
                {
                    "kind": "mcp_trajectory",
                    "ref": "workers/worker-8/episode/mcp.jsonl",
                    "sha256": "c" * 64,
                    "size_bytes": 17,
                }
            ],
            worker_id="worker-7",
        )


def test_record_identity_survives_worker_replacement(tmp_path) -> None:
    store = TrainingRecordStore(tmp_path)
    sample = training_sample()
    first = replace(context(), worker_id="worker-1")
    replacement = replace(context(), worker_id="worker-9")

    assert store.record_id(first, sample) == store.record_id(replacement, sample)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
