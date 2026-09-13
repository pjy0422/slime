"""Distributed CPU parity for reference and DTAP HAE projections."""

from __future__ import annotations

import json
import os
from argparse import Namespace

import _cp_dist_helpers
import pytest
from _cp_dist_helpers import free_port, stub_megatron_in_worker

NUM_GPUS = 0


def _metadata(policy_mode: str) -> dict:
    def turn(turn_idx, start, switch, reward, done):
        boundary = turn_idx == 0 or switch == "SWITCH"
        if policy_mode == "dtap":
            roles = {
                "switch": [[start, start + 1]],
                "subgoal": [],
                "high_subgoal": [[start + 1, start + 3]],
                "low_subgoal": [[start + 3, start + 5]],
                "action": [[start + 5, start + 8]],
            }
            hierarchy = {
                "version": 1,
                "policy_mode": "dtap",
                "option_id": f"option-{turn_idx}",
                "previous_option_id": None if turn_idx == 0 else "option-0",
                "high_subgoal": f"goal-{turn_idx}",
                "low_subgoal": f"step-{turn_idx}",
                "feedback_ref": None,
            }
        else:
            roles = {
                "switch": [[start, start + 1]],
                "subgoal": [[start + 1, start + 4]],
                "high_subgoal": [[start + 1, start + 4]],
                "low_subgoal": [],
                "action": [[start + 5, start + 8]],
            }
            hierarchy = None
        return {
            "turn_idx": turn_idx,
            "response_span": [start, start + 8],
            "reward": reward,
            "done": done,
            "truncated": False,
            "anchor_key": None,
            "switch": switch,
            "role_spans": roles,
            "value_positions": {
                "high": start if boundary else None,
                "low": start + (5 if policy_mode == "dtap" else 4),
            },
            "format_valid": True,
            "hierarchy": hierarchy,
        }

    return {
        "multi_turn": {
            "version": 1,
            "context_revision": 0,
            "turns": [
                turn(0, 0, "SWITCH", 1.0, False),
                turn(1, 8, "SWITCH", 3.0, True),
            ],
        }
    }


def _worker(rank: int, cp_size: int, master_port: int, result_path: str, policy_mode: str) -> None:
    import torch
    import torch.distributed as dist

    stub_megatron_in_worker(cp_size, rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=cp_size)
    try:
        from megatron.core import mpu

        cp_group = dist.new_group(ranks=list(range(cp_size)))
        dp_groups = [dist.new_group(ranks=[cp_rank]) for cp_rank in range(cp_size)]
        mpu.get_context_parallel_group = lambda: cp_group
        mpu.get_data_parallel_group = lambda with_context_parallel=False, **_kwargs: (
            cp_group if with_context_parallel else dp_groups[rank]
        )
        mpu.is_pipeline_last_stage = lambda: True

        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp, slice_log_prob_with_cp
        from slime.backends.megatron_utils.loss import _compute_hae_advantages

        total_length = 24
        response_length = 16
        full_values = torch.zeros(response_length, 2)
        low_offset = 5 if policy_mode == "dtap" else 4
        full_values[low_offset, 0] = 0.2
        full_values[8 + low_offset, 0] = 0.5
        full_values[0, 1] = 1.0
        full_values[8, 1] = 1.5
        full_kl = torch.linspace(0.01, 0.16, response_length)
        local_values = slice_log_prob_with_cp(full_values, total_length, response_length)
        local_kl = slice_log_prob_with_cp(full_kl, total_length, response_length)
        rollout_data = {
            "values": [local_values],
            "metadata": [_metadata(policy_mode)],
            "rollout_ids": [31],
            "total_lengths": [total_length],
            "response_lengths": [response_length],
            "loss_masks": [torch.ones(response_length)],
        }
        args = Namespace(
            kl_coef=0.2,
            gamma=0.9,
            lambd=0.8,
            hae_high_lambd=0.7,
            normalize_advantages=True,
            hae_policy_mode=policy_mode,
            hae_dtap_switch_credit=False,
        )
        advantages, _ = _compute_hae_advantages(args, rollout_data, [local_kl])
        reconstructed = {
            "advantages": all_gather_with_cp(advantages[0], total_length, response_length).tolist(),
            "low_returns": all_gather_with_cp(rollout_data["low_returns"][0], total_length, response_length).tolist(),
            "high_returns": all_gather_with_cp(
                rollout_data["high_returns"][0], total_length, response_length
            ).tolist(),
            "low_masks": rollout_data["low_value_masks"][0].tolist(),
            "high_masks": rollout_data["high_value_masks"][0].tolist(),
            "low_sums": rollout_data["low_value_mask_sums"].tolist(),
            "high_sums": rollout_data["high_value_mask_sums"].tolist(),
        }
        if rank == 0:
            with open(result_path, "w") as output:
                json.dump(reconstructed, output)
    finally:
        dist.destroy_process_group()


def _run(cp_size: int, tmp_path, policy_mode: str) -> dict:
    import torch.multiprocessing as mp

    result_path = str(tmp_path / f"hae_{policy_mode}_cp{cp_size}.json")
    mp.spawn(_worker, args=(cp_size, free_port(), result_path, policy_mode), nprocs=cp_size, join=True)
    with open(result_path) as data:
        return json.load(data)


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("policy_mode", ["reference", "dtap"])
def test_hae_projection_is_context_parallel_invariant(cp_size, policy_mode, tmp_path) -> None:
    baseline = _run(1, tmp_path, policy_mode)
    actual = _run(cp_size, tmp_path, policy_mode)
    for field in ("advantages", "low_returns", "high_returns"):
        assert actual[field] == pytest.approx(baseline[field], abs=1e-6)
    assert actual["low_masks"] == baseline["low_masks"]
    assert actual["high_masks"] == baseline["high_masks"]
    assert actual["low_sums"] == baseline["low_sums"] == [2.0]
    assert actual["high_sums"] == baseline["high_sums"] == [2.0]


_ = _cp_dist_helpers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
