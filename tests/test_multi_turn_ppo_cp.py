"""Distributed CPU parity test for M8.1 context parallel projection."""

from __future__ import annotations

import json
import os
from argparse import Namespace

import _cp_dist_helpers
import pytest
from _cp_dist_helpers import free_port, stub_megatron_in_worker

NUM_GPUS = 0


def _metadata() -> dict:
    def turn(turn_idx, span, reward, done):
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

    return {
        "multi_turn": {
            "version": 1,
            "context_revision": 0,
            "turns": [turn(0, [0, 2], 0.25, False), turn(1, [5, 10], 1.5, True)],
        }
    }


def _worker(rank: int, cp_size: int, master_port: int, result_path: str) -> None:
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

        from slime.backends.megatron_utils.cp_utils import (
            all_gather_with_cp,
            get_sum_of_sample_mean,
            slice_log_prob_with_cp,
        )
        from slime.backends.megatron_utils.loss import _compute_multi_turn_ppo_advantages

        total_length = 16
        response_length = 10
        full_values = torch.tensor([0.2, 8.0, 8.0, 8.0, 8.0, 0.4, 8.0, 8.0, 8.0, 8.0])
        full_kl = torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7])
        local_values = slice_log_prob_with_cp(full_values, total_length, response_length)
        local_kl = slice_log_prob_with_cp(full_kl, total_length, response_length)
        rollout_data = {
            "values": [local_values],
            "metadata": [_metadata()],
            "rollout_ids": [12],
            "total_lengths": [total_length],
            "response_lengths": [response_length],
            "loss_masks": [torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])],
        }
        args = Namespace(kl_coef=0.2, gamma=0.95, lambd=0.8, normalize_advantages=True)

        advantages, returns = _compute_multi_turn_ppo_advantages(args, rollout_data, [local_kl])
        full_advantages = all_gather_with_cp(advantages[0], total_length, response_length)
        full_returns = all_gather_with_cp(returns[0], total_length, response_length)
        local_value_losses = slice_log_prob_with_cp(
            torch.arange(response_length, dtype=torch.float32), total_length, response_length
        )
        value_reducer = get_sum_of_sample_mean(
            [total_length],
            [response_length],
            rollout_data["value_masks"],
            rollout_data["value_mask_sums"],
        )
        sparse_value_loss = value_reducer(local_value_losses)
        dist.all_reduce(sparse_value_loss, group=cp_group)
        if rank == 0:
            with open(result_path, "w") as output:
                json.dump(
                    {
                        "advantages": full_advantages.tolist(),
                        "returns": full_returns.tolist(),
                        "value_masks": rollout_data["value_masks"][0].tolist(),
                        "value_mask_sums": rollout_data["value_mask_sums"].tolist(),
                        "sparse_value_loss": sparse_value_loss.item(),
                    },
                    output,
                )
    finally:
        dist.destroy_process_group()


def _run(cp_size: int, tmp_path) -> dict:
    import torch.multiprocessing as mp

    result_path = str(tmp_path / f"multi_turn_ppo_cp{cp_size}.json")
    mp.spawn(_worker, args=(cp_size, free_port(), result_path), nprocs=cp_size, join=True)
    with open(result_path) as data:
        return json.load(data)


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_multi_turn_ppo_targets_are_context_parallel_invariant(cp_size, tmp_path) -> None:
    baseline = _run(1, tmp_path)
    actual = _run(cp_size, tmp_path)

    assert actual["advantages"] == pytest.approx(baseline["advantages"], abs=1e-6)
    assert actual["returns"] == pytest.approx(baseline["returns"], abs=1e-6)
    assert actual["value_masks"] == baseline["value_masks"]
    assert actual["value_mask_sums"] == baseline["value_mask_sums"] == [2.0]
    assert actual["sparse_value_loss"] == pytest.approx(baseline["sparse_value_loss"], abs=1e-6)


_ = _cp_dist_helpers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
