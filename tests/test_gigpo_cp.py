"""Distributed CPU parity test for M8.3 GiGPO turn projection."""

from __future__ import annotations

import json
import os

import _cp_dist_helpers
import pytest
from _cp_dist_helpers import free_port, stub_megatron_in_worker

NUM_GPUS = 0


def _metadata() -> dict:
    def turn(turn_idx, span, reward, anchor_key, done):
        return {
            "turn_idx": turn_idx,
            "response_span": span,
            "reward": reward,
            "done": done,
            "truncated": False,
            "anchor_key": anchor_key,
            "switch": None,
            "role_spans": {"switch": [], "subgoal": [], "high_subgoal": [], "low_subgoal": [], "action": []},
            "value_positions": {"high": None, "low": None},
            "format_valid": True,
        }

    return {
        "multi_turn": {
            "version": 1,
            "context_revision": 0,
            "turns": [turn(0, [0, 2], 0.0, "A", False), turn(1, [5, 10], 1.0, "B", True)],
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
        mpu.get_context_parallel_group = lambda: cp_group
        mpu.is_pipeline_last_stage = lambda: True

        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp
        from slime.backends.megatron_utils.loss import _compute_precomputed_turn_advantages

        total_length = 16
        response_length = 10
        rollout_data = {
            "metadata": [_metadata()],
            "rollout_ids": [12],
            "turn_credits": [[0.95, 0.5]],
            "total_lengths": [total_length],
            "response_lengths": [response_length],
            "loss_masks": [torch.tensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 1])],
        }

        advantages, returns = _compute_precomputed_turn_advantages(rollout_data, estimator_name="gigpo")
        full_advantages = all_gather_with_cp(advantages[0], total_length, response_length)
        full_returns = all_gather_with_cp(returns[0], total_length, response_length)
        if rank == 0:
            with open(result_path, "w") as output:
                json.dump({"advantages": full_advantages.tolist(), "returns": full_returns.tolist()}, output)
    finally:
        dist.destroy_process_group()


def _run(cp_size: int, tmp_path) -> dict:
    import torch.multiprocessing as mp

    result_path = str(tmp_path / f"gigpo_cp{cp_size}.json")
    mp.spawn(_worker, args=(cp_size, free_port(), result_path), nprocs=cp_size, join=True)
    with open(result_path) as data:
        return json.load(data)


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_gigpo_projection_is_context_parallel_invariant(cp_size, tmp_path) -> None:
    baseline = _run(1, tmp_path)
    actual = _run(cp_size, tmp_path)

    assert actual["advantages"] == pytest.approx(baseline["advantages"], abs=1e-6)
    assert actual["returns"] == pytest.approx(baseline["returns"], abs=1e-6)
    assert actual["advantages"] == pytest.approx([0.95, 0.95, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])


_ = _cp_dist_helpers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
