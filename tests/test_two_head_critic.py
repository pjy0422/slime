"""CPU contracts for M8.4a two-head critic infrastructure."""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch

NUM_GPUS = 0


@pytest.fixture
def loss_module(monkeypatch):
    import slime.backends.megatron_utils as megatron_utils

    missing = object()
    module_names = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    original_modules = {name: sys.modules.get(name, missing) for name in module_names}
    original_attributes = {name: getattr(megatron_utils, name, missing) for name in ("loss", "cp_utils")}
    for name in module_names:
        sys.modules.pop(name, None)
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


def _args(**overrides) -> Namespace:
    values = {
        "allgather_cp": False,
        "critic_value_heads": 2,
        "critic_high_value_loss_coef": 0.5,
        "value_clip": 1000.0,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.unit
def test_get_values_preserves_two_head_axis(loss_module) -> None:
    logits = torch.tensor([[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]])

    _, result = loss_module.get_values(
        logits,
        args=_args(),
        unconcat_tokens=[torch.arange(5)],
        total_lengths=[5],
        response_lengths=[3],
    )

    assert result["values"][0].shape == (3, 2)
    torch.testing.assert_close(result["values"][0], torch.tensor([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]]))


@pytest.mark.unit
def test_allgather_redistribution_preserves_trailing_head_axis(monkeypatch, loss_module) -> None:
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(loss_module.dist.nn, "all_reduce", lambda tensor, group=None: tensor)
    monkeypatch.setattr(loss_module, "slice_log_prob_with_cp", lambda tensor, *_args: tensor)
    result = {"values": [torch.tensor([[1.0, 10.0], [2.0, 20.0]])]}

    loss_module._allgather_cp_redistribute(
        result,
        logits_local_len=3,
        total_lengths=[6],
        response_lengths=[4],
    )

    assert result["values"][0].shape == (4, 2)
    torch.testing.assert_close(
        result["values"][0],
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [0.0, 0.0], [0.0, 0.0]]),
    )


@pytest.mark.unit
def test_two_head_value_loss_uses_independent_sparse_reducers(loss_module) -> None:
    logits = torch.zeros(1, 5, 2, requires_grad=True)
    batch = {
        "values": [torch.zeros(4, 2)],
        "low_returns": [torch.tensor([1.0, 999.0, 3.0, 999.0])],
        "high_returns": [torch.tensor([999.0, 5.0, 999.0, 999.0])],
        "unconcat_tokens": [torch.arange(5)],
        "total_lengths": [5],
        "response_lengths": [4],
    }

    def low_reducer(values):
        return values[[0, 2]].mean()

    def high_reducer(values):
        return values[[1]].mean()

    value_loss, metrics = loss_module.value_loss_function(_args(), batch, logits, (low_reducer, high_reducer))

    # low=(1^2 + 3^2)/2=5, high=5^2=25, combined=5 + 0.5*25.
    torch.testing.assert_close(value_loss, torch.tensor(17.5))
    torch.testing.assert_close(metrics["low_value_loss"], torch.tensor(5.0))
    torch.testing.assert_close(metrics["high_value_loss"], torch.tensor(25.0))
    value_loss.backward()
    assert logits.grad is not None


@pytest.mark.unit
def test_two_head_loss_function_builds_separate_mask_reducers(monkeypatch, loss_module) -> None:
    observed = {}

    def fake_value_loss(_args, _batch, _logits, reducers):
        observed["low"] = reducers[0](torch.tensor([2.0, 100.0, 6.0, 100.0]))
        observed["high"] = reducers[1](torch.tensor([100.0, 9.0, 100.0, 100.0]))
        total = observed["low"] + observed["high"]
        return total, {"value_loss": total.detach()}

    monkeypatch.setattr(loss_module, "value_loss_function", fake_value_loss)
    args = _args(
        loss_type="value_loss",
        advantage_estimator="hae",
        calculate_per_token_loss=False,
        custom_loss_function_path=None,
        recompute_loss_function=False,
    )
    batch = {
        "loss_masks": [torch.ones(4)],
        "rollout_mask_sums": [torch.tensor(4.0)],
        "low_value_masks": [torch.tensor([1.0, 0.0, 1.0, 0.0])],
        "high_value_masks": [torch.tensor([0.0, 1.0, 0.0, 0.0])],
        "low_value_mask_sums": [torch.tensor(2.0)],
        "high_value_mask_sums": [torch.tensor(1.0)],
        "total_lengths": [5],
        "response_lengths": [4],
    }

    reduced, _, _ = loss_module.loss_function(args, batch, 1, 1, torch.zeros(1, 5, 2))

    torch.testing.assert_close(observed["low"], torch.tensor(4.0))
    torch.testing.assert_close(observed["high"], torch.tensor(9.0))
    torch.testing.assert_close(reduced, torch.tensor(13.0))


@pytest.mark.unit
def test_two_head_value_loss_ignores_shared_actor_token_denominator(monkeypatch, loss_module) -> None:
    def fake_value_loss(_args, _batch, _logits, reducers):
        low = reducers[0](torch.tensor([2.0, 100.0, 6.0, 100.0]))
        high = reducers[1](torch.tensor([100.0, 9.0, 100.0, 100.0]))
        total = low + high
        return total, {"value_loss": total.detach()}

    monkeypatch.setattr(loss_module, "value_loss_function", fake_value_loss)
    args = _args(
        loss_type="value_loss",
        advantage_estimator="hae",
        calculate_per_token_loss=True,
        custom_loss_function_path=None,
        recompute_loss_function=False,
    )
    batch = {
        "loss_masks": [torch.ones(4)],
        "rollout_mask_sums": [torch.tensor(4.0)],
        "low_value_masks": [torch.tensor([1.0, 0.0, 1.0, 0.0])],
        "high_value_masks": [torch.tensor([0.0, 1.0, 0.0, 0.0])],
        "low_value_mask_sums": [torch.tensor(2.0)],
        "high_value_mask_sums": [torch.tensor(1.0)],
        "total_lengths": [5],
        "response_lengths": [4],
    }

    reduced, normalizer, _ = loss_module.loss_function(args, batch, 1, 1, torch.zeros(1, 5, 2))

    # Each head retains its own mean: low=4, high=9. A shared denominator
    # would incorrectly produce (2+6+9)/(2+1).
    torch.testing.assert_close(reduced, torch.tensor(13.0))
    assert normalizer.item() == 1


@pytest.mark.unit
def test_two_head_loss_rejects_missing_high_mask(loss_module) -> None:
    args = _args(
        loss_type="value_loss",
        advantage_estimator="hae",
        calculate_per_token_loss=False,
        custom_loss_function_path=None,
        recompute_loss_function=False,
    )
    batch = {
        "loss_masks": [torch.ones(2)],
        "rollout_mask_sums": [torch.tensor(2.0)],
        "low_value_masks": [torch.ones(2)],
        "low_value_mask_sums": [torch.tensor(2.0)],
        "total_lengths": [3],
        "response_lengths": [2],
    }

    with pytest.raises(ValueError, match="high value masks"):
        loss_module.loss_function(args, batch, 1, 1, torch.zeros(1, 3, 2))


@pytest.mark.unit
def test_two_head_value_loss_handles_empty_high_target_set(loss_module) -> None:
    args = _args(
        loss_type="value_loss",
        advantage_estimator="hae",
        calculate_per_token_loss=False,
        custom_loss_function_path=None,
        recompute_loss_function=False,
    )
    batch = {
        "values": [torch.zeros(2, 2)],
        "low_returns": [torch.tensor([1.0, 100.0])],
        "high_returns": [torch.tensor([100.0, 100.0])],
        "unconcat_tokens": [torch.arange(3)],
        "loss_masks": [torch.ones(2)],
        "rollout_mask_sums": [torch.tensor(2.0)],
        "low_value_masks": [torch.tensor([1.0, 0.0])],
        "high_value_masks": [torch.zeros(2)],
        "low_value_mask_sums": [torch.tensor(1.0)],
        "high_value_mask_sums": [torch.tensor(0.0)],
        "total_lengths": [3],
        "response_lengths": [2],
    }

    value_loss, _, metrics = loss_module.loss_function(args, batch, 1, 1, torch.zeros(1, 3, 2))

    torch.testing.assert_close(value_loss, torch.tensor(1.0))
    high_loss_index = metrics["keys"].index("high_value_loss") + 1
    torch.testing.assert_close(metrics["values"][high_loss_index], torch.tensor(0.0))
    assert torch.isfinite(value_loss)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
