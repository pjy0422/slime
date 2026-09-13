import sys
import types
from argparse import Namespace

import pytest
import torch


NUM_GPUS = 0


def test_get_values_does_not_apply_rollout_temperature(monkeypatch):
    import slime.backends.megatron_utils as megatron_utils

    missing = object()
    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
    previous_loss_attr = getattr(megatron_utils, "loss", missing)
    previous_cp_utils_attr = getattr(megatron_utils, "cp_utils", missing)
    for name in ("loss", "cp_utils"):
        if hasattr(megatron_utils, name):
            delattr(megatron_utils, name)

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from slime.backends.megatron_utils.loss import get_values

        args = Namespace(rollout_temperature=0.5, allgather_cp=False)
        logits = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32)
        tokens = [torch.tensor([10, 11, 12, 13], dtype=torch.long)]

        _, result = get_values(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=[4],
            response_lengths=[2],
        )

        torch.testing.assert_close(result["values"][0], torch.tensor([2.0, 3.0]))
    finally:
        if previous_loss is None:
            sys.modules.pop("slime.backends.megatron_utils.loss", None)
        else:
            sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils
        for name, previous in (("loss", previous_loss_attr), ("cp_utils", previous_cp_utils_attr)):
            if previous is missing:
                if hasattr(megatron_utils, name):
                    delattr(megatron_utils, name)
            else:
                setattr(megatron_utils, name, previous)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
