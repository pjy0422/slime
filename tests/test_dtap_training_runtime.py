"""CPU-only contract tests for the pinned native training runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# isort: off
from examples.dtap_agent_rl.scripts.training_runtime import (
    dependency_probe,
    docker_commands,
    load_manifest,
    prepare,
    runtime_setup_digest,
)

# isort: on

NUM_GPUS = 0
MANIFEST = Path("examples/dtap_agent_rl/environment/training-runtime.json")


def test_training_runtime_manifest_is_immutable_and_gpu_work_is_deferred() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest["image"]["reference"].endswith(manifest["image"]["manifest_digest"])
    assert manifest["runtime"]["cuda"] == "12.9"
    assert manifest["runtime"]["flash_attn"] == "2.8.3"
    assert manifest["runtime"]["megatron_commit"] == "1dcf0dafa884ad52ffb243625717a3471643e087"
    assert manifest["verification"]["gpu_execution"] == "deferred-until-capacity"
    assert len(manifest["verification"]["expected_pip_check_conflicts"]) == 12
    assert "/home/" not in json.dumps(manifest)
    assert runtime_setup_digest(manifest) == manifest["runtime_setup_digest"]


def test_training_runtime_probe_hides_gpus_and_checks_native_stack() -> None:
    manifest = load_manifest(MANIFEST)
    pull, probe = docker_commands(manifest)

    assert pull == ["docker", "pull", manifest["image"]["reference"]]
    assert "CUDA_VISIBLE_DEVICES=" in probe
    assert "--pull=never" in probe
    assert "--network" in probe and "none" in probe
    source = dependency_probe(manifest)
    assert '"megatron.core"' in source
    assert '"sglang"' in source
    assert '"flash_attn"' in source
    assert '"/sgl-workspace/sglang"' in source
    assert '"transformer_engine.pytorch"' in source
    assert "torch.cuda.is_available()" in source
    assert "actual_pip_conflicts != expected_pip_conflicts" in source


def test_training_runtime_prepare_executes_only_pinned_commands(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, check):
        calls.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert prepare(MANIFEST, pull=True, dry_run=False) == 0
    assert len(calls) == 2
    assert calls[0][0][2].startswith("docker.io/slimerl/slime@sha256:")
    assert calls[1][0][0:2] == ["docker", "run"]
    assert all(check is True for _, check in calls)


def test_training_runtime_manifest_rejects_mutable_reference(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["image"]["reference"] = "docker.io/slimerl/slime:latest-cu129"
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="immutable"):
        load_manifest(path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
