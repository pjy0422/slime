"""GPU-free validation for the pinned SGLang/Megatron training image."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "dtap-training-runtime"


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("version") != 1:
        raise ValueError("unsupported training runtime manifest")
    image = payload.get("image", {})
    digest = image.get("manifest_digest", "")
    reference = image.get("reference", "")
    if not digest.startswith("sha256:") or not reference.endswith(digest):
        raise ValueError("training runtime image must use an immutable sha256 reference")
    if payload.get("secret_policy") != "environment-only; no credential values are stored":
        raise ValueError("training runtime secret policy is missing")
    if payload.get("runtime_setup_digest") != runtime_setup_digest(payload):
        raise ValueError("training runtime setup digest mismatch")
    return payload


def dependency_probe(manifest: dict[str, Any]) -> str:
    runtime = json.dumps(manifest["runtime"], sort_keys=True)
    setup_digest = runtime_setup_digest(manifest)
    return f"""
import importlib.metadata as metadata
import json
import subprocess
import sys

expected = json.loads({runtime!r})
expected_pip_conflicts = set(json.loads({json.dumps(manifest['verification']['expected_pip_check_conflicts'], sort_keys=True)!r}))
versions = {{
    "python": ".".join(map(str, sys.version_info[:2])),
    "torch": metadata.version("torch"),
    "sglang": metadata.version("sglang"),
    "flash_attn": metadata.version("flash-attn"),
    "transformer_engine": metadata.version("transformer-engine"),
}}
for key in ("python", "torch", "sglang", "flash_attn", "transformer_engine"):
    if versions[key] != expected[key]:
        raise SystemExit(f"{{key}} mismatch: {{versions[key]}} != {{expected[key]}}")
import torch
if torch.version.cuda != expected["cuda"]:
    raise SystemExit(f"CUDA build mismatch: {{torch.version.cuda}} != {{expected['cuda']}}")
if torch.cuda.is_available():
    raise SystemExit("GPU-free probe unexpectedly has a visible CUDA device")
for module in ("sglang", "flash_attn", "megatron.core", "transformer_engine.pytorch"):
    __import__(module)
for name, path, expected_key in (
    ("Megatron", "/root/Megatron-LM", "megatron_commit"),
    ("SGLang", "/sgl-workspace/sglang", "sglang_commit"),
):
    head = subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected[expected_key]:
        raise SystemExit(f"{{name}} commit mismatch: {{head}}")
pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
)
actual_pip_conflicts = {{line.strip() for line in pip_check.stdout.splitlines() if line.strip()}}
if actual_pip_conflicts != expected_pip_conflicts:
    unexpected = sorted(actual_pip_conflicts - expected_pip_conflicts)
    missing = sorted(expected_pip_conflicts - actual_pip_conflicts)
    raise SystemExit(f"pip check conflicts changed: unexpected={{unexpected}}, missing={{missing}}")
print(json.dumps({{
    "status": "dependency-ready",
    "gpu_execution": "deferred",
    "reviewed_pip_conflicts": len(actual_pip_conflicts),
    "runtime_setup_digest": {setup_digest!r},
    **versions,
}}, sort_keys=True))
""".strip()


def runtime_setup_digest(manifest: Mapping[str, Any]) -> str:
    identity = {
        "image_reference": manifest["image"]["reference"],
        "runtime": manifest["runtime"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def docker_commands(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    reference = manifest["image"]["reference"]
    pull = ["docker", "pull", reference]
    probe = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--env",
        "CUDA_VISIBLE_DEVICES=",
        "--entrypoint",
        "python",
        reference,
        "-c",
        dependency_probe(manifest),
    ]
    return pull, probe


def prepare(manifest_path: Path, *, pull: bool, dry_run: bool) -> int:
    manifest = load_manifest(manifest_path)
    pull_command, probe_command = docker_commands(manifest)
    commands = ([pull_command] if pull else []) + [probe_command]
    if dry_run:
        for command in commands:
            print("+", " ".join(command[:2]), "...")
        print("GPU execution, NCCL, VRAM, and throughput remain deferred")
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    default_manifest = Path(__file__).resolve().parents[1] / "environment" / "training-runtime.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--skip-pull", action="store_true", help="probe an image that is already local")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return prepare(args.manifest, pull=not args.skip_pull, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
