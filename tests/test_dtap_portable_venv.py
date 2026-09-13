"""CPU tests for the portable DTAP venv migration boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.dtap_agent_rl.scripts.portable_venv import (
    _locked_versions,
    dtap_worktree_digest,
    load_manifest,
    overlay_patch_paths,
    overlay_series_digest,
    portable_requirements,
)

NUM_GPUS = 0


def test_portable_requirements_remove_only_expected_editables() -> None:
    freeze = """
PyYAML==6.0.3
-e /workspace/slime
-e git+https://github.com/pjy0422/DecodingTrust-Agent@abc#egg=decodingtrust_agent_sdk
-e git+https://github.com/pjy0422/slime@abc#egg=dtap_traj&subdirectory=tools/dtap-trajectory-viewer
torch==2.11.0
"""

    requirements, editables = portable_requirements(freeze)

    assert requirements == ["PyYAML==6.0.3", "torch==2.11.0"]
    assert editables == ["decodingtrust-agent-sdk", "dtap-traj", "slime"]


@pytest.mark.parametrize(
    "line",
    [
        "-e /workspace/unknown-project",
        "package @ file:///workspace/package.whl",
        "package @ https://user:password@example.com/package.whl",
        "package @ https://example.com/package.whl?token=secret",
    ],
)
def test_portable_requirements_reject_host_paths_and_credentials(line: str) -> None:
    freeze = "\n".join(
        [
            "-e /workspace/slime",
            "-e git+https://example.com/DecodingTrust-Agent#egg=decodingtrust_agent_sdk",
            "-e git+https://example.com/slime#egg=dtap_traj&subdirectory=tools/dtap-trajectory-viewer",
            line,
        ]
    )

    with pytest.raises(ValueError):
        portable_requirements(freeze)


def test_locked_versions_requires_exact_unique_pins(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("Foo_Bar==1.0\n")
    assert _locked_versions(lock) == {"foo-bar": "1.0"}

    lock.write_text("foo==1\nfoo==2\n")
    with pytest.raises(ValueError, match="invalid or duplicate"):
        _locked_versions(lock)

    lock.write_text("foo>=1\n")
    with pytest.raises(ValueError, match="non-exact"):
        _locked_versions(lock)


def test_manifest_schema_and_lock_digest_are_committed() -> None:
    environment_dir = Path("examples/dtap_agent_rl/environment")
    manifest = load_manifest(environment_dir / "runtime-manifest.json")
    lock = environment_dir / manifest["lock"]["file"]

    assert hashlib.sha256(lock.read_bytes()).hexdigest() == manifest["lock"]["sha256"]
    assert len(_locked_versions(lock)) == manifest["lock"]["package_count"]
    assert manifest["secret_policy"] == "environment-only; no credential values are exported"
    serialized = json.dumps(manifest)
    lock_text = lock.read_text()
    assert "/home/" not in serialized
    assert "/media/" not in serialized
    assert "file:" not in lock_text
    assert "/home/" not in lock_text
    assert "/media/" not in lock_text
    assert not any(path.endswith(".qcow2") for path in manifest["repositories"]["dtap"]["overlay_untracked_files"])


def test_bootstrap_dry_run_is_non_mutating(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    dtap_root = tmp_path / "DecodingTrust-Agent"

    completed = subprocess.run(
        [
            "bash",
            "examples/dtap_agent_rl/scripts/bootstrap_portable_venv.sh",
            "--venv",
            str(venv),
            "--dtap-root",
            str(dtap_root),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "uv venv" in completed.stdout
    assert "--require-editable-paths" in completed.stdout
    assert "dry-run complete" in completed.stdout
    assert not venv.exists()
    assert not dtap_root.exists()


def test_export_cli_preserves_a_venv_python_symlink(monkeypatch, tmp_path: Path) -> None:
    from examples.dtap_agent_rl.scripts import portable_venv

    python_link = tmp_path / "venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(Path(sys.executable).resolve())
    observed = {}

    def fake_run(*command, cwd=None):
        observed.setdefault("commands", []).append((command, cwd))
        if command[1:] == ("-m", "pip", "freeze", "--all"):
            return "\n".join(
                [
                    "foo==1",
                    "-e /workspace/slime",
                    "-e git+https://example.com/DecodingTrust-Agent#egg=decodingtrust_agent_sdk",
                    "-e git+https://example.com/slime#egg=dtap_traj&subdirectory=tools/dtap-trajectory-viewer",
                ]
            )
        raise AssertionError(command)

    monkeypatch.setattr(portable_venv, "_run", fake_run)
    monkeypatch.setattr(portable_venv, "_runtime_probe", lambda python: (_ for _ in ()).throw(RuntimeError(python)))
    args = type(
        "Args",
        (),
        {
            "python": os.fspath(python_link),
            "slime_root": os.fspath(tmp_path),
            "dtap_root": os.fspath(tmp_path),
            "output_dir": os.fspath(tmp_path / "output"),
        },
    )()

    with pytest.raises(RuntimeError, match=str(python_link)):
        portable_venv.export_environment(args)
    assert observed["commands"][0][0][0] == str(python_link)


def test_dtap_worktree_digest_detects_tracked_and_untracked_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    clean_digest = dtap_worktree_digest(repo, [])

    tracked.write_text("patched\n")
    patched_digest = dtap_worktree_digest(repo, [])
    (repo / "new.txt").write_text("overlay\n")
    untracked_digest = dtap_worktree_digest(repo, ["new.txt"])

    assert clean_digest != patched_digest
    assert patched_digest != untracked_digest


def test_overlay_series_has_one_ordered_source_of_truth() -> None:
    integration_dir = Path("examples/dtap_agent_rl/dtap_integration")
    paths = overlay_patch_paths(integration_dir)

    assert len(paths) == 20
    assert paths[0].name == "m4-runtime-integration.patch"
    assert paths[-1].name == "p7-holdout-e2e-stability.patch"
    assert len(overlay_series_digest(integration_dir)) == 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
