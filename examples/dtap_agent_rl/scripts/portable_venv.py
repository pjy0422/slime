"""Export and verify a portable snapshot of the DTAP development venv."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

SCHEMA_VERSION = 1
LOCK_NAME = "requirements-linux-x86_64-py312.lock.txt"
MANIFEST_NAME = "runtime-manifest.json"
EXPECTED_EDITABLES = {
    "decodingtrust-agent-sdk": ("DecodingTrust-Agent", "decodingtrust_agent_sdk"),
    "dtap-traj": ("tools/dtap-trajectory-viewer", "dtap_traj"),
    "slime": ("/slime",),
}
EXTERNAL_TOOL_VERSION_ARGS = {
    "claude": ("--version",),
    "node": ("--version",),
    "openclaw": ("--version",),
}
_SECRET_QUERY_KEY = re.compile(r"(?i)(?:token|key|password|secret|auth)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _editable_name(line: str) -> str | None:
    lowered = line.lower().replace("_", "-")
    for name, markers in EXPECTED_EDITABLES.items():
        if any(marker.lower().replace("_", "-") in lowered for marker in markers):
            return name
    return None


def _check_requirement_is_portable(line: str) -> None:
    lowered = line.lower()
    if "file:" in lowered:
        raise ValueError(f"local file requirement is not portable: {line!r}")
    for match in re.finditer(r"https?://[^\s]+", line):
        parsed = urlsplit(match.group(0))
        if parsed.username or parsed.password:
            raise ValueError("requirement URL contains embedded credentials")
        if any(_SECRET_QUERY_KEY.search(key) for key, _ in parse_qsl(parsed.query)):
            raise ValueError("requirement URL contains a secret-like query parameter")


def portable_requirements(freeze_output: str) -> tuple[list[str], list[str]]:
    """Remove known local editables and reject other host-specific entries."""

    requirements: list[str] = []
    editables: list[str] = []
    for raw_line in freeze_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-e ", "--editable ")):
            editable = _editable_name(line)
            if editable is None:
                raise ValueError(f"unexpected editable requirement: {line!r}")
            if editable in editables:
                raise ValueError(f"duplicate editable requirement: {editable}")
            editables.append(editable)
            continue
        _check_requirement_is_portable(line)
        requirements.append(line)
    missing = set(EXPECTED_EDITABLES) - set(editables)
    if missing:
        raise ValueError(f"missing expected editable requirements: {sorted(missing)}")

    def requirement_name(line: str) -> str:
        return re.split(r"[<>=!~ @]", line, maxsplit=1)[0].lower()

    return sorted(requirements, key=requirement_name), sorted(editables)


def _run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def overlay_patch_paths(integration_dir: Path) -> list[Path]:
    """Load the exact ordered patch series shared with the apply script."""

    paths = []
    for line in (integration_dir / "patch-series.txt").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kind, patch_name = stripped.split()
        if kind not in {"base", "incremental"}:
            raise ValueError(f"invalid DTAP overlay patch kind: {kind!r}")
        path = integration_dir / "patches" / patch_name
        if not path.is_file():
            raise ValueError(f"DTAP overlay patch is missing: {patch_name}")
        paths.append(path)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("DTAP overlay patch series is empty or contains duplicates")
    return paths


def overlay_series_digest(integration_dir: Path) -> str:
    lines = "".join(f"{_sha256(path)}\n" for path in overlay_patch_paths(integration_dir))
    return hashlib.sha256(lines.encode()).hexdigest()


def overlay_untracked_files(integration_dir: Path, dtap_root: Path) -> list[str]:
    """Find source files created by the overlay, excluding downloaded assets."""

    candidates = set()
    pattern = re.compile(r"^diff --git a/(.+) b/(.+)$", re.MULTILINE)
    for patch in overlay_patch_paths(integration_dir):
        for _, relative_path in pattern.findall(patch.read_text()):
            path = Path(relative_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"overlay patch contains an unsafe path: {relative_path!r}")
            tracked = (
                subprocess.run(
                    ["git", "cat-file", "-e", f"HEAD:{relative_path}"],
                    cwd=dtap_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
            if not tracked and (dtap_root / path).is_file():
                candidates.add(path.as_posix())
    return sorted(candidates)


def dtap_worktree_digest(repo: Path, overlay_untracked: list[str]) -> str:
    """Hash tracked changes and only source files created by the DTAP overlay."""

    digest = hashlib.sha256()
    changed = subprocess.run(
        ["git", "diff", "--name-only", "-z", "HEAD"], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout.split(b"\0")
    source_paths = [os.fsdecode(path) for path in changed if path]
    for relative_path in sorted(set(source_paths) | set(overlay_untracked)):
        encoded_path = os.fsencode(relative_path)
        path = repo / relative_path
        digest.update(b"source\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        if not path.exists() and not path.is_symlink():
            digest.update(b"deleted\0")
            continue
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        else:
            content = path.read_bytes()
            if b"\0" not in content:
                content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n") + b"\n"
            digest.update(content)
    return digest.hexdigest()


def _safe_remote(repo: Path) -> str:
    remote = _run("git", "remote", "get-url", "origin", cwd=repo)
    _check_requirement_is_portable(remote)
    parsed = urlsplit(remote)
    if parsed.scheme in {"http", "https"} and (parsed.username or parsed.password):
        raise ValueError("git remote contains embedded credentials")
    return remote


def _runtime_probe(python: Path) -> dict[str, Any]:
    code = """
import json, platform
import importlib.metadata
import torch
print(json.dumps({
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "system": platform.system(),
    "machine": platform.machine(),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_capabilities": sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}),
    "uv_version": importlib.metadata.version("uv"),
}))
"""
    return json.loads(_run(str(python), "-c", code))


def _overlay_marker(dtap_root: Path) -> tuple[Path, str]:
    marker = Path(
        _run(
            "git",
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "dtap-agent-rl-overlay.sha256",
            cwd=dtap_root,
        )
    )
    if not marker.is_file() or not re.fullmatch(r"[0-9a-f]{64}\n?", marker.read_text()):
        raise ValueError("DTAP checkout is missing a valid applied-overlay marker")
    return marker, marker.read_text().strip()


def _external_tool_versions() -> dict[str, str]:
    versions = {}
    for command, version_args in EXTERNAL_TOOL_VERSION_ARGS.items():
        executable = shutil.which(command)
        if executable is None:
            raise ValueError(f"required external tool is unavailable: {command}")
        versions[command] = _run(executable, *version_args).splitlines()[0]
    return versions


def export_environment(args: argparse.Namespace) -> None:
    # A venv's Python is commonly a symlink to the base interpreter. Resolving
    # it would silently export the base environment instead of the requested
    # venv, so retain the executable path exactly (apart from making it absolute).
    python = Path(os.path.abspath(args.python))
    slime_root = Path(args.slime_root).resolve()
    dtap_root = Path(args.dtap_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze = _run(str(python), "-m", "pip", "freeze", "--all")
    requirements, editables = portable_requirements(freeze)
    lock_path = output_dir / LOCK_NAME
    lock_path.write_text("\n".join(requirements) + "\n")

    probe = _runtime_probe(python)
    uv_version = probe.pop("uv_version")
    if probe["system"] != "Linux" or probe["machine"] != "x86_64":
        raise ValueError("the portable DTAP environment currently supports Linux x86_64 only")
    integration_dir = slime_root / "examples/dtap_agent_rl/dtap_integration"
    _, applied_overlay_digest = _overlay_marker(dtap_root)
    overlay_digest = overlay_series_digest(integration_dir)
    if applied_overlay_digest != overlay_digest:
        raise ValueError("DTAP overlay marker is stale; apply the current slime patch series first")
    overlay_files = overlay_untracked_files(
        integration_dir,
        dtap_root,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "lock": {
            "file": LOCK_NAME,
            "sha256": _sha256(lock_path),
            "package_count": len(requirements),
        },
        "platform": probe,
        "bootstrap": {
            "uv_version": uv_version,
            "required_commands": ["git", "docker", "nvidia-smi", *sorted(EXTERNAL_TOOL_VERSION_ARGS)],
            "external_tool_versions": _external_tool_versions(),
        },
        "repositories": {
            "slime": {"install": "current_checkout", "remote": _safe_remote(slime_root)},
            "dtap": {
                "remote": _safe_remote(dtap_root),
                "commit": _run("git", "rev-parse", "HEAD", cwd=dtap_root),
                "overlay_digest": overlay_digest,
                "overlay_untracked_files": overlay_files,
                "worktree_digest": dtap_worktree_digest(dtap_root, overlay_files),
                "overlay_script": "examples/dtap_agent_rl/dtap_integration/apply.sh",
            },
        },
        "editable_projects": editables,
        "secret_policy": "environment-only; no credential values are exported",
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported portable environment manifest schema")
    lock = manifest.get("lock")
    repositories = manifest.get("repositories")
    if not isinstance(lock, dict) or not isinstance(repositories, dict):
        raise ValueError("portable environment manifest is incomplete")
    return manifest


def _locked_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lock_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"lock contains a non-exact requirement: {line!r}")
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if not name or not version or normalized in versions:
            raise ValueError(f"lock contains an invalid or duplicate requirement: {line!r}")
        versions[normalized] = version
    return versions


def verify_environment(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    lock_path = manifest_path.parent / manifest["lock"]["file"]
    if _sha256(lock_path) != manifest["lock"]["sha256"]:
        raise ValueError("portable environment lock digest does not match the manifest")
    locked = _locked_versions(lock_path)
    if len(locked) != manifest["lock"]["package_count"]:
        raise ValueError("portable environment package count does not match the manifest")

    expected_platform = manifest["platform"]
    if platform.system() != expected_platform["system"] or platform.machine() != expected_platform["machine"]:
        raise ValueError("host platform does not match the portable environment")
    if platform.python_version() != expected_platform["python_version"]:
        raise ValueError("Python patch version does not match the portable environment")

    mismatches = []
    installed = {
        re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    for name, version in locked.items():
        if installed.get(name) != version:
            mismatches.append(f"{name}: expected {version}, found {installed.get(name, 'missing')}")
    if mismatches:
        raise ValueError("locked package mismatch:\n" + "\n".join(mismatches))

    slime_root = Path(args.slime_root).resolve()
    dtap_root = Path(args.dtap_root).resolve()
    expected_dtap = manifest["repositories"]["dtap"]
    if _run("git", "rev-parse", "HEAD", cwd=dtap_root) != expected_dtap["commit"]:
        raise ValueError("DTAP commit does not match the portable environment")
    _, overlay_digest = _overlay_marker(dtap_root)
    if overlay_digest != expected_dtap["overlay_digest"]:
        raise ValueError("DTAP overlay digest does not match the portable environment")
    if dtap_worktree_digest(dtap_root, expected_dtap["overlay_untracked_files"]) != expected_dtap["worktree_digest"]:
        raise ValueError("DTAP patched worktree does not match the portable environment")
    if _safe_remote(slime_root) != manifest["repositories"]["slime"]["remote"]:
        raise ValueError("slime remote does not match the portable environment")
    if _safe_remote(dtap_root) != expected_dtap["remote"]:
        raise ValueError("DTAP remote does not match the portable environment")
    missing_editables = set(manifest["editable_projects"]) - set(installed)
    if missing_editables:
        raise ValueError(f"portable environment is missing editable projects: {sorted(missing_editables)}")

    if args.require_editable_paths:
        expected_paths = {
            "slime": slime_root,
            "decodingtrust-agent-sdk": dtap_root,
            "dtap-traj": slime_root / "tools/dtap-trajectory-viewer",
        }
        for name, expected_path in expected_paths.items():
            direct_url_text = importlib.metadata.distribution(name).read_text("direct_url.json")
            direct_url = json.loads(direct_url_text) if direct_url_text else {}
            source_url = direct_url.get("url", "")
            if direct_url.get("dir_info", {}).get("editable") is not True or not source_url.startswith("file://"):
                raise ValueError(f"{name} is not installed as a local editable project")
            actual_path = Path(unquote(urlsplit(source_url).path)).resolve()
            if actual_path != expected_path.resolve():
                raise ValueError(f"{name} editable path does not match the restored checkout")

    if not args.skip_system_checks:
        for command in manifest["bootstrap"]["required_commands"]:
            if shutil.which(command) is None:
                raise ValueError(f"required command is unavailable: {command}")
        actual_tool_versions = _external_tool_versions()
        if actual_tool_versions != manifest["bootstrap"]["external_tool_versions"]:
            raise ValueError("external policy/victim tool versions do not match the portable environment")
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import torch

        if (
            torch.__version__ != expected_platform["torch_version"]
            or torch.version.cuda != expected_platform["torch_cuda"]
        ):
            raise ValueError("PyTorch/CUDA build does not match the portable environment")
        if not torch.cuda.is_available():
            raise ValueError("PyTorch cannot access an NVIDIA GPU")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="export the active venv into a portable lock")
    export_parser.add_argument("--python", default=sys.executable)
    export_parser.add_argument("--slime-root", required=True)
    export_parser.add_argument("--dtap-root", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.set_defaults(func=export_environment)

    verify_parser = subparsers.add_parser("verify", help="verify a restored venv and source checkouts")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--slime-root", required=True)
    verify_parser.add_argument("--dtap-root", required=True)
    verify_parser.add_argument("--skip-system-checks", action="store_true")
    verify_parser.add_argument("--require-editable-paths", action="store_true")
    verify_parser.set_defaults(func=verify_environment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
