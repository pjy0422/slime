"""Filesystem-to-SQLite metadata indexer for DTAP trajectory bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .db import TrajectoryDB

_METADATA_FILES = (
    "result.json",
    "episode-manifest.json",
    "judge-result.json",
    "judge-verdict.json",
    "policy.jsonl",
    "victim-trajectory.json",
    "victim-mcp-events.jsonl",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def _victim_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    trajectory = value.get("trajectory") if isinstance(value, dict) else None
    return len(trajectory) if isinstance(trajectory, list) else None


def discover_episode_dirs(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    candidates: set[Path] = set()
    for name in ("episode-manifest.json", "result.json"):
        for hit in root.rglob(name):
            parent = hit.parent.resolve()
            try:
                parent.relative_to(root)
            except ValueError:
                continue
            if (parent / "policy.jsonl").is_file() or (parent / "victim-trajectory.json").is_file():
                candidates.add(parent)
    if root.is_dir() and ((root / "policy.jsonl").is_file() or (root / "victim-trajectory.json").is_file()):
        candidates.add(root)
    return sorted(candidates)


def _source_mtime_ns(path: Path) -> int:
    values = []
    for name in _METADATA_FILES:
        target = path / name
        if target.exists():
            try:
                values.append(target.stat().st_mtime_ns)
            except OSError:
                pass
    return max(values, default=0)


def extract_episode_metadata(
    path: str | Path,
    root: str | Path,
    *,
    include_event_counts: bool = True,
) -> dict[str, Any]:
    path = Path(path).resolve()
    root = Path(root).resolve()
    relative = path.relative_to(root)
    result = _json(path / "result.json")
    manifest = _json(path / "episode-manifest.json")
    merged = {**manifest, **result}

    rel_parts = relative.parts
    domain = merged.get("domain")
    threat = merged.get("threat_model")
    if not domain and len(rel_parts) >= 2:
        domain = rel_parts[-2] if rel_parts[-1] in {"direct", "indirect"} else rel_parts[0]
    if not threat:
        threat = next((part for part in reversed(rel_parts) if part in {"direct", "indirect"}), None)

    episode_id = str(merged.get("episode_id") or "").strip()
    if not episode_id:
        episode_id = f"{domain or 'episode'}:{threat or 'unknown'}:{path.name}"

    # A root can be either one run (domain/threat) or a collection of runs
    # (run/domain/threat). Keep the single-run behavior while making the Run
    # facet useful for collections.
    run_name = root.name
    if len(rel_parts) >= 3:
        run_name = rel_parts[0]

    return {
        "episode_id": episode_id,
        "run_name": run_name,
        "domain": str(domain) if domain is not None else None,
        "threat_model": str(threat) if threat is not None else None,
        "status": merged.get("status"),
        "episode_status": merged.get("episode_status"),
        "risk_category": merged.get("risk_category"),
        "attack_success": merged.get("attack_success"),
        "evaluation_completed": merged.get("evaluation_completed"),
        "placement_applicable": merged.get("placement_applicable"),
        "placement_covered": merged.get("placement_covered"),
        "placement_actions": merged.get("placement_actions"),
        "placements_verified": merged.get("placements_verified"),
        "environment_steps": merged.get("environment_steps"),
        "policy_events": _line_count(path / "policy.jsonl") if include_event_counts else None,
        "victim_events": _victim_count(path / "victim-trajectory.json") if include_event_counts else None,
        "artifact_path": str(path),
        "source_mtime_ns": _source_mtime_ns(path),
    }


def index_root(root: str | Path, db: TrajectoryDB) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    scanned = 0
    updated = 0
    for episode_dir in discover_episode_dirs(root):
        scanned += 1
        item = extract_episode_metadata(episode_dir, root, include_event_counts=False)
        existing = db.get_episode(item["episode_id"])
        if existing and existing.get("source_mtime_ns") == item["source_mtime_ns"] and existing.get("artifact_path") == item["artifact_path"]:
            continue
        item["policy_events"] = _line_count(episode_dir / "policy.jsonl")
        item["victim_events"] = _victim_count(episode_dir / "victim-trajectory.json")
        db.upsert_episode(item)
        updated += 1
    return {"root": str(root), "scanned": scanned, "updated": updated}
