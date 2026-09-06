"""Load one viewer-ready DTAP episode bundle using the existing parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .parser import build_timeline, find_policy_trace, find_victim_mcp_events, find_victim_trace


def _first(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load_episode_bundle(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    victim = find_victim_trace(root)
    policy = find_policy_trace(root)
    mcp = find_victim_mcp_events(root)
    original = _first(root, ("original-config.yaml", "original_config.yaml"))
    submitted = _first(root, ("submitted-config.yaml", "submitted_config.yaml", "attack.yaml"))
    prompt = _first(root, ("policy-prompt.txt", "policy_prompt.txt"))
    manifest = _read_json(_first(root, ("episode-manifest.json",)))
    meta = {"episode_id": str(manifest.get("episode_id"))} if manifest.get("episode_id") else {}
    data = build_timeline(
        victim,
        meta=meta,
        policy_trace_path=policy,
        policy_prompt_path=prompt,
        original_yaml_path=original,
        submitted_yaml_path=submitted,
        victim_mcp_events_path=mcp,
    )
    result = _read_json(_first(root, ("result.json",)))
    judge = _read_json(_first(root, ("judge-verdict.json", "judge-result.json")))
    evaluation = {
        key: result[key]
        for key in (
            "status",
            "evaluation_completed",
            "failure_class",
            "attack_success",
            "episode_status",
            "placement_applicable",
            "placement_covered",
            "placement_actions",
            "placements_verified",
        )
        if key in result
    }
    if judge:
        evaluation["judge"] = judge
    if evaluation:
        data["evaluation"] = evaluation
    return data
