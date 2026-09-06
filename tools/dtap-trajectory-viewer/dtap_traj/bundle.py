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


def _judge_source(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return "deterministic"
    if metadata.get("judge_tool") == "llm_judge" or metadata.get("verdict_source") == "llm_judge":
        return "llm_as_judge"
    if any(metadata.get(key) is not None for key in ("llm_judge", "llm_model", "gpt_model", "gpt_score", "gpt_rationale")):
        return "llm_as_judge"
    return "deterministic"


def _judge_component(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    success_key = f"{name}_success"
    message_key = f"{name}_message"
    metadata_key = f"{name}_metadata"
    if not any(key in result for key in (success_key, message_key, metadata_key)):
        return None
    metadata = result.get(metadata_key)
    return {
        "name": name,
        "success": result.get(success_key),
        "message": result.get(message_key),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "source": _judge_source(metadata),
    }


def load_judge_results(root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    result = _read_json(_first(root, ("judge-result.json", "judge_result.json")))
    verdict = _read_json(_first(root, ("judge-verdict.json", ".m4-verdict.json")))
    components = [component for name in ("task", "attack") if (component := _judge_component(result, name)) is not None]
    return {
        "available": bool(result or verdict),
        "components": components,
        "error": result.get("error") if result else None,
        "reward_firewall": verdict,
        "raw": result,
    }


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
    judges = load_judge_results(root)
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
    if judges["raw"]:
        evaluation["judge"] = judges["raw"]
    elif judges["reward_firewall"]:
        evaluation["judge"] = judges["reward_firewall"]
    if evaluation:
        data["evaluation"] = evaluation
    data["judges"] = judges
    return data
