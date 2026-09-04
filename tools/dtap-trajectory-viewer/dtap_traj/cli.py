"""Command-line interface for the self-contained DTAP trajectory viewer."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from .parser import build_timeline, find_policy_trace, find_victim_trace
from .render import write_html


def _first(root: Path, names: tuple[str, ...]) -> Path | None:
    directory = root if root.is_dir() else root.parent
    for name in names:
        direct = directory / name
        if direct.is_file():
            return direct
    for name in names:
        hits = sorted(directory.rglob(name))
        if hits:
            return hits[0]
    return None


def _guess_original(root: Path) -> Path | None:
    return _first(root, ("original-config.yaml", "original_config.yaml", "original/config.yaml"))


def _guess_submitted(root: Path) -> Path | None:
    return _first(root, ("submitted-config.yaml", "submitted_config.yaml", "candidate/config.yaml", "attack.yaml"))


def _guess_policy_prompt(root: Path) -> Path | None:
    return _first(root, ("policy-prompt.txt", "policy_prompt.txt"))


def _resolve_optional(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def _build(src: Path, args: argparse.Namespace) -> dict:
    explicit_victim = _resolve_optional(args.victim_trace)
    explicit_policy = _resolve_optional(args.policy_trace)
    if src.is_file() and "policy" in src.name.lower() and not explicit_policy:
        policy_trace = src
        victim_trace = explicit_victim
    else:
        victim_trace = explicit_victim or find_victim_trace(src)
        policy_trace = explicit_policy or (find_policy_trace(src) if src.is_dir() else None)
    submitted = _resolve_optional(args.submitted_yaml or args.yaml)
    original = _resolve_optional(args.original_yaml)
    policy_prompt = _resolve_optional(args.policy_prompt)
    if not args.no_yaml:
        submitted = submitted or _guess_submitted(src)
        original = original or _guess_original(src)
    policy_prompt = policy_prompt or _guess_policy_prompt(src)
    if victim_trace is None and policy_trace is None:
        raise FileNotFoundError(f"no victim or policy trace found under {src}")
    return build_timeline(
        victim_trace,
        policy_trace_path=policy_trace,
        policy_prompt_path=policy_prompt,
        original_yaml_path=original,
        submitted_yaml_path=submitted,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dtap-traj",
        description="View a DTAP victim trace, policy trace, and config diff together.",
    )
    parser.add_argument("path", help="run directory, victim trace, or policy stream-json")
    parser.add_argument("--victim-trace", help="explicit DTAP OpenClaw JSONL")
    parser.add_argument("--policy-trace", help="explicit Claude policy stream-json JSONL")
    parser.add_argument("--policy-prompt", help="policy instruction text shown as the first event")
    parser.add_argument("--original-yaml", help="original benchmark config.yaml")
    parser.add_argument("--submitted-yaml", help="policy-submitted config.yaml")
    parser.add_argument("-y", "--yaml", help="backward-compatible alias for --submitted-yaml")
    parser.add_argument("-o", "--out", default="trajectory.html", help="output HTML")
    parser.add_argument("-t", "--title", help="page title")
    parser.add_argument("--json", action="store_true", help="print viewer data as JSON")
    parser.add_argument("--open", action="store_true", help="open generated HTML")
    parser.add_argument("--no-yaml", action="store_true", help="disable YAML auto-discovery")
    args = parser.parse_args(argv)
    src = Path(args.path).expanduser().resolve()
    if not src.exists():
        parser.error(f"no such path: {src}")
    try:
        data = _build(src, args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    output = Path(args.out).expanduser().resolve()
    write_html(data, output, title=args.title)
    print(
        f"✓ {len(data['policy_timeline'])} policy events, "
        f"{len(data['timeline'])} victim events, "
        f"{len(data['payloads'])} submitted payloads → {output}"
    )
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
