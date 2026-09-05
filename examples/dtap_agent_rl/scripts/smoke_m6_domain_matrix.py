"""Run one representative M6 E2E task per domain and threat model in parallel."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import shutil
import sys
from pathlib import Path
from typing import Any


DOMAINS = (
    "browser", "code", "crm", "customer-service", "finance", "legal",
    "macos", "medical", "os-filesystem", "research", "telecom", "travel",
    "windows", "workflow",
)


def _first_record(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
    raise ValueError(f"empty benchmark list: {path}")


def _task_dir(dtap_root: Path, record: dict[str, Any]) -> Path:
    return dtap_root.joinpath(
        "dataset", str(record["domain"]), "malicious",
        str(record["threat_model"]), str(record["risk_category"]),
        str(record["task_id"]),
    )


def _passed_payload(stdout: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    matches: list[tuple[int, dict[str, Any]]] = []
    for index, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status") == "passed":
            matches.append((end, value))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def _stored_results(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for threat_model in ("direct", "indirect"):
            path = root / domain / threat_model / "result.json"
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                value = {"result_error": type(exc).__name__}
            if (
                not isinstance(value, dict)
                or value.get("domain") != domain
                or value.get("threat_model") != threat_model
                or value.get("status") not in ("passed", "failed")
            ):
                value = {
                    "domain": domain,
                    "threat_model": threat_model,
                    "status": "failed",
                    "result_error": "invalid stored result",
                }
            results.append(value)
    return results


def _failure_count(results: list[dict[str, Any]]) -> int:
    """Platform names never exempt failures from the test verdict."""
    return sum(item.get("status") != "passed" for item in results)


async def _run_case(
    args: argparse.Namespace,
    *,
    domain: str,
    threat_model: str,
    slot: int,
) -> dict[str, Any]:
    record = _first_record(args.dtap_root / "benchmark" / domain / f"{threat_model}.jsonl")
    task_dir = _task_dir(args.dtap_root, record)
    case_dir = args.artifacts_root / domain / threat_model
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if args.resume and result_path.exists():
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("status") == "passed":
                return previous
        except (OSError, json.JSONDecodeError):
            pass

    start = args.port_range_start + slot * args.port_range_stride
    end = start + 511
    command = [
        args.python, "-m", "examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e",
        "--task-dir", str(task_dir), "--dtap-root", str(args.dtap_root),
        "--python", args.python, "--policy-model", args.policy_model,
        "--victim-model", args.victim_model,
        "--victim-agent-type", args.victim_agent_type,
        "--policy-max-turns", str(args.policy_max_turns),
        "--victim-max-turns", str(args.victim_max_turns),
        "--timeout", str(args.timeout), "--m6-placement",
        "--port-range-start", str(start), "--artifacts-dir", str(case_dir),
    ]
    env = os.environ.copy()
    env.update({
        "DT_DISABLE_DEFAULT_PORTS": "1",
        "DT_PORT_RANGE": f"{start}-{end}",
        "DTAP_ENV_VERIFICATION": "placement",
        "DTAP_ENV_VERIFICATION_STRICT": "1",
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(args.dtap_root), env.get("PYTHONPATH", "")) if part
        ),
    })
    process = await asyncio.create_subprocess_exec(
        *command, cwd=args.slime_root, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        raw_out, raw_err = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                # Let the smoke's asyncio cancellation unwind its nested DTAP
                # runner, which owns a separate process group for victim work.
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        raise
    stdout = raw_out.decode(errors="replace")
    stderr = raw_err.decode(errors="replace")
    (case_dir / "run.stdout.log").write_text(stdout, encoding="utf-8")
    (case_dir / "run.stderr.log").write_text(stderr, encoding="utf-8")
    payload = _passed_payload(stdout)
    result: dict[str, Any] = {
        "status": "passed" if process.returncode == 0 and payload else "failed",
        "domain": domain,
        "threat_model": threat_model,
        "task_dir": str(task_dir),
        "task_id": str(record["task_id"]),
        "risk_category": str(record["risk_category"]),
        "returncode": process.returncode,
        "port_range": f"{start}-{end}",
    }
    if payload:
        result.update({
            key: payload.get(key) for key in (
                "attack_success", "episode_status", "environment_steps",
                "placement_actions", "placements_verified",
                "matches_source_template",
            )
        })
    else:
        tail = (stderr or stdout)[-2000:]
        result["error_tail"] = tail
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


async def _main(args: argparse.Namespace) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    if shutil.which("openclaw") is None:
        raise RuntimeError("openclaw CLI is required")
    highest = args.port_range_start + (args.max_parallel - 1) * args.port_range_stride + 511
    if args.port_range_stride < 512 or highest > 65535:
        raise ValueError("parallel workers require disjoint valid 512-port ranges")
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    for domain in args.domains:
        for threat_model in args.threat_models:
            queue.put_nowait((domain, threat_model))
    results: list[dict[str, Any]] = []

    async def worker(slot: int) -> None:
        while not queue.empty():
            try:
                domain, threat_model = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(await _run_case(
                    args, domain=domain, threat_model=threat_model, slot=slot,
                ))
            finally:
                queue.task_done()

    await asyncio.gather(*(worker(slot) for slot in range(args.max_parallel)))
    results.sort(key=lambda item: (item["domain"], item["threat_model"]))
    stored = _stored_results(args.artifacts_root)
    stored.sort(key=lambda item: (item["domain"], item["threat_model"]))
    summary = {
        "policy_model": args.policy_model,
        "victim_model": args.victim_model,
        "victim_agent_type": args.victim_agent_type,
        "total": len(stored),
        "passed": sum(item["status"] == "passed" for item in stored),
        "failed": _failure_count(stored),
        "selected_total": len(results),
        "selected_failed": _failure_count(results),
        "results": stored,
    }
    (args.artifacts_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary["selected_failed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtap-root", type=Path, required=True)
    parser.add_argument("--slime-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--threat-models", nargs="+", choices=("direct", "indirect"), default=["direct", "indirect"])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--port-range-start", type=int, default=20_000)
    parser.add_argument("--port-range-stride", type=int, default=1_024)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--policy-model", default="deepseek-v4-flash")
    parser.add_argument("--victim-model", default="deepseek-v4-flash")
    parser.add_argument("--victim-agent-type", default="openclaw")
    parser.add_argument("--policy-max-turns", type=int, default=64)
    parser.add_argument("--victim-max-turns", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.dtap_root = args.dtap_root.expanduser().resolve()
    args.slime_root = args.slime_root.expanduser().resolve()
    args.artifacts_root = args.artifacts_root.expanduser().resolve()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
