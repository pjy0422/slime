"""Exercise the M6 action receipt and placement lookup against one real DTAP task."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from examples.dtap_agent_rl.episode import load_task_snapshot
from examples.dtap_agent_rl.placement import DtapPlacementRunner, PlacementCoordinator
from examples.dtap_agent_rl.policy_contract import PolicyContract
from examples.dtap_agent_rl.scheduler import AttemptScheduler
from examples.dtap_agent_rl.scripts.smoke_m2_api import live_view
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.validation import ValidationContext


def _first_environment_step(raw: dict, tool: str | None = None) -> dict:
    for turn in raw.get("Attack", {}).get("attack_turns", []):
        for step in turn.get("attack_steps", []):
            if step.get("type") == "environment" and (tool is None or step.get("injection_mcp_tool") == tool):
                return {**step, "turn_id": turn["turn_id"]}
    raise RuntimeError("task has no matching environment attack step")


async def run(args: argparse.Namespace) -> dict:
    snapshot = load_task_snapshot(args.task_dir)
    view = await live_view(snapshot)
    step = json.loads(args.step_json) if args.step_json else _first_environment_step(snapshot.raw_config, args.tool)
    if args.turn_id is not None:
        step["turn_id"] = args.turn_id
    policy = M4SecurityPolicy(
        max_submit_calls=1,
        max_parallel_attempts=1,
        max_queued_attempts=1,
        queue_wait_timeout_seconds=args.timeout,
    )
    scheduler = AttemptScheduler(max_parallel=1, max_queued=1, wait_timeout=args.timeout)
    runner = DtapPlacementRunner(
        dtap_root=args.dtap_root,
        security_policy=policy,
        scheduler=scheduler,
        python_executable=args.python,
        timeout_seconds=args.timeout,
    )
    with tempfile.TemporaryDirectory(prefix="slime-m6-placement-") as temp:
        placement = PlacementCoordinator(
            validation_context=ValidationContext.from_view(view),
            source_task_dir=snapshot.task_dir,
            source_manifest=snapshot.benchmark_manifest,
            episode_root=Path(temp) / "placement",
            runner=runner,
            security_policy=policy,
            policy_contract=PolicyContract(),
            max_actions=1,
        )
        receipt = await placement.apply(step)
        result = placement.validate(receipt.get("action_id", "")) if receipt.get("accepted") else None
        return {
            "step": {key: value for key, value in step.items() if key != "kwargs"},
            "receipt": receipt,
            "placement": result,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--dtap-root", required=True, type=Path)
    parser.add_argument("--python", default=None)
    parser.add_argument("--step-json")
    parser.add_argument("--tool", help="select an existing qualified environment tool")
    parser.add_argument("--turn-id", type=int, help="override the source example turn")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    placement = result.get("placement")
    return 0 if isinstance(placement, dict) and placement.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
