"""One-submit real DTAP M3 smoke using a policy-authored JSON plan."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from examples.dtap_agent_rl.attempt_runner import DtapAttemptRunner
from examples.dtap_agent_rl.episode import load_task_snapshot
from examples.dtap_agent_rl.episode_runtime import EpisodeRuntimeState
from examples.dtap_agent_rl.scripts.smoke_m2_api import live_view
from examples.dtap_agent_rl.submission import SubmissionCoordinator
from examples.dtap_agent_rl.validation import ValidationContext


async def _main(args) -> None:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    snapshot = load_task_snapshot(args.task_dir)
    view = await live_view(snapshot)
    runner = DtapAttemptRunner(
        max_parallel=1,
        agent_type=args.agent_type,
        model=args.model,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout,
        dtap_root=args.dtap_root,
    )
    with tempfile.TemporaryDirectory(prefix="slime-m3-dtap-") as temp_dir:
        controller = SubmissionCoordinator(
            validation_context=ValidationContext.from_view(view),
            runtime=EpisodeRuntimeState(max_submissions=1),
            source_task_dir=snapshot.task_dir,
            episode_root=Path(temp_dir) / "episode",
            runner=runner,
        )
        receipt = await controller.submit(plan)
        snapshot.assert_config_unchanged()
        print(json.dumps(receipt, sort_keys=True))
        if not receipt.get("accepted"):
            raise RuntimeError(f"real DTAP submission was not accepted: {receipt}")
        print("M3 REAL DTAP ONE-SUBMIT SMOKE: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dtap-root", type=Path, required=True)
    parser.add_argument("--agent-type", default="openaisdk")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    args.task_dir = args.task_dir.expanduser().resolve()
    args.plan = args.plan.expanduser().resolve()
    args.dtap_root = args.dtap_root.expanduser().resolve()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
