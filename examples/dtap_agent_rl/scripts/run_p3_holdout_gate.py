"""Run the disjoint 24-case Linux E2E matrix for post-M7 P3."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtap-root", required=True, type=Path)
    parser.add_argument("--slime-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--retry-passes", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--policy-model", default="deepseek-v4-flash")
    parser.add_argument("--victim-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--feedback-mode",
        choices=("disabled", "final", "final+deterministic", "final+deterministic+digestor"),
        default="final+deterministic",
    )
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="use only when the target checkout already contains the managed overlay",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for the P3 holdout gate")
    slime = args.slime_root.expanduser().resolve()
    dtap = args.dtap_root.expanduser().resolve()
    artifacts = args.artifacts_root.expanduser().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(dtap), str(slime), env.get("PYTHONPATH", "")) if part
    )
    if not args.skip_overlay:
        subprocess.run(
            [str(slime / "examples/dtap_agent_rl/dtap_integration/apply.sh"), str(dtap)],
            cwd=slime,
            env=env,
            check=True,
        )
    command = [
        sys.executable,
        "-m",
        "examples.dtap_agent_rl.scripts.smoke_m6_domain_matrix",
        "--dtap-root",
        str(dtap),
        "--slime-root",
        str(slime),
        "--artifacts-root",
        str(artifacts),
        "--selection-profile",
        "holdout-v1",
        "--max-parallel",
        str(args.max_parallel),
        "--policy-model",
        args.policy_model,
        "--victim-model",
        args.victim_model,
        "--victim-agent-type",
        "openclaw",
        "--max-submissions",
        "2",
        "--feedback-mode",
        args.feedback_mode,
    ]
    if args.resume:
        command.append("--resume")
    parallel_index = command.index("--max-parallel") + 1
    completed = subprocess.run(command, cwd=slime, env=env, check=False)
    retry_parallel = args.max_parallel
    for _ in range(args.retry_passes):
        if completed.returncode == 0:
            break
        time.sleep(max(0.0, args.retry_delay))
        retry_parallel = max(1, retry_parallel // 2)
        command[parallel_index] = str(retry_parallel)
        if "--resume" not in command:
            command.append("--resume")
        completed = subprocess.run(command, cwd=slime, env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
