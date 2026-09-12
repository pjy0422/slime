"""Apply the managed DTAP overlay and run the P0-P2 Linux release gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtap-root", required=True, type=Path)
    parser.add_argument("--slime-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-root", type=Path, default=Path("m6-linux-artifacts"))
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()
    slime = args.slime_root.expanduser().resolve()
    dtap = args.dtap_root.expanduser().resolve()
    artifacts = args.artifacts_root.expanduser().resolve()
    python = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(dtap), str(slime), env.get("PYTHONPATH", "")) if part)
    env["DTAP_ROOT"] = str(dtap)

    _run(
        [str(slime / "examples/dtap_agent_rl/dtap_integration/apply.sh"), str(dtap)],
        cwd=slime,
        env=env,
    )
    runtime_command = [
        python,
        "-m",
        "examples.dtap_agent_rl.scripts.verify_m6_runtime_lock",
        "--dtap-root",
        str(dtap),
    ]
    if args.skip_images:
        runtime_command.append("--skip-images")
    _run(runtime_command, cwd=slime, env=env)
    _run(
        [python, "-m", "pytest", "-q", "tests/test_env_verification.py", "tests/test_openclaw_mcp_events.py"],
        cwd=dtap,
        env=env,
    )
    _run(
        [python, "-m", "pytest", "-q", "examples/dtap_agent_rl/tests", "tools/dtap-trajectory-viewer/tests"],
        cwd=slime,
        env=env,
    )
    _run(
        [
            python,
            "-m",
            "examples.dtap_agent_rl.scripts.audit_m6_adapter_coverage",
            "--dtap-root",
            str(dtap),
            "--output",
            str(artifacts / "adapter-coverage.json"),
        ],
        cwd=slime,
        env=env,
    )
    if args.skip_e2e:
        return
    if not env.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for the 24-case E2E gate")
    _run(
        [
            python,
            "-m",
            "examples.dtap_agent_rl.scripts.smoke_m6_domain_matrix",
            "--dtap-root",
            str(dtap),
            "--slime-root",
            str(slime),
            "--artifacts-root",
            str(artifacts / "domain-matrix"),
            "--max-parallel",
            str(args.max_parallel),
            "--policy-model",
            "deepseek-v4-flash",
            "--victim-model",
            "deepseek-v4-flash",
            "--victim-agent-type",
            "openclaw",
        ],
        cwd=slime,
        env=env,
    )


if __name__ == "__main__":
    main()
