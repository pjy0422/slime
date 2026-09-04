"""Run exactly one DTAP task in one fresh TaskExecutor process."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path


VICTIM_BASE_URL_ENV = "DTAP_VICTIM_ANTHROPIC_BASE_URL"
VICTIM_AUTH_FROM_API_KEY_ENV = "DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN"


def _configure_victim_provider() -> None:
    """Translate trusted runner-only aliases inside the isolated DTAP child.

    The M4 parent still refuses to inherit policy adapter variables directly.
    These aliases must be explicitly allowlisted by the trusted runner config.
    """

    base_url = os.environ.pop(VICTIM_BASE_URL_ENV, "").strip()
    auth_from_key = os.environ.pop(VICTIM_AUTH_FROM_API_KEY_ENV, "") == "1"
    if base_url:
        if not base_url.startswith(("https://", "http://")):
            raise RuntimeError("invalid trusted victim provider URL")
        os.environ["ANTHROPIC_BASE_URL"] = base_url
    if auth_from_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("victim API key is required for auth-token mapping")
        os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key


@contextmanager
def _task_environment(instances):
    updates: dict[str, str] = {}
    for instance in instances.values():
        updates.update({str(key): str(value) for key, value in instance.ports.items()})
        name = instance.env_name.upper().replace("-", "_")
        updates[f"{name}_PROJECT_NAME"] = str(instance.project_name)
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _run(args) -> int:
    _configure_victim_provider()
    from dt_arena.src.types.task import AttackConfig, TaskConfig
    from eval.task_runner import run_single_task
    from utils.task_executor import ScheduledTask, TaskExecutor, get_task_environments

    task_cfg = TaskConfig.from_yaml(str(args.task_dir / "config.yaml"))
    attack_cfg = AttackConfig.from_yaml(str(args.task_dir / "config.yaml"))
    scheduled = ScheduledTask(
        task_dir=args.task_dir,
        environments=frozenset(get_task_environments(args.task_dir)),
        original_index=0,
        domain=getattr(task_cfg, "domain", None),
        threat_model=getattr(attack_cfg, "threat_model", None),
        risk_category=getattr(attack_cfg, "risk_category", None),
        task_id=getattr(task_cfg, "task_id", None),
    )
    executor = TaskExecutor(max_parallel=1)

    async def callback(task, instances):
        if args.started_path is not None:
            args.started_path.parent.mkdir(parents=True, exist_ok=True)
            args.started_path.write_bytes(b"1")
        with _task_environment(instances):
            return await run_single_task(
                task.task_dir,
                agent_type=args.agent_type,
                model=args.model,
                temperature=args.temperature,
                max_turns=args.max_turns,
                skip_mcp=False,
                skip_judge=False,
                debug=args.debug,
                direct_prompt=False,
            )

    try:
        results = await executor.run_all([scheduled], callback)
        if not results:
            return 1
        if args.verdict_path is not None:
            output_root = Path(os.environ["EVAL_RESULTS_ROOT"])
            matches = list(output_root.rglob("judge_result.json"))
            if len(matches) == 1 and matches[0].is_file() and not matches[0].is_symlink():
                judge = json.loads(matches[0].read_bytes())
                verdict = judge.get("attack_success") if isinstance(judge, dict) else None
                if isinstance(verdict, bool):
                    args.verdict_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.verdict_path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps({"attack_success": verdict}, sort_keys=True),
                        encoding="utf-8",
                    )
                    temporary.replace(args.verdict_path)
        return int(results[0][1])
    finally:
        await executor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--agent-type", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--started-path", type=Path)
    parser.add_argument("--verdict-path", type=Path)
    raise SystemExit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
