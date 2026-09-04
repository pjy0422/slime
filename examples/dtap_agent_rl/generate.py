"""slime custom-generate entry point for M3 DTAP trajectories."""

from __future__ import annotations

import inspect
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .episode import load_task_snapshot
from .m3 import run_m3_episode
from .service import EpisodeRegistry
from .submission import EpisodeSubmissionRegistry


@dataclass
class M3GenerateRuntime:
    adapter: Any
    adapter_url: str
    catalog_provider: Any
    runner: Any
    sandbox_factory: Callable[[Any], Any]
    attempts_root: Path | str
    max_submissions: int
    view_registry: EpisodeRegistry
    submission_registry: EpisodeSubmissionRegistry
    max_context_tokens: int | None = None
    harness: Any = None
    snapshot_loader: Callable[..., Any] = load_task_snapshot


_RUNTIME: M3GenerateRuntime | None = None


def configure_runtime(runtime: M3GenerateRuntime) -> None:
    """Trusted process startup hook; call once before serving rollouts."""

    global _RUNTIME
    _RUNTIME = runtime


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _session_id(base_sample: Any) -> str:
    existing = getattr(base_sample, "session_id", None)
    if existing:
        return str(existing)
    return f"dtap-m3-{secrets.token_hex(16)}"


def _abort_sample(base_sample: Any, reason: str) -> list[Any]:
    base_sample.reward = 0.0
    base_sample.remove_sample = True
    status_enum = getattr(type(base_sample), "Status", None)
    if status_enum is not None and hasattr(status_enum, "ABORTED"):
        base_sample.status = status_enum.ABORTED
    base_sample.metadata = {
        **(getattr(base_sample, "metadata", None) or {}),
        "abort_reason": reason,
    }
    return [base_sample]


async def generate(
    args: Any,
    base_sample: Any,
    sampling_params: Mapping[str, Any],
    evaluation: bool = False,
) -> list[Any]:
    """Run one persistent policy context containing up to H DTAP submissions."""

    del args, evaluation
    runtime = _RUNTIME
    if runtime is None:
        return _abort_sample(base_sample, "m3_runtime_not_configured")
    metadata = getattr(base_sample, "metadata", None) or {}
    task_dir = metadata.get("task_dir")
    if not task_dir:
        return _abort_sample(base_sample, "missing_task_dir")

    session_id = _session_id(base_sample)
    base_sample.session_id = session_id
    session_opened = False
    finish_attempted = False
    try:
        await _maybe_await(
            runtime.adapter.open_session(
                session_id,
                sampling_defaults=dict(sampling_params),
                max_context_tokens=runtime.max_context_tokens,
            )
        )
        session_opened = True
        snapshot = runtime.snapshot_loader(task_dir)
        sandbox_context = runtime.sandbox_factory(base_sample)
        async with sandbox_context as sandbox:
            result = await run_m3_episode(
                snapshot=snapshot,
                catalog_provider=runtime.catalog_provider,
                view_registry=runtime.view_registry,
                submission_registry=runtime.submission_registry,
                runner=runtime.runner,
                attempts_root=runtime.attempts_root,
                max_submissions=runtime.max_submissions,
                sandbox=sandbox,
                session_id=session_id,
                adapter_url=runtime.adapter_url,
                prompt=str(metadata.get("prompt") or getattr(base_sample, "prompt", "")),
                workdir=str(metadata.get("workdir") or "/workspace/empty"),
                time_budget_sec=int(metadata.get("time_budget_sec") or 1800),
                harness=runtime.harness,
            )

        reward = result.runtime.final_reward
        finish_attempted = True
        samples = await runtime.adapter.finish_session(
            session_id,
            base_sample=base_sample,
            reward=float(reward if reward is not None else 0.0),
            extra_metadata={
                "dtap_status": result.runtime.status.value,
                "dtap_submissions_used": result.runtime.submissions_used,
                "dtap_max_submissions": result.runtime.max_submissions,
                "agent_exit_code": result.harness_return_code,
            },
        )
        if not samples:
            return _abort_sample(base_sample, "adapter_session_empty")
        if result.runtime.remove_sample:
            for sample in samples:
                _abort_sample(sample, "dtap_infrastructure_error")
        return list(samples)
    except Exception as exc:
        if not finish_attempted:
            finish_attempted = True
            try:
                samples = await runtime.adapter.finish_session(
                    session_id,
                    base_sample=base_sample,
                    reward=0.0,
                    extra_metadata={"dtap_status": "infra_error"},
                )
                if samples:
                    for sample in samples:
                        _abort_sample(sample, f"exception:{type(exc).__name__}")
                    return list(samples)
            except Exception:
                pass
        return _abort_sample(base_sample, f"exception:{type(exc).__name__}")
    finally:
        drop_session = getattr(runtime.adapter, "drop_session", None)
        if session_opened and drop_session is not None:
            await _maybe_await(drop_session(session_id, wait_timeout=30))
