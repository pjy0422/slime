"""slime custom-generate entry point for hardened M4 trajectories."""

from __future__ import annotations

import inspect
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authority import EpisodeAuthorityRegistry
from .episode import load_task_snapshot
from .m4 import run_m4_episode
from .sandbox_policy import SandboxPolicyVerifier
from .security_policy import M4SecurityPolicy


@dataclass
class M4GenerateRuntime:
    adapter: Any
    adapter_url: str
    policy_mcp_url: str
    catalog_provider: Any
    runner: Any
    sandbox_factory: Callable[[Any], Any]
    sandbox_verifier: SandboxPolicyVerifier
    attempts_root: Path | str
    max_submissions: int
    security_policy: M4SecurityPolicy
    authority_registry: EpisodeAuthorityRegistry
    max_context_tokens: int | None = None
    harness_factory: Callable[[str], Any] | None = None
    snapshot_loader: Callable[..., Any] = load_task_snapshot
    audit_sink: Any = None
    placement_runner: Any = None
    max_placement_actions: int | None = None


_RUNTIME: M4GenerateRuntime | None = None


def configure_runtime(runtime: M4GenerateRuntime) -> None:
    global _RUNTIME
    if runtime.security_policy.max_submit_calls < runtime.max_submissions:
        raise ValueError("Q must be greater than or equal to H")
    if getattr(runtime.runner, "m4_hardened", False) is not True:
        raise ValueError("M4 runtime requires a hardened attempt runner")
    runner_policy = getattr(runtime.runner, "security_policy", None)
    if runner_policy is not None and runner_policy != runtime.security_policy:
        raise ValueError("M4 runner and generate security policies differ")
    _RUNTIME = runtime


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _session_id(base_sample: Any) -> str:
    existing = getattr(base_sample, "session_id", None)
    return str(existing) if existing else f"dtap-m4-{secrets.token_hex(16)}"


def _safe_metadata(base_sample: Any, **updates: Any) -> None:
    original = getattr(base_sample, "metadata", None) or {}
    allowed_input = {
        key: original[key]
        for key in ("sample_id", "index", "task_id", "domain")
        if key in original and isinstance(original[key], (str, int, float, bool, type(None)))
    }
    base_sample.metadata = {**allowed_input, **updates}


def _abort_sample(base_sample: Any, reason: str) -> list[Any]:
    base_sample.reward = 0.0
    base_sample.remove_sample = True
    status_enum = getattr(type(base_sample), "Status", None)
    if status_enum is not None and hasattr(status_enum, "ABORTED"):
        base_sample.status = status_enum.ABORTED
    _safe_metadata(base_sample, m4_schema="v1", abort_reason=reason)
    return [base_sample]


async def generate(
    args: Any,
    base_sample: Any,
    sampling_params: Mapping[str, Any],
    evaluation: bool = False,
) -> list[Any]:
    del args, evaluation
    runtime = _RUNTIME
    if runtime is None:
        return _abort_sample(base_sample, "runtime_unavailable")
    input_metadata = dict(getattr(base_sample, "metadata", None) or {})
    task_dir = input_metadata.get("task_dir")
    if not task_dir:
        return _abort_sample(base_sample, "task_unavailable")
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
        async with runtime.sandbox_factory(base_sample) as sandbox:
            result = await run_m4_episode(
                snapshot=snapshot,
                catalog_provider=runtime.catalog_provider,
                authority_registry=runtime.authority_registry,
                runner=runtime.runner,
                attempts_root=runtime.attempts_root,
                max_submissions=runtime.max_submissions,
                security_policy=runtime.security_policy,
                sandbox_verifier=runtime.sandbox_verifier,
                sandbox=sandbox,
                adapter_session_id=session_id,
                adapter_url=runtime.adapter_url,
                policy_mcp_url=runtime.policy_mcp_url,
                prompt=str(input_metadata.get("prompt") or getattr(base_sample, "prompt", "")),
                workdir=str(input_metadata.get("workdir") or "/workspace/empty"),
                time_budget_sec=int(input_metadata.get("time_budget_sec") or 1800),
                harness_factory=runtime.harness_factory,
                audit_sink=runtime.audit_sink,
                placement_runner=runtime.placement_runner,
                max_placement_actions=runtime.max_placement_actions,
            )
        _safe_metadata(
            base_sample,
            m4_schema="v1",
            dtap_status=result.runtime.status.value,
            submissions_used=result.runtime.submissions_used,
            max_submissions=result.runtime.max_submissions,
            submit_calls=result.runtime.submit_calls,
            max_submit_calls=result.runtime.max_submit_calls,
        )
        finish_attempted = True
        samples = await runtime.adapter.finish_session(
            session_id,
            base_sample=base_sample,
            reward=float(result.runtime.final_reward or 0.0),
            extra_metadata=dict(base_sample.metadata),
        )
        if not samples:
            return _abort_sample(base_sample, "session_unavailable")
        if result.runtime.remove_sample:
            for sample in samples:
                _abort_sample(sample, "evaluation_unavailable")
        return list(samples)
    except Exception:
        if session_opened and not finish_attempted:
            finish_attempted = True
            _safe_metadata(base_sample, m4_schema="v1", dtap_status="unavailable")
            try:
                samples = await runtime.adapter.finish_session(
                    session_id,
                    base_sample=base_sample,
                    reward=0.0,
                    extra_metadata=dict(base_sample.metadata),
                )
                if samples:
                    for sample in samples:
                        _abort_sample(sample, "evaluation_unavailable")
                    return list(samples)
            except Exception:
                pass
        return _abort_sample(base_sample, "evaluation_unavailable")
    finally:
        drop = getattr(runtime.adapter, "drop_session", None)
        if session_opened and drop is not None:
            await _maybe_await(drop(session_id, wait_timeout=30))
