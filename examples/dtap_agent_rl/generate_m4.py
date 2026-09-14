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
from .training_record import (
    ArtifactReferenceRequest,
    TrainingRecordContext,
    TrainingRecordStore,
    build_record,
    restore_samples,
    task_reference,
    validate_artifact_references,
)


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
    feedback_builder: Any = None
    training_record_store: TrainingRecordStore | None = None
    training_run_id: str | None = None
    resume_training_records: bool = True
    rollout_seed: int | None = None
    tuning_trial_id: str | None = None
    runtime_setup_digest: str | None = None
    worker_id: str | None = None
    artifact_reference_provider: Callable[[ArtifactReferenceRequest], Any] | None = None


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
    if runtime.training_record_store is not None and not runtime.training_run_id:
        raise ValueError("training_run_id is required when training records are enabled")
    if runtime.artifact_reference_provider is not None and (
        runtime.training_record_store is None or not runtime.worker_id
    ):
        raise ValueError("artifact references require a training record store and worker_id")
    if runtime.placement_runner is not None:
        runner_scheduler = getattr(runtime.runner, "scheduler", None)
        placement_scheduler = getattr(runtime.placement_runner, "scheduler", None)
        if (
            runner_scheduler is not None
            and placement_scheduler is not None
            and runner_scheduler is not placement_scheduler
        ):
            raise ValueError("victim and placement runners must share one scheduler")
        runner_ports = getattr(runtime.runner, "port_pool", None)
        placement_ports = getattr(runtime.placement_runner, "port_pool", None)
        if runner_ports is not None and placement_ports is not None and runner_ports is not placement_ports:
            raise ValueError("victim and placement runners must share one port pool")
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


def _record_context(
    runtime: M4GenerateRuntime,
    metadata: Mapping[str, Any],
    *,
    public_episode_id: str,
) -> TrainingRecordContext:
    return TrainingRecordContext(
        training_run_id=str(runtime.training_run_id),
        task_ref=task_reference(metadata),
        public_episode_id=public_episode_id,
        seed=runtime.rollout_seed,
        tuning_trial_id=runtime.tuning_trial_id,
        runtime_setup_digest=runtime.runtime_setup_digest,
        feedback_mode=str(metadata["feedback_mode"]) if metadata.get("feedback_mode") is not None else None,
        hierarchy_mode=str(metadata["hae_policy_mode"]) if metadata.get("hae_policy_mode") is not None else None,
        worker_id=runtime.worker_id,
    )


def _reproduction(runtime: M4GenerateRuntime, sampling_params: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        key: sampling_params[key]
        for key in ("temperature", "top_p", "top_k", "max_new_tokens", "sampling_seed")
        if key in sampling_params and isinstance(sampling_params[key], (int, float, bool, type(None)))
    }
    return {
        "rollout_seed": runtime.rollout_seed,
        "sampling": allowed,
        "tuning_trial_id": runtime.tuning_trial_id,
        "runtime_setup_digest": runtime.runtime_setup_digest,
    }


async def _record_episode_summary(
    runtime: M4GenerateRuntime,
    *,
    result: Any,
    record_id: str,
    session_id: str,
) -> dict[str, Any]:
    summary = dict(getattr(result, "record_summary", None) or {})
    provider = runtime.artifact_reference_provider
    if provider is None:
        summary["artifacts"] = []
        return summary
    request = ArtifactReferenceRequest(
        training_run_id=str(runtime.training_run_id),
        worker_id=str(runtime.worker_id),
        record_id=record_id,
        public_episode_id=result.public_episode_id,
        private_adapter_session_id=session_id,
    )
    supplied = await _maybe_await(provider(request))
    summary["artifacts"] = validate_artifact_references(supplied, worker_id=runtime.worker_id)
    return summary


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
        record_store = runtime.training_record_store
        record_id = None
        if record_store is not None:
            provisional_context = _record_context(runtime, input_metadata, public_episode_id="pending")
            record_id = record_store.record_id(provisional_context, base_sample)
            if runtime.resume_training_records:
                prior = record_store.load(record_id)
                if prior is not None:
                    restored = restore_samples(prior)
                    if restored:
                        return restored
                    return _abort_sample(base_sample, f"resumed_{prior['outcome']['eligibility_reason']}")
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
                feedback_builder=runtime.feedback_builder,
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
            samples = _abort_sample(base_sample, "session_unavailable")
            if record_store is not None and record_id is not None:
                context = _record_context(runtime, input_metadata, public_episode_id=result.public_episode_id)
                record_store.write(
                    build_record(
                        record_id=record_id,
                        context=context,
                        status="infra_error",
                        failure_class="session_unavailable",
                        samples=samples,
                        reproduction=_reproduction(runtime, sampling_params),
                    )
                )
            return samples
        if result.runtime.remove_sample:
            for sample in samples:
                _abort_sample(sample, "evaluation_unavailable")
        if record_store is not None and record_id is not None:
            failure_class = getattr(result, "failure_class", None)
            if failure_class is None and result.runtime.remove_sample:
                failure_class = result.runtime.infrastructure_stage
            context = _record_context(runtime, input_metadata, public_episode_id=result.public_episode_id)
            episode_summary = await _record_episode_summary(
                runtime,
                result=result,
                record_id=record_id,
                session_id=session_id,
            )
            record_store.write(
                build_record(
                    record_id=record_id,
                    context=context,
                    status=result.runtime.status.value,
                    failure_class=failure_class,
                    samples=samples,
                    reproduction=_reproduction(runtime, sampling_params),
                    episode=episode_summary,
                )
            )
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
                    if record_store is not None and record_id is not None:
                        try:
                            context = _record_context(runtime, input_metadata, public_episode_id="unavailable")
                            record_store.write(
                                build_record(
                                    record_id=record_id,
                                    context=context,
                                    status="infra_error",
                                    failure_class="evaluation_unavailable",
                                    samples=samples,
                                    reproduction=_reproduction(runtime, sampling_params),
                                )
                            )
                        except Exception:
                            pass
                    return list(samples)
            except Exception:
                pass
        samples = _abort_sample(base_sample, "evaluation_unavailable")
        if record_store is not None and record_id is not None:
            try:
                context = _record_context(runtime, input_metadata, public_episode_id="unavailable")
                record_store.write(
                    build_record(
                        record_id=record_id,
                        context=context,
                        status="infra_error",
                        failure_class="evaluation_unavailable",
                        samples=samples,
                        reproduction=_reproduction(runtime, sampling_params),
                    )
                )
            except Exception:
                pass
        return samples
    finally:
        drop = getattr(runtime.adapter, "drop_session", None)
        if session_opened and drop is not None:
            await _maybe_await(drop(session_id, wait_timeout=30))
