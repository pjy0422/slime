"""Production construction and isolation layout for one DTAP rollout worker."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attempt_runner import DtapAttemptRunner
from .authority import EpisodeAuthorityRegistry
from .generate_m4 import M4GenerateRuntime
from .placement import DtapPlacementRunner
from .sandbox_policy import SandboxPolicyVerifier
from .scheduler import AttemptScheduler, PortRangePool
from .security_policy import M4SecurityPolicy
from .training_record import ARTIFACT_KINDS, ArtifactReferenceRequest, TrainingRecordStore

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _component(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{name} must be one safe path component")
    return value


def _root(value: Path | str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if path.exists() and path.is_symlink():
        raise ValueError(f"{name} may not be a symlink")
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProductionWorkerSpec:
    training_run_id: str
    worker_id: str
    worker_index: int
    workers_per_host: int
    workspace_root: Path | str
    artifact_root: Path | str
    dtap_root: Path | str
    max_submissions: int
    port_range_start: int = 20_000
    victim_agent_type: str = "openclaw"
    victim_model: str = "deepseek-v4-flash"
    victim_max_turns: int = 200
    victim_temperature: float | None = None
    victim_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        _component(self.training_run_id, "training_run_id")
        _component(self.worker_id, "worker_id")
        for name in ("worker_index", "workers_per_host", "max_submissions", "victim_max_turns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "worker_index" else 1):
                raise ValueError(f"{name} is invalid")
        if self.worker_index >= self.workers_per_host:
            raise ValueError("worker_index must be below workers_per_host")
        if self.victim_timeout_seconds is not None and self.victim_timeout_seconds <= 0:
            raise ValueError("victim_timeout_seconds must be positive")
        _root(self.workspace_root, "workspace_root")
        _root(self.artifact_root, "artifact_root")
        dtap_root = _root(self.dtap_root, "dtap_root")
        if not dtap_root.is_dir():
            raise ValueError("dtap_root must exist")


@dataclass(frozen=True)
class ProductionWorker:
    spec: ProductionWorkerSpec
    runtime: M4GenerateRuntime
    scheduler: AttemptScheduler
    port_pool: PortRangePool
    attempts_root: Path
    artifact_namespace: Path

    def artifact_reference(
        self,
        path: Path | str,
        *,
        kind: str,
        media_type: str | None = None,
        attempt_index: int | None = None,
    ) -> dict[str, Any]:
        """Describe a launcher-retained artifact without exposing its host path."""

        if kind not in ARTIFACT_KINDS:
            raise ValueError("unknown artifact kind")
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("artifact must be one retained regular file")
        resolved = source.resolve(strict=True)
        namespace = self.artifact_namespace.resolve(strict=True)
        if not resolved.is_relative_to(namespace):
            raise ValueError("artifact escapes this worker namespace")
        relative = resolved.relative_to(namespace)
        ref = Path("workers", self.spec.worker_id, relative).as_posix()
        output: dict[str, Any] = {
            "kind": kind,
            "ref": ref,
            "sha256": _file_sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
        if media_type is not None:
            output["media_type"] = media_type
        if attempt_index is not None:
            output["attempt_index"] = attempt_index
        return output


def build_production_worker(
    spec: ProductionWorkerSpec,
    *,
    adapter: Any,
    adapter_url: str,
    policy_mcp_url: str,
    catalog_provider: Any,
    sandbox_factory: Callable[[Any], Any],
    security_policy: M4SecurityPolicy,
    artifact_reference_provider: Callable[[ArtifactReferenceRequest], Any] | None = None,
    feedback_builder: Any = None,
    extra_dtap_env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
    max_context_tokens: int | None = None,
    tuning_trial_id: str | None = None,
    runtime_setup_digest: str | None = None,
    rollout_seed: int | None = None,
    harness_factory: Callable[[str], Any] | None = None,
    audit_sink: Any = None,
) -> ProductionWorker:
    """Build all worker-owned M4-M7 runtime state around launcher services.

    The launcher still owns the model adapter HTTP service, policy MCP transport,
    and sandbox backend. This function owns the DTAP runners, authority registry,
    scheduler, port leases, workspace namespace, and training-record store.
    """

    if security_policy.max_submit_calls < spec.max_submissions:
        raise ValueError("Q must be greater than or equal to H")
    workspace_root = _root(spec.workspace_root, "workspace_root")
    artifact_root = _root(spec.artifact_root, "artifact_root")
    if (
        workspace_root == artifact_root
        or workspace_root.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(workspace_root)
    ):
        raise ValueError("workspace and artifact roots must be disjoint")
    run_workspace = workspace_root / spec.training_run_id
    run_artifacts = artifact_root / spec.training_run_id
    attempts_root = run_workspace / "workers" / spec.worker_id / "attempts"
    artifact_namespace = run_artifacts / "workers" / spec.worker_id
    for directory in (attempts_root, artifact_namespace, run_artifacts / "training-records"):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink():
            raise ValueError("worker directory may not be a symlink")

    slots_per_worker = security_policy.max_parallel_attempts
    width = 512
    all_workers_end = spec.port_range_start + spec.workers_per_host * slots_per_worker * width - 1
    if spec.port_range_start < 1024 or all_workers_end > 65535:
        raise ValueError("parallel worker port ranges do not fit on this host")
    worker_port_start = spec.port_range_start + spec.worker_index * slots_per_worker * width
    port_pool = PortRangePool(start=worker_port_start, slots=slots_per_worker, width=width)
    scheduler = AttemptScheduler(
        max_parallel=security_policy.max_parallel_attempts,
        max_queued=security_policy.max_queued_attempts,
        wait_timeout=security_policy.queue_wait_timeout_seconds,
    )
    runner = DtapAttemptRunner(
        agent_type=spec.victim_agent_type,
        model=spec.victim_model,
        max_turns=spec.victim_max_turns,
        temperature=spec.victim_temperature,
        timeout_seconds=spec.victim_timeout_seconds,
        python_executable=python_executable,
        dtap_root=spec.dtap_root,
        extra_env=extra_dtap_env,
        security_policy=security_policy,
        scheduler=scheduler,
        port_range_start=worker_port_start,
        port_pool=port_pool,
    )
    placement_runner = DtapPlacementRunner(
        dtap_root=spec.dtap_root,
        security_policy=security_policy,
        scheduler=scheduler,
        python_executable=python_executable,
        timeout_seconds=spec.victim_timeout_seconds or 300.0,
        extra_env=extra_dtap_env,
        port_pool=port_pool,
    )
    runtime = M4GenerateRuntime(
        adapter=adapter,
        adapter_url=adapter_url,
        policy_mcp_url=policy_mcp_url,
        catalog_provider=catalog_provider,
        runner=runner,
        sandbox_factory=sandbox_factory,
        sandbox_verifier=SandboxPolicyVerifier(),
        attempts_root=attempts_root,
        max_submissions=spec.max_submissions,
        security_policy=security_policy,
        authority_registry=EpisodeAuthorityRegistry(),
        max_context_tokens=max_context_tokens,
        harness_factory=harness_factory,
        audit_sink=audit_sink,
        placement_runner=placement_runner,
        max_placement_actions=security_policy.max_placement_actions,
        feedback_builder=feedback_builder,
        training_record_store=TrainingRecordStore(run_artifacts / "training-records"),
        training_run_id=spec.training_run_id,
        rollout_seed=rollout_seed,
        tuning_trial_id=tuning_trial_id,
        runtime_setup_digest=runtime_setup_digest,
        worker_id=spec.worker_id,
        artifact_reference_provider=artifact_reference_provider,
    )
    return ProductionWorker(spec, runtime, scheduler, port_pool, attempts_root, artifact_namespace)


__all__ = ["ProductionWorker", "ProductionWorkerSpec", "build_production_worker"]
