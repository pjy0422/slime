"""CPU contracts for isolated DTAP production rollout workers."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# isort: off
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.worker_runtime import ProductionWorkerSpec, build_production_worker

# isort: on

NUM_GPUS = 0


@asynccontextmanager
async def sandbox_factory(_sample):
    yield object()


def build(tmp_path: Path, index: int, *, workers: int = 2, parallel: int = 2):
    dtap_root = tmp_path / "dtap"
    dtap_root.mkdir(exist_ok=True)
    policy = M4SecurityPolicy(
        max_submit_calls=3,
        max_parallel_attempts=parallel,
        max_queued_attempts=4,
    )
    spec = ProductionWorkerSpec(
        training_run_id="cpu-run",
        worker_id=f"worker-{index}",
        worker_index=index,
        workers_per_host=workers,
        workspace_root=tmp_path / "workspace",
        artifact_root=tmp_path / "artifacts",
        dtap_root=dtap_root,
        max_submissions=2,
    )
    return build_production_worker(
        spec,
        adapter=object(),
        adapter_url=f"http://adapter-{index}:18001",
        policy_mcp_url=f"http://mcp-{index}:18002",
        catalog_provider=object(),
        sandbox_factory=sandbox_factory,
        security_policy=policy,
        extra_dtap_env={"VICTIM_CREDENTIAL": f"credential-{index}"},
    )


def test_worker_construction_owns_shared_runtime_resources_and_disjoint_namespaces(tmp_path) -> None:
    first = build(tmp_path, 0)
    second = build(tmp_path, 1)

    assert first.runtime.runner.scheduler is first.scheduler
    assert first.runtime.placement_runner.scheduler is first.scheduler
    assert first.runtime.runner.port_pool is first.port_pool
    assert first.runtime.placement_runner.port_pool is first.port_pool
    assert first.runtime.authority_registry is not second.runtime.authority_registry
    assert first.attempts_root != second.attempts_root
    assert first.artifact_namespace != second.artifact_namespace
    assert first.runtime.training_record_store.root == second.runtime.training_record_store.root
    assert (first.port_pool.start, first.port_pool.end) == (20_000, 21_023)
    assert (second.port_pool.start, second.port_pool.end) == (21_024, 22_047)

    first_env = first.runtime.security_policy.build_dtap_child_env(
        host_env={"HOST_SECRET": "never-inherit"},
        explicit_env=first.runtime.runner.extra_env,
    )
    second_env = second.runtime.security_policy.build_dtap_child_env(
        host_env={"HOST_SECRET": "never-inherit"},
        explicit_env=second.runtime.runner.extra_env,
    )
    assert first_env["VICTIM_CREDENTIAL"] == "credential-0"
    assert second_env["VICTIM_CREDENTIAL"] == "credential-1"
    assert "HOST_SECRET" not in first_env and "HOST_SECRET" not in second_env


@pytest.mark.asyncio
async def test_parallel_worker_port_leases_never_overlap(tmp_path) -> None:
    workers = [build(tmp_path, index, workers=3) for index in range(3)]

    async def leases(worker):
        async with worker.port_pool.lease() as first:
            async with worker.port_pool.lease() as second:
                await asyncio.sleep(0)
                return first, second

    ranges = [item for pair in await asyncio.gather(*(leases(worker) for worker in workers)) for item in pair]
    occupied = [set(range(start, end + 1)) for start, end in ranges]
    assert all(left.isdisjoint(right) for index, left in enumerate(occupied) for right in occupied[index + 1 :])


def test_artifact_reference_is_digest_bound_and_cannot_cross_workers(tmp_path) -> None:
    first = build(tmp_path, 0)
    second = build(tmp_path, 1)
    artifact = first.artifact_namespace / "episodes" / "public" / "policy.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"event":"tool"}\n', encoding="utf-8")

    reference = first.artifact_reference(artifact, kind="policy_trajectory", media_type="application/jsonl")

    assert reference["ref"] == "workers/worker-0/episodes/public/policy.jsonl"
    assert reference["size_bytes"] == artifact.stat().st_size
    assert len(reference["sha256"]) == 64
    with pytest.raises(ValueError, match="escapes"):
        second.artifact_reference(artifact, kind="policy_trajectory")


def test_host_port_capacity_is_validated_before_worker_start(tmp_path) -> None:
    with pytest.raises(ValueError, match="do not fit"):
        build(tmp_path, 0, workers=100, parallel=2)


def test_workspace_and_retained_artifact_roots_may_not_overlap(tmp_path) -> None:
    dtap_root = tmp_path / "dtap"
    dtap_root.mkdir()
    shared = tmp_path / "shared"
    spec = ProductionWorkerSpec(
        training_run_id="cpu-run",
        worker_id="worker-0",
        worker_index=0,
        workers_per_host=1,
        workspace_root=shared,
        artifact_root=shared / "artifacts",
        dtap_root=dtap_root,
        max_submissions=1,
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        build_production_worker(
            spec,
            adapter=object(),
            adapter_url="http://adapter:18001",
            policy_mcp_url="http://mcp:18002",
            catalog_provider=object(),
            sandbox_factory=sandbox_factory,
            security_policy=M4SecurityPolicy(max_submit_calls=1),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
