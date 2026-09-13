from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

VIEWER_ROOT = Path(__file__).resolve().parents[1] / "tools" / "dtap-trajectory-viewer"
if str(VIEWER_ROOT) not in sys.path:
    sys.path.insert(0, str(VIEWER_ROOT))

from dtap_traj.cli import main
from dtap_traj.db import TrajectoryDB
from dtap_traj.server import create_app
from dtap_traj.tuning import (
    TuningArtifactError,
    create_tuning_trial,
    index_tuning_root,
    load_tuning_bundle,
    rank_successful_trials,
    update_tuning_trial,
)

NUM_GPUS = 0


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def profiles(tmp_path: Path, *, hardware: str = "cpu-fixture") -> dict[str, Path]:
    source = tmp_path / "profiles" / hardware
    return {
        "hardware": write_json(
            source / "hardware.json",
            {"label": hardware, "accelerator_count": 0, "synthetic": True},
        ),
        "model": write_json(source / "model.json", {"label": "tiny-model", "parameters": 10}),
        "workload": write_json(
            source / "workload.json",
            {"label": "dtap-cpu-contract", "prompt_tokens_p95": 128, "turns_p95": 2},
        ),
        "software": write_json(source / "software.json", {"slime_commit": "fixture", "cuda": None}),
    }


def create_trial(
    tmp_path: Path,
    trial_id: str,
    *,
    tp: int,
    objective: float | None,
    peak_vram_gb: float | None = None,
    status: str = "success",
    hardware: str = "cpu-fixture",
) -> Path:
    profile = profiles(tmp_path, hardware=hardware)
    config = write_json(
        tmp_path / "configs" / f"{trial_id}.json",
        {
            "sglang": {"tp": tp, "replicas": 1, "concurrency": 4},
            "megatron": {"tp": 1, "pp": 1, "cp": 1, "ep": 1, "etp": 1},
            "allocation": {"train_gpus": 0, "rollout_gpus": 0},
        },
    )
    trial = create_tuning_trial(
        tmp_path / "artifacts",
        phase="sglang",
        config=config,
        trial_id=trial_id,
        hypothesis=f"compare TP={tp}",
        parent_trial_ids=["baseline"] if tp > 1 else [],
        tags=["synthetic-cpu"],
        **profile,
    )
    if status == "success":
        metrics = write_json(
            tmp_path / "metrics" / f"{trial_id}.json",
            {
                "rollout_tok_s": objective,
                "peak_vram_gb": peak_vram_gb,
                "wall_time_s": 10,
                "gpu_seconds": 0,
                "repeats": 3,
                "durations_s": {"generation": 6, "victim": 2, "judge": 1, "idle": 1},
            },
        )
        update_tuning_trial(
            trial,
            status="success",
            metrics=metrics,
            observation="synthetic contract measurement",
            next_step="review another manually selected candidate",
        )
    elif status != "planned":
        update_tuning_trial(
            trial,
            status=status,
            failure_class=status,
            observation="synthetic failure boundary",
            next_step="reduce the candidate resource requirement",
        )
    return trial


def test_manual_trial_lifecycle_and_idempotent_index(tmp_path):
    trial = create_trial(tmp_path, "tune-tp1", tp=1, objective=120.0, peak_vram_gb=10.0)
    bundle = load_tuning_bundle(trial)
    assert bundle["result"]["status"] == "success"
    assert bundle["manifest"]["hypothesis"] == "compare TP=1"

    db = TrajectoryDB(tmp_path / "index.sqlite3")
    first = index_tuning_root(tmp_path / "artifacts", db)
    assert first == {
        "root": str((tmp_path / "artifacts").resolve()),
        "scanned": 1,
        "updated": 1,
        "errors": [],
    }
    assert index_tuning_root(tmp_path / "artifacts", db)["updated"] == 0
    item = db.get_tuning_trial("tune-tp1")
    assert item is not None
    assert item["sglang_tp"] == 1
    assert item["objective_name"] == "rollout_tok_s"
    assert item["objective_value"] == 120.0
    assert item["tags"] == ["synthetic-cpu"]
    assert len(item["config_digest"]) == 64
    update_tuning_trial(trial, status="running", observation="keep this observation")
    update_tuning_trial(trial, status="success", next_step="record another candidate")
    preserved = load_tuning_bundle(trial)["result"]
    assert preserved["observation"] == "keep this observation"
    assert preserved["next_step"] == "record another candidate"


def test_failed_trials_are_retained_and_ranking_is_cohort_safe(tmp_path):
    create_trial(tmp_path, "fast", tp=1, objective=150, peak_vram_gb=14)
    create_trial(tmp_path, "efficient", tp=2, objective=140, peak_vram_gb=8)
    create_trial(tmp_path, "dominated", tp=4, objective=100, peak_vram_gb=16)
    create_trial(tmp_path, "oom-boundary", tp=8, objective=None, status="oom")
    db = TrajectoryDB(tmp_path / "rank.sqlite3")
    assert index_tuning_root(tmp_path / "artifacts", db)["updated"] == 4
    assert db.list_tuning_trials(status="oom")["items"][0]["failure_class"] == "oom"

    ranked = rank_successful_trials(db.list_tuning_trials(status="success")["items"])
    assert ranked["recommended"]["trial_id"] == "fast"
    assert {item["trial_id"] for item in ranked["pareto"]} == {"fast", "efficient"}

    create_trial(
        tmp_path,
        "different-host",
        tp=1,
        objective=999,
        peak_vram_gb=4,
        hardware="other-cpu-fixture",
    )
    assert index_tuning_root(tmp_path / "artifacts", db)["updated"] == 1
    mixed = rank_successful_trials(db.list_tuning_trials(status="success")["items"])
    assert mixed["recommended"] is None
    assert "matching hardware" in mixed["reason"]


def test_performance_api_filters_details_compare_and_recommendations(tmp_path):
    create_trial(tmp_path, "api-tp1", tp=1, objective=100, peak_vram_gb=10)
    create_trial(tmp_path, "api-tp2", tp=2, objective=125, peak_vram_gb=12)
    client = TestClient(create_app(tmp_path / "artifacts", db_path=tmp_path / "api.sqlite3"))

    facets = client.get("/api/tuning/facets").json()
    assert facets["total"] == 2
    assert facets["successful"] == 2
    trials = client.get("/api/tuning/trials", params={"phase": "sglang"}).json()
    assert trials["total"] == 2
    fingerprint = trials["items"][0]["hardware_fingerprint"]
    assert client.get("/api/tuning/trials", params={"hardware_fingerprint": fingerprint}).json()["total"] == 2

    detail = client.get("/api/tuning/trials/api-tp1").json()
    assert detail["config"]["sglang"]["tp"] == 1
    assert detail["metrics"]["durations_s"]["generation"] == 6
    comparison = client.get(
        "/api/tuning/compare",
        params=[("trial_id", "api-tp1"), ("trial_id", "api-tp2")],
    ).json()
    assert "sglang.tp" in comparison["varying_config_keys"]
    recommendation = client.get(
        "/api/tuning/recommendations",
        params={"phase": "sglang", "hardware_fingerprint": fingerprint},
    ).json()
    assert recommendation["recommended"]["trial_id"] == "api-tp2"

    outside = create_trial(tmp_path / "outside", "outside", tp=1, objective=1)
    outside_item = TrajectoryDB(tmp_path / "outside.sqlite3")
    index_tuning_root(tmp_path / "outside" / "artifacts", outside_item)
    injected = outside_item.get_tuning_trial("outside")
    assert injected is not None and Path(injected["artifact_path"]) == outside
    client.app.state.db.upsert_tuning_trial(injected)
    assert client.get("/api/tuning/trials/outside").status_code == 404


def test_cli_creates_and_updates_trial_without_running_a_sweep(tmp_path, capsys):
    profile = profiles(tmp_path)
    config = write_json(tmp_path / "config.json", {"sglang": {"tp": 1}})
    root = tmp_path / "cli-artifacts"
    args = ["tune", "create", str(root), "--phase", "sglang", "--config", str(config)]
    for name, path in profile.items():
        args.extend([f"--{name}", str(path)])
    args.extend(["--trial-id", "cli-trial", "--hypothesis", "manual candidate"])
    assert main(args) == 0
    assert "created planned tuning trial" in capsys.readouterr().out
    trial = root / "tuning" / "cli-trial"
    assert json.loads((trial / "result.json").read_text())["status"] == "planned"

    metrics = write_json(tmp_path / "cli-metrics.json", {"rollout_tok_s": 42})
    assert (
        main(
            [
                "tune",
                "update",
                str(trial),
                "--status",
                "success",
                "--metrics",
                str(metrics),
                "--observation",
                "reviewed",
                "--next-step",
                "try TP=2",
            ]
        )
        == 0
    )
    assert load_tuning_bundle(trial)["result"]["next_step"] == "try TP=2"


def test_invalid_artifacts_fail_closed_and_success_requires_metrics(tmp_path):
    trial = create_trial(tmp_path, "planned", tp=1, objective=None, status="planned")
    with pytest.raises(TuningArtifactError, match="requires metrics"):
        update_tuning_trial(trial, status="success")
    manifest = json.loads((trial / "tuning-manifest.json").read_text())
    manifest["schema_version"] = 99
    write_json(trial / "tuning-manifest.json", manifest)
    db = TrajectoryDB(tmp_path / "invalid.sqlite3")
    indexed = index_tuning_root(tmp_path / "artifacts", db)
    assert indexed["updated"] == 0
    assert "unsupported tuning schema" in indexed["errors"][0]["error"]


def test_trial_creation_rejects_credentials_before_creating_artifacts(tmp_path):
    profile = profiles(tmp_path)
    config = write_json(tmp_path / "secret-config.json", {"api_key": "do-not-record"})
    with pytest.raises(TuningArtifactError, match="forbidden credential"):
        create_tuning_trial(
            tmp_path / "artifacts",
            phase="sglang",
            config=config,
            trial_id="secret-trial",
            **profile,
        )
    assert not (tmp_path / "artifacts" / "tuning" / "secret-trial").exists()


def test_duplicate_trial_ids_do_not_silently_replace_indexed_artifacts(tmp_path):
    create_trial(tmp_path / "collection" / "a", "duplicate", tp=1, objective=10)
    create_trial(tmp_path / "collection" / "b", "duplicate", tp=2, objective=20)
    db = TrajectoryDB(tmp_path / "duplicate.sqlite3")
    indexed = index_tuning_root(tmp_path / "collection", db)
    assert indexed["scanned"] == 2
    assert indexed["updated"] == 1
    assert "duplicate trial_id" in indexed["errors"][0]["error"]


def test_performance_workspace_assets_and_gpu_deferral_are_documented():
    web = VIEWER_ROOT / "dtap_traj" / "web"
    javascript = (web / "app.js").read_text()
    css = (web / "app.css").read_text()
    readme = (VIEWER_ROOT / "README.md").read_text()
    plan = (
        Path(__file__).resolve().parents[1] / "examples" / "dtap_agent_rl" / "M8_IMPLEMENTATION_PLAN.md"
    ).read_text()
    assert 'data-mode="performance"' in javascript
    assert "/api/tuning/compare" in javascript
    assert ".breakdown" in css
    assert "synthetic CPU fixtures only" in readme
    assert "no usable GPU" in plan
    assert "automatic sweeper" in plan


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
