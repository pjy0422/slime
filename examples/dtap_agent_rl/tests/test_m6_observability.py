import json
import os
from pathlib import Path

from examples.dtap_agent_rl.benchmark_manifest import (
    ALL_CASES,
    DEFAULT_CASES,
    DOMAIN_ENTRIES,
    MANIFEST_SHA256,
    THREAT_MODELS,
)
from examples.dtap_agent_rl.scripts.audit_m6_adapter_coverage import audit
from examples.dtap_agent_rl.scripts.smoke_m6_domain_matrix import (
    DOMAINS,
    EXCLUDED_PLATFORM_DOMAINS,
    _summary_metrics,
)
from examples.dtap_agent_rl.scripts.verify_m6_runtime_lock import verify


def _dtap_root() -> Path:
    default = Path(__file__).resolve().parents[3].parent / "DecodingTrust-Agent"
    return Path(os.environ.get("DTAP_ROOT", default)).resolve()


def test_default_matrix_keeps_vm_platforms_opt_in():
    assert EXCLUDED_PLATFORM_DOMAINS == {"macos", "windows"}
    assert "macos" not in DOMAINS
    assert "windows" not in DOMAINS
    assert len(DOMAINS) == 12
    assert THREAT_MODELS == ("direct", "indirect")
    assert len(ALL_CASES) == 28
    assert len(DEFAULT_CASES) == 24
    assert len(MANIFEST_SHA256) == 64
    assert {
        entry["name"] for entry in DOMAIN_ENTRIES if entry["vm_backed"]
    } == EXCLUDED_PLATFORM_DOMAINS
    assert all(
        not entry["default_enabled"]
        for entry in DOMAIN_ENTRIES if entry["vm_backed"]
    )


def test_manifest_coordinates_exist_in_the_dtap_benchmark_inventory():
    root = _dtap_root() / "benchmark"
    missing = [
        f"{domain}/{threat_model}.jsonl"
        for domain, threat_model in ALL_CASES
        if not (root / domain / f"{threat_model}.jsonl").is_file()
    ]
    assert missing == []


def test_summary_separates_completion_reward_and_placement_coverage():
    metrics = _summary_metrics([
        {
            "status": "passed", "evaluation_completed": True,
            "attack_success": False, "placement_applicable": False,
            "placement_covered": False, "placement_actions": 0,
            "placements_verified": 0, "environment_tools": [],
            "action_applied": True, "domain": "legal", "threat_model": "direct",
        },
        {
            "status": "passed", "evaluation_completed": True,
            "attack_success": True, "placement_applicable": True,
            "placement_covered": True, "placement_actions": 1,
            "placements_verified": 1,
            "environment_tools": ["gmail-injection:inject_email"],
            "action_applied": True, "domain": "legal", "threat_model": "indirect",
        },
        {"status": "failed", "failure_class": "infrastructure"},
    ])

    assert metrics["evaluation_completed"] == 2
    assert metrics["attack_successes"] == 1
    assert metrics["action_applied"] == 2
    assert metrics["placement_applicable"] == 1
    assert metrics["placement_covered"] == 1
    assert metrics["failures_by_class"]["infrastructure"] == 1
    assert metrics["placement_by_tool"]["gmail-injection:inject_email"] == {
        "attempted": 1, "verified": 1,
    }
    assert metrics["placement_by_injection_mcp"]["gmail-injection"] == {
        "attempted": 1, "verified": 1,
    }
    assert metrics["placement_by_domain"]["legal"] == {
        "evaluations": 2, "applicable": 1, "covered": 1,
    }
    assert metrics["placement_by_threat_model"]["indirect"] == {
        "evaluations": 1, "applicable": 1, "covered": 1,
    }


def test_enabled_adapter_inventory_is_consistent():
    result = audit(_dtap_root())
    assert result["status"] == "passed"
    assert result["missing_implementations"] == []
    assert result["verified_mutators"] == 128
    assert result["guest_platforms"]["windows-injection"]["verified"] == 7
    assert result["guest_platforms"]["macos-injection"]["verified"] == 6
    assert result["classified_non_placement_tools"] == 49
    assert result["unclassified_tools"] == 0
    assert result["unsupported_mutators"] == 0
    assert all(not row["registry_overlap"] for row in result["servers"])


def test_runtime_lock_schema_and_non_image_checks():
    root = Path(__file__).parents[1]
    lock = root / "dtap_integration/runtime-lock.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert all("@sha256:" in value for value in payload["container_images"].values())
    assert set(payload["guest_disks"]) == {"windows", "macos"}
    assert all(
        len(file["sha256"]) == 64 and file["size"] > 1_000_000_000
        for guest in payload["guest_disks"].values()
        for file in guest["files"].values()
    )
    assert payload["system_tools"]["jq"]["version"] == "jq-1.8.2"
    assert len(payload["system_tools"]["jq"]["sha256"]) == 64
    result = verify(
        lock, _dtap_root(),
        check_images=False,
    )
    assert result["status"] == "passed"
