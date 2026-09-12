import json

import pytest
from examples.dtap_agent_rl.benchmark_manifest import matrix_cases
from examples.dtap_agent_rl.scripts.smoke_m6_domain_matrix import (
    _failure_class,
    _failure_count,
    _first_record,
    _matching_resume_result,
    _passed_payload,
    _selected_record,
    _stored_results,
    _task_dir,
)


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("UNSUPPORTED_PLACEMENT", "unsupported_placement"),
        ("PLACEMENT_MISMATCH", "placement"),
        ("judge process failed", "judge"),
        ("OpenClaw victim failed", "victim"),
        ("INVALID_SUBMISSION validation", "validation"),
        ("GLM did not produce a plan", "policy"),
        ("docker daemon unavailable", "infrastructure"),
    ],
)
def test_failure_classes_are_machine_separable(tail, expected):
    assert _failure_class({"status": "failed", "error_tail": tail}) == expected


def test_matrix_resolves_malicious_benchmark_record(tmp_path):
    record = {
        "domain": "travel",
        "type": "malicious",
        "threat_model": "indirect",
        "risk_category": "data-exfiltration",
        "task_id": "001",
    }
    benchmark = tmp_path / "indirect.jsonl"
    benchmark.write_text("\n" + json.dumps(record) + "\n", encoding="utf-8")

    assert _first_record(benchmark) == record
    assert _task_dir(tmp_path, record) == tmp_path.joinpath(
        "dataset",
        "travel",
        "malicious",
        "indirect",
        "data-exfiltration",
        "001",
    )


def test_selection_profiles_choose_disjoint_records(tmp_path):
    benchmark = tmp_path / "direct.jsonl"
    benchmark.write_text(
        "".join(json.dumps({"task_id": f"task-{index}"}) + "\n" for index in range(51)),
        encoding="utf-8",
    )

    release, release_index = _selected_record(benchmark, "release-v1")
    holdout, holdout_index = _selected_record(benchmark, "holdout-v1")

    assert (release_index, release["task_id"]) == (0, "task-0")
    assert (holdout_index, holdout["task_id"]) == (50, "task-50")
    assert release != holdout


def test_unavailable_selection_profile_index_fails_closed(tmp_path):
    benchmark = tmp_path / "direct.jsonl"
    benchmark.write_text('{"task_id":"only"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="holdout-v1.*index 50 is unavailable"):
        _selected_record(benchmark, "holdout-v1")


def test_resume_requires_the_same_profile_index_and_task(tmp_path):
    result_path = tmp_path / "result.json"
    result = {
        "status": "passed",
        "task_id": "task-50",
        "risk_category": "category-50",
        "selection_profile": "holdout-v1",
        "benchmark_index": 50,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert (
        _matching_resume_result(
            result_path,
            selection_profile="holdout-v1",
            benchmark_index=50,
            task_id="task-50",
            risk_category="category-50",
        )
        == result
    )
    assert (
        _matching_resume_result(
            result_path,
            selection_profile="release-v1",
            benchmark_index=0,
            task_id="task-0",
            risk_category="category-0",
        )
        is None
    )
    assert (
        _matching_resume_result(
            result_path,
            selection_profile="holdout-v1",
            benchmark_index=50,
            task_id="task-50",
            risk_category="different-category",
        )
        is None
    )


def test_manifest_generates_stable_default_and_explicit_vm_matrices():
    assert matrix_cases(("travel", "browser"), ("indirect",)) == (
        ("browser", "indirect"),
        ("travel", "indirect"),
    )
    assert matrix_cases(("windows", "macos"), ("direct",)) == (
        ("macos", "direct"),
        ("windows", "direct"),
    )


@pytest.mark.parametrize(
    ("domains", "threat_models"),
    [(("unknown",), ("direct",)), (("browser",), ("unsupported",))],
)
def test_manifest_rejects_unknown_matrix_coordinates(domains, threat_models):
    with pytest.raises(ValueError, match="unknown benchmark coordinates"):
        matrix_cases(domains, threat_models)


def test_matrix_extracts_passed_report_from_mixed_server_output():
    report = {"status": "passed", "attack_success": False, "environment_steps": 2}
    stdout = "server started\n" + json.dumps(report) + "\nserver stopped\n"

    assert _passed_payload(stdout) == report
    assert _passed_payload("server only") is None


def test_subset_resume_summary_can_collect_the_complete_result_tree(tmp_path):
    for domain, threat_model, status in (
        ("browser", "direct", "passed"),
        ("medical", "indirect", "failed"),
    ):
        path = tmp_path / domain / threat_model / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "domain": domain,
                    "threat_model": threat_model,
                    "status": status,
                }
            ),
            encoding="utf-8",
        )

    assert [item["status"] for item in _stored_results(tmp_path)] == ["passed", "failed"]


@pytest.mark.parametrize("domain", ["windows", "macos", "browser"])
def test_platform_name_does_not_hide_failures(domain):
    assert (
        _failure_count(
            [
                {"domain": domain, "status": "failed", "expected_platform_failure": True},
                {"domain": domain, "status": "passed"},
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "[]",
        "null",
        "{}",
        '{"domain":"browser","threat_model":"direct","status":"unknown"}',
        '{"domain":"medical","threat_model":"direct","status":"passed"}',
        '{"domain":"browser","threat_model":"indirect","status":"passed"}',
    ],
)
def test_invalid_stored_results_are_reported_as_failures(tmp_path, contents):
    path = tmp_path / "browser" / "direct" / "result.json"
    path.parent.mkdir(parents=True)
    path.write_text(contents, encoding="utf-8")
    results = _stored_results(tmp_path)
    assert results == [
        {
            "domain": "browser",
            "threat_model": "direct",
            "status": "failed",
            "result_error": "invalid stored result",
        }
    ]
    assert _failure_count(results) == 1


def test_absent_results_are_not_fabricated(tmp_path):
    assert _stored_results(tmp_path) == []
