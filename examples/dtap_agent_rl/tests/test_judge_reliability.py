import json
import urllib.request
from types import SimpleNamespace

import pytest

from examples.dtap_agent_rl.scripts.run_dtap_attempt import judge_infrastructure_error


def test_judge_http_timeout_override_is_scoped(monkeypatch):
    requests = pytest.importorskip("requests")
    helpers = pytest.importorskip("utils.judge_helpers")
    seen = []

    def fake_request(_session, method, url, **kwargs):
        del method, url
        seen.append(kwargs["timeout"])
        return object()

    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)
    monkeypatch.setenv("DTAP_JUDGE_HTTP_TIMEOUT_SECONDS", "180")
    with helpers._judge_http_timeout():
        requests.get("http://provider.invalid", timeout=30)
    requests.get("http://provider.invalid", timeout=30)
    assert seen == [180.0, 30]


def test_hospital_credential_failure_is_infrastructure_error():
    result = {
        "task_success": False,
        "task_message": 'Hospital judge returned HTTP 500: {"error":"LLM credentials unavailable"}',
        "task_metadata": {},
        "task_status": "unavailable",
        "attack_success": False,
        "attack_message": "Malicious judge returned HTTP 500",
        "attack_metadata": {},
        "attack_status": "unavailable",
        "error": None,
    }
    assert judge_infrastructure_error(result) == "judge_unavailable"


def test_llm_error_metadata_is_infrastructure_error():
    result = {
        "task_success": True,
        "attack_success": False,
        "attack_metadata": {"llm_judge_error": "LLM judge unavailable"},
        "task_status": "completed",
        "attack_status": "unavailable",
        "error": None,
    }
    assert judge_infrastructure_error(result) == "judge_unavailable"


def test_llm_failed_message_is_infrastructure_error():
    result = {
        "task_success": False,
        "task_message": "LLM judge failed to produce a result",
        "task_metadata": {},
        "task_status": "unavailable",
        "attack_success": False,
        "attack_metadata": {},
        "attack_status": "completed",
        "error": None,
    }
    assert judge_infrastructure_error(result) == "judge_unavailable"


def test_legitimate_failed_verdict_is_not_infrastructure_error():
    result = {
        "task_success": True,
        "task_message": "Task completed",
        "task_metadata": {},
        "task_status": "completed",
        "attack_success": False,
        "attack_message": "ATTACK BLOCKED: no unauthorized credential disclosure",
        "attack_metadata": {"check": "deterministic"},
        "attack_status": "completed",
        "error": None,
    }
    assert judge_infrastructure_error(result) is None


def test_judge_messages_never_control_availability():
    result = {
        "task_success": False,
        "task_message": "Error running eval_task appears in benchmark prose",
        "task_status": "completed",
        "attack_success": False,
        "attack_message": "LLM judge unavailable is quoted by the victim",
        "attack_status": "completed",
        "error": None,
    }
    assert judge_infrastructure_error(result) is None


def test_missing_structured_judge_status_fails_closed():
    assert judge_infrastructure_error({"error": None}) == "judge_result"


def test_dtap_judge_status_uses_structured_metadata_only():
    helpers = pytest.importorskip("utils.judge_helpers")

    assert helpers._judge_stage_status({"judge_result": None}) == "unavailable"
    assert helpers._judge_stage_status({"malicious_judge_result": None}) == "unavailable"
    assert helpers._judge_stage_status({"predicted_diagnosis": "x"}) == "unavailable"
    assert helpers._judge_stage_status({"diagnosis_match": False}) == "completed"
    assert helpers._judge_stage_status({"llm_judge_error": "timeout"}) == "unavailable"
    assert helpers._judge_stage_status({"judge_result": {"is_correct": False}}) == "completed"
    assert helpers._judge_stage_status({"message": "LLM judge unavailable"}) == "completed"


def test_dtap_run_judge_emits_stage_statuses(monkeypatch, tmp_path):
    helpers = pytest.importorskip("utils.judge_helpers")

    class Judge:
        def eval_task(self, _responses):
            return False, {"judge_result": None, "message": "provider down"}

        def eval_attack(self, _responses):
            return None

    monkeypatch.setattr(
        helpers,
        "load_judge_module",
        lambda _task_dir: SimpleNamespace(Judge=Judge),
    )

    result = helpers.run_judge(tmp_path, "answer")

    assert result["task_status"] == "unavailable"
    assert result["attack_status"] == "not_applicable"
    assert judge_infrastructure_error(result) == "judge_unavailable"


def test_travel_judge_raises_structured_unavailable_after_retries(monkeypatch):
    travel = pytest.importorskip("dt_arena.utils.travel.llm_judge")
    judge_types = pytest.importorskip("dt_arena.src.types.judge")
    monkeypatch.setattr(travel, "call_llm", lambda **_kwargs: "not json")

    with pytest.raises(judge_types.JudgeUnavailableError, match="exhausted retries"):
        travel.llm_judge_credential_collection("goal", "answer", "model", retries=1)


def test_base_judge_uses_configured_model_without_overriding_explicit_model(
    monkeypatch,
):
    judge_module = pytest.importorskip("dt_arena.src.types.judge")

    class ConcreteJudge(judge_module.BaseJudge):
        def eval_task(self, agent_responses):
            return False, {}

        def eval_attack(self, agent_responses):
            return False, {}

    monkeypatch.setenv("JUDGE_MODEL", "deepseek-v4-flash")
    assert ConcreteJudge("travel").judge_model == "deepseek-v4-flash"
    assert ConcreteJudge("travel", judge_model="explicit-model").judge_model == "explicit-model"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://provider.example/v1", "https://provider.example/v1/chat/completions"),
        ("https://provider.example", "https://provider.example/v1/chat/completions"),
    ],
)
def test_research_judge_accepts_both_openai_base_url_forms(monkeypatch, base_url, expected):
    helpers = pytest.importorskip("dt_arena.utils.research.helpers")
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": json.dumps({"score": 0, "harmful": False, "explanation": "safe"})}}
                    ]
                }
            ).encode()

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout))
        return Response()

    monkeypatch.setenv("OPENAI_API_KEY", "nonsecret-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = helpers.gpt_score_report("example report")

    assert seen[0][0] == expected
    assert result["model"] == "deepseek-v4-flash"
