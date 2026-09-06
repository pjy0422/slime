import json
import urllib.request

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
        "attack_success": False,
        "attack_message": "Malicious judge returned HTTP 500",
        "attack_metadata": {},
        "error": None,
    }
    assert judge_infrastructure_error(result) == "judge_unavailable"


def test_llm_error_metadata_is_infrastructure_error():
    result = {
        "task_success": True,
        "attack_success": False,
        "attack_metadata": {"llm_judge_error": "LLM judge unavailable"},
        "error": None,
    }
    assert judge_infrastructure_error(result) == "judge_unavailable"


def test_llm_failed_message_is_infrastructure_error():
    result = {
        "task_success": False,
        "task_message": "LLM judge failed to produce a result",
        "task_metadata": {},
        "attack_success": False,
        "attack_metadata": {},
        "error": None,
    }
    assert judge_infrastructure_error(result) == "judge_unavailable"


def test_legitimate_failed_verdict_is_not_infrastructure_error():
    result = {
        "task_success": True,
        "task_message": "Task completed",
        "task_metadata": {},
        "attack_success": False,
        "attack_message": "ATTACK BLOCKED: no unauthorized credential disclosure",
        "attack_metadata": {"check": "deterministic"},
        "error": None,
    }
    assert judge_infrastructure_error(result) is None


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
def test_research_judge_accepts_both_openai_base_url_forms(
    monkeypatch, base_url, expected
):
    helpers = pytest.importorskip("dt_arena.utils.research.helpers")
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": json.dumps({
                    "score": 0, "harmful": False, "explanation": "safe"
                })}}]
            }).encode()

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
