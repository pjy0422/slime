import pytest

from examples.dtap_agent_rl.policy_contract import (
    PolicyContract,
    PolicyContractViolation,
    PolicyLeakageGuard,
    canonical_policy_json,
)


def test_receipt_is_equivalent_when_trusted_runner_details_change():
    contract = PolicyContract()
    base = {
        "accepted": True,
        "submission": 1,
        "success": False,
        "terminal": False,
        "remaining_submissions": 2,
    }
    noisy = {
        **base,
        "judge_result": {"rationale": "SECRET"},
        "victim_output": "SECRET",
        "task_success": True,
        "runtime_identity": "docker-1",
    }
    assert canonical_policy_json(contract.from_internal_submit(base)) == canonical_policy_json(
        contract.from_internal_submit(noisy)
    )


def test_mutation_rejection_is_coarse_and_drops_internal_error_details():
    contract = PolicyContract()
    receipt = contract.from_internal_submit(
        {
            "accepted": False,
            "terminal": False,
            "remaining_submissions": 3,
            "errors": [
                {
                    "code": "YAML_SCHEMA_MISMATCH",
                    "message": "/trusted/config.yaml parser traceback SECRET",
                }
            ],
        }
    )
    assert receipt == {
        "accepted": False,
        "terminal": False,
        "remaining_submissions": 3,
        "error": {"code": "INVALID_SUBMISSION"},
    }


@pytest.mark.parametrize(
    "secret",
    [
        "Bearer abcdefghijklmnopqrstuvwxyz",
        "sk-abcdefghijklmnop123456",
        "0123456789abcdef0123456789abcdef.abcdefghijkl",
        "".join(("-----BEGIN PRIVATE", " KEY-----")),
    ],
)
def test_leakage_guard_rejects_common_unregistered_credential_shapes(secret):
    with pytest.raises(PolicyContractViolation):
        PolicyLeakageGuard().validate({"feedback": secret})


@pytest.mark.parametrize(
    "payload",
    [
        {"judge_result": {}},
        {"nested": {"trajectory_path": "/trusted/a"}},
        {"value": "prefix super-secret-token suffix"},
    ],
)
def test_leakage_guard_rejects_forbidden_keys_and_secret_fragments(payload):
    guard = PolicyLeakageGuard(secrets=("super-secret-token",))
    with pytest.raises(PolicyContractViolation):
        guard.validate(payload)
