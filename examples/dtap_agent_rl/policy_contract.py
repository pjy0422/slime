"""Typed, allowlisted M4 policy-visible response boundary."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class PolicyContractViolation(RuntimeError):
    pass


_FORBIDDEN_KEYS = frozenset(
    {
        "judge_result",
        "judge_rationale",
        "victim_output",
        "trajectory_path",
        "runtime_identity",
        "task_success",
        "stdout",
        "stderr",
        "output_dir",
        "config_path",
        "exception",
        "traceback",
    }
)

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer[ ]+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b[a-f0-9]{32}\.[A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def canonical_policy_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _exact_keys(value: Mapping[str, Any], allowed: set[str], *, required: set[str]) -> None:
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise PolicyContractViolation("invalid feedback schema")


def _validate_feedback_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the M7 DTO before it crosses the existing leakage guard."""
    value = dict(raw)
    _exact_keys(
        value,
        {"schema_version", "final_response", "deterministic", "digest", "reasoning_summary"},
        required={"schema_version", "final_response"},
    )
    if value["schema_version"] != 1 or not isinstance(value["final_response"], str):
        raise PolicyContractViolation("invalid feedback schema")
    deterministic = value.get("deterministic")
    if deterministic is not None:
        if not isinstance(deterministic, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(
            deterministic,
            {"tool_sequence", "injections", "trace_complete"},
            required={"tool_sequence", "injections", "trace_complete"},
        )
        if not isinstance(deterministic["trace_complete"], bool):
            raise PolicyContractViolation("invalid feedback schema")
        if not isinstance(deterministic["tool_sequence"], (list, tuple)):
            raise PolicyContractViolation("invalid feedback schema")
        for call in deterministic["tool_sequence"]:
            if not isinstance(call, Mapping):
                raise PolicyContractViolation("invalid feedback schema")
            _exact_keys(call, {"index", "tool", "status"}, required={"index", "tool", "status"})
            if (
                not isinstance(call["index"], int)
                or isinstance(call["index"], bool)
                or not isinstance(call["tool"], str)
                or call["status"] not in {"ok", "error", "incomplete"}
            ):
                raise PolicyContractViolation("invalid feedback schema")
        if not isinstance(deterministic["injections"], (list, tuple)):
            raise PolicyContractViolation("invalid feedback schema")
        injection_keys = {
            "step_index", "injection_type", "polluted_tool_called",
            "injected_target_accessed", "access_state", "matched_tool", "match_basis",
            "locator_targeted", "access_call_status", "response_contains_injection",
            "presented_to_model",
        }
        for item in deterministic["injections"]:
            if not isinstance(item, Mapping):
                raise PolicyContractViolation("invalid feedback schema")
            _exact_keys(item, injection_keys, required=injection_keys)
            if item["injection_type"] not in {"prompt", "tool", "environment", "skill"}:
                raise PolicyContractViolation("invalid feedback schema")
            if not isinstance(item["step_index"], int) or isinstance(item["step_index"], bool):
                raise PolicyContractViolation("invalid feedback schema")
            for boolean_name in (
                "polluted_tool_called", "injected_target_accessed", "locator_targeted",
                "response_contains_injection", "presented_to_model",
            ):
                if item[boolean_name] is not None and not isinstance(item[boolean_name], bool):
                    raise PolicyContractViolation("invalid feedback schema")
            if item["matched_tool"] is not None and not isinstance(item["matched_tool"], str):
                raise PolicyContractViolation("invalid feedback schema")
            if item["match_basis"] not in {
                "qualified_tool", "exact_hashed_arguments", "not_supported",
                "not_applicable", None,
            }:
                raise PolicyContractViolation("invalid feedback schema")
            if item["access_state"] not in {"accessed", "not_accessed", "unknown", "not_applicable"}:
                raise PolicyContractViolation("invalid feedback schema")
            if item["access_call_status"] not in {"ok", "error", "incomplete", "not_observed", "not_applicable"}:
                raise PolicyContractViolation("invalid feedback schema")
    digest = value.get("digest")
    if digest is not None:
        if not isinstance(digest, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(digest, {"diagnosis", "preserve", "reconsider", "confidence"}, required={"diagnosis", "preserve", "reconsider", "confidence"})
        if (
            not isinstance(digest["diagnosis"], str)
            or not isinstance(digest["preserve"], (list, tuple))
            or not isinstance(digest["reconsider"], (list, tuple))
            or digest["confidence"] not in {"low", "medium", "high"}
        ):
            raise PolicyContractViolation("invalid feedback schema")
        if any(not isinstance(path, str) for path in (*digest["preserve"], *digest["reconsider"])):
            raise PolicyContractViolation("invalid feedback schema")
    summary = value.get("reasoning_summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise PolicyContractViolation("invalid feedback schema")
        _exact_keys(summary, {"source", "summary"}, required={"source", "summary"})
        if summary["source"] not in {"explicit_reasoning", "assistant_rationale", "unavailable"} or not isinstance(summary["summary"], str):
            raise PolicyContractViolation("invalid feedback schema")
    return value


@dataclass(frozen=True)
class PolicyLeakageGuard:
    secrets: tuple[str, ...] = field(default_factory=tuple, repr=False)
    forbidden_fragments: tuple[str, ...] = ()
    forbidden_keys: frozenset[str] = _FORBIDDEN_KEYS
    max_response_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if self.max_response_bytes < 256:
            raise ValueError("max_response_bytes is too small")

    def validate(self, payload: Any) -> dict[str, Any]:
        self._check_keys(payload)
        encoded = canonical_policy_json(payload)
        if len(encoded.encode("utf-8")) > self.max_response_bytes:
            raise PolicyContractViolation("policy response exceeded configured limit")
        needles: Iterable[str] = (*self.secrets, *self.forbidden_fragments)
        for needle in needles:
            if isinstance(needle, str) and len(needle) >= 8 and needle in encoded:
                raise PolicyContractViolation("policy response matched forbidden material")
        if any(pattern.search(encoded) for pattern in _SENSITIVE_VALUE_PATTERNS):
            raise PolicyContractViolation("policy response matched sensitive material")
        parsed = json.loads(encoded)
        if not isinstance(parsed, dict):
            raise PolicyContractViolation("policy response must be an object")
        return parsed

    def _check_keys(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in self.forbidden_keys:
                    raise PolicyContractViolation("policy response contains forbidden field")
                self._check_keys(item)
        elif isinstance(value, list):
            for item in value:
                self._check_keys(item)


@dataclass(frozen=True)
class PolicyContract:
    guard: PolicyLeakageGuard = PolicyLeakageGuard()

    def public_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.guard.validate(dict(payload))

    def accepted_submit(
        self,
        *,
        submission: int,
        success: bool,
        terminal: bool,
        remaining_submissions: int,
        feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(success, bool):
            raise PolicyContractViolation("success must be boolean")
        payload: dict[str, Any] = {
            "accepted": True,
            "submission": submission,
            "success": success,
            "terminal": bool(terminal),
            "remaining_submissions": remaining_submissions,
        }
        if feedback is not None:
            payload["feedback"] = _validate_feedback_projection(feedback)
        return self.public_payload(payload)

    def rejected_submit(
        self,
        *,
        code: str,
        terminal: bool,
        remaining_submissions: int,
    ) -> dict[str, Any]:
        allowed = {
            "INVALID_SUBMISSION",
            "EPISODE_TERMINAL",
            "POLICY_LIMIT",
            "EVALUATION_UNAVAILABLE",
        }
        if code not in allowed:
            code = "INVALID_SUBMISSION"
        return self.public_payload(
            {
                "accepted": False,
                "terminal": bool(terminal),
                "remaining_submissions": remaining_submissions,
                "error": {"code": code},
            }
        )

    def from_internal_submit(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        if receipt.get("accepted") is True:
            return self.accepted_submit(
                submission=int(receipt["submission"]),
                success=receipt["success"],
                terminal=bool(receipt["terminal"]),
                remaining_submissions=int(receipt["remaining_submissions"]),
                feedback=(
                    receipt.get("feedback")
                    if isinstance(receipt.get("feedback"), Mapping)
                    else None
                ),
            )
        raw_code = None
        error = receipt.get("error")
        if isinstance(error, Mapping):
            raw_code = error.get("code")
        errors = receipt.get("errors")
        if raw_code is None and isinstance(errors, list) and errors and isinstance(errors[0], Mapping):
            raw_code = errors[0].get("code")
        if raw_code in {"EPISODE_TERMINAL"}:
            code = "EPISODE_TERMINAL"
        elif raw_code in {"POLICY_LIMIT"}:
            code = "POLICY_LIMIT"
        elif raw_code in {"INFRA_ERROR", "SECURITY_ABORT", "EVALUATION_UNAVAILABLE"}:
            code = "EVALUATION_UNAVAILABLE"
        else:
            code = "INVALID_SUBMISSION"
        return self.rejected_submit(
            code=code,
            terminal=bool(receipt.get("terminal", False)),
            remaining_submissions=int(receipt.get("remaining_submissions", 0)),
        )
