"""Typed, allowlisted M4 policy-visible response boundary."""

from __future__ import annotations

import json
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


def canonical_policy_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
    ) -> dict[str, Any]:
        if not isinstance(success, bool):
            raise PolicyContractViolation("success must be boolean")
        return self.public_payload(
            {
                "accepted": True,
                "submission": submission,
                "success": success,
                "terminal": bool(terminal),
                "remaining_submissions": remaining_submissions,
            }
        )

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
