"""Typed internal and policy-visible M7 feedback data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal


class FeedbackMode(str, Enum):
    DISABLED = "disabled"
    FINAL_ONLY = "final"
    FINAL_DETERMINISTIC = "final+deterministic"
    FINAL_DETERMINISTIC_DIGESTOR = "final+deterministic+digestor"


@dataclass(frozen=True)
class ReasoningSummaryConfig:
    """An independent opt-in inside mode C; it is never silently enabled."""

    enabled: bool = False
    timeout_seconds: float = 20.0
    max_chars: int = 2_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.max_chars < 64:
            raise ValueError("invalid reasoning-summary limits")


@dataclass(frozen=True)
class ToolCallObservation:
    index: int
    tool: str
    status: Literal["ok", "error", "incomplete"]


@dataclass(frozen=True)
class InjectionObservation:
    step_index: int
    injection_type: Literal["prompt", "tool", "environment", "skill"]
    polluted_tool_called: bool | None
    injected_target_accessed: bool | None
    access_state: Literal["accessed", "not_accessed", "unknown", "not_applicable"]
    matched_tool: str | None = None
    match_basis: Literal[
        "qualified_tool", "exact_hashed_arguments", "not_supported", "not_applicable"
    ] | None = None
    # These fields prevent `accessed` from being mistaken for content exposure.
    locator_targeted: bool | None = None
    access_call_status: Literal[
        "ok", "error", "incomplete", "not_observed", "not_applicable"
    ] = "not_applicable"
    response_contains_injection: bool | None = None
    presented_to_model: bool | None = None


@dataclass(frozen=True)
class DeterministicFeedback:
    tool_sequence: tuple[ToolCallObservation, ...]
    injections: tuple[InjectionObservation, ...]
    trace_complete: bool


@dataclass(frozen=True)
class VictimTraceItem:
    source: Literal["trajectory", "mcp_event", "provider_reasoning"]
    source_index: int
    kind: Literal["user", "reasoning", "tool_call", "tool_result", "assistant"]
    content: Any


@dataclass(frozen=True)
class VictimVisibleTrace:
    """Allowlisted evidence streams, not a claim of a complete model transcript."""

    items: tuple[VictimTraceItem, ...]
    interaction_complete: bool
    reasoning_source: Literal[
        "explicit_reasoning", "assistant_rationale", "unavailable", "disabled"
    ]


@dataclass(frozen=True)
class RepairDigest:
    diagnosis: str
    preserve: tuple[str, ...]
    reconsider: tuple[str, ...]
    confidence: Literal["low", "medium", "high"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasoningSummary:
    source: Literal["explicit_reasoning", "assistant_rationale", "unavailable"]
    summary: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackEvidence:
    final_response: str
    deterministic: DeterministicFeedback
    victim_trace: VictimVisibleTrace


def deterministic_to_dict(value: DeterministicFeedback) -> dict[str, Any]:
    return asdict(value)
