from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .actions import ValidatedAttackStep


@dataclass(frozen=True)
class DtapActionApi:
    AttackStepConfig: Any
    AttackTurn: Any
    AttackConfig: Any


def load_dtap_action_api() -> DtapActionApi:
    try:
        from dt_arena.src.types.task import AttackConfig, AttackStepConfig, AttackTurn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("DTAP is not importable; add DecodingTrust-Agent to PYTHONPATH") from exc
    return DtapActionApi(AttackStepConfig=AttackStepConfig, AttackTurn=AttackTurn, AttackConfig=AttackConfig)


def to_dtap_step(step: ValidatedAttackStep, *, api: DtapActionApi | None = None):
    api = api or load_dtap_action_api()
    return api.AttackStepConfig(
        type=step.type,
        mode=step.mode,
        content=step.content,
        injected_tool=step.injected_tool,
        injection_mcp_tool=step.injection_mcp_tool,
        kwargs=step.kwargs or {},
        skill_name=step.skill_name,
        row=step.row,
    )


def to_dtap_attack_config(
    steps: Iterable[ValidatedAttackStep],
    *,
    threat_model: str | None,
    malicious_goal: str | None = None,
    api: DtapActionApi | None = None,
):
    """Materialize validated policy actions into DTAP's in-memory config types.

    M2 does not write config.yaml. Global tool/skill injections are placed in
    turn 1 because DTAP merges them before agent initialization.
    """
    api = api or load_dtap_action_api()
    grouped: dict[int, list[Any]] = defaultdict(list)
    for step in steps:
        grouped[step.dtap_turn_id].append(to_dtap_step(step, api=api))
    turns = [api.AttackTurn(turn_id=turn_id, attack_steps=grouped[turn_id]) for turn_id in sorted(grouped)]
    return api.AttackConfig(
        threat_model=threat_model,
        malicious_goal=malicious_goal,
        attack_turns=turns,
    )
