"""Slime session boundary for an H-submission M3 trajectory."""

from __future__ import annotations

import inspect
from typing import Any, Mapping


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def run_h_turn_episode(
    *,
    adapter: Any,
    policy: Any,
    controller: Any,
    session_id: str | None = None,
    base_sample: Any = None,
    sampling_defaults: Mapping[str, Any] | None = None,
    max_context_tokens: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    open_session_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Open and finish exactly one policy session around all macro submissions.

    With ``base_sample`` this uses slime's current adapter API. The legacy
    keyword-only form remains useful for isolated protocol tests.
    """

    if session_id is None:
        session_id = await _maybe_await(adapter.open_session(**dict(open_session_kwargs or {})))
    else:
        await _maybe_await(
            adapter.open_session(
                session_id,
                sampling_defaults=dict(sampling_defaults or {}),
                max_context_tokens=max_context_tokens,
            )
        )
    policy_error: BaseException | None = None
    try:
        await policy.run_until_terminal(session_id=session_id, submit=controller.submit)
        if not controller.runtime.terminal:
            controller.runtime.record_infrastructure_failure(stage="policy_ended_nonterminal")
    except BaseException as exc:
        if not controller.runtime.terminal:
            controller.runtime.record_infrastructure_failure(stage="policy_runtime")
        policy_error = exc
    reward = controller.runtime.final_reward
    try:
        if base_sample is None:
            samples = await adapter.finish_session(
                session_id,
                reward=reward,
                remove_sample=controller.runtime.remove_sample,
            )
        else:
            samples = await adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=float(reward if reward is not None else 0.0),
                extra_metadata={
                    **dict(extra_metadata or {}),
                    "dtap_status": controller.runtime.status.value,
                    "dtap_submissions_used": controller.runtime.submissions_used,
                },
            )
            if controller.runtime.remove_sample:
                for sample in samples:
                    sample.remove_sample = True
                    status_enum = getattr(type(sample), "Status", None)
                    if status_enum is not None and hasattr(status_enum, "ABORTED"):
                        sample.status = status_enum.ABORTED
    finally:
        drop_session = getattr(adapter, "drop_session", None)
        if base_sample is not None and drop_session is not None:
            await _maybe_await(drop_session(session_id, wait_timeout=30))
    if policy_error is not None:
        raise policy_error
    return samples
