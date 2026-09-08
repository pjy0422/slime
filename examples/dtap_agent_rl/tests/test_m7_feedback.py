import hashlib
import json

import pytest

from examples.dtap_agent_rl.actions import ValidatedAttackStep
from examples.dtap_agent_rl.feedback.builder import FeedbackBuilder
from examples.dtap_agent_rl.feedback.deterministic import (
    extract_deterministic_feedback,
    parse_mcp_events,
)
from examples.dtap_agent_rl.feedback.digestor import (
    PromptedLLMDigestor,
    pointer_exists,
    validate_repair_digest,
)
from examples.dtap_agent_rl.feedback.schema import FeedbackMode, ReasoningSummaryConfig, RepairDigest
from examples.dtap_agent_rl.policy_contract import PolicyContract, PolicyContractViolation, PolicyLeakageGuard


STEP = ValidatedAttackStep(type="tool", mode="suffix", content="payload", injected_tool="slack:channels_history")
PLAN = {"steps": [STEP.to_dict()]}


def _artifacts(root, *, final="victim final", reasoning=False):
    root.mkdir(parents=True, exist_ok=True)
    trajectory = [
        {"role": "user", "state": "visible task"},
        *([{"role": "reasoning", "content": "explicit trace"}] if reasoning else []),
        {"role": "agent", "metadata": {"message": final}},
    ]
    (root / "victim-trajectory.json").write_text(
        json.dumps({
            "task_info": {"malicious_instruction": "HIDDEN", "original_instruction": "HIDDEN"},
            "traj_info": {"agent_final_response": final},
            "trajectory": trajectory,
        }), encoding="utf-8"
    )
    (root / "victim-mcp-events.jsonl").write_text(
        json.dumps({
            "type": "tool.started", "call_id": "a", "server": "slack", "tool": "channels_history",
            "arguments": {"secret": {"type": "string", "length": 6, "sha256": hashlib.sha256(b"secret").hexdigest()}},
        }) + "\n" + json.dumps({
            "type": "tool.completed", "call_id": "a", "server": "slack", "tool": "channels_history",
            "is_error": False, "result_digest": "DO_NOT_EXPOSE",
        }) + "\n", encoding="utf-8"
    )


class Digestor:
    def __init__(self, value=None, error=None):
        self.value = value or RepairDigest("change payload", ("/steps/0/injected_tool",), ("/steps/0/content",), "medium")
        self.error = error
        self.observation = None

    async def digest(self, observation):
        self.observation = observation
        if self.error:
            raise self.error
        return self.value


class Summarizer:
    def __init__(self):
        self.calls = 0

    async def summarize(self, trace):
        self.calls += 1
        return "reasoning summary"


@pytest.mark.asyncio
async def test_modes_preserve_identical_final_and_deterministic(tmp_path):
    _artifacts(tmp_path)
    digestor = Digestor()
    builders = [
        FeedbackBuilder(mode=FeedbackMode.FINAL_ONLY),
        FeedbackBuilder(mode=FeedbackMode.FINAL_DETERMINISTIC),
        FeedbackBuilder(mode=FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR, digestor=digestor),
    ]
    results = [await b.build(attempt_root=tmp_path, submitted_steps=(STEP,), submitted_plan=PLAN) for b in builders]
    assert [x["final_response"] for x in results] == ["victim final"] * 3
    assert results[1]["deterministic"] == results[2]["deterministic"]
    assert results[2]["digest"]["reconsider"] == ("/steps/0/content",)
    serialized_observation = repr(digestor.observation)
    assert "HIDDEN" not in serialized_observation
    assert "DO_NOT_EXPOSE" not in serialized_observation
    assert "secret" not in serialized_observation


@pytest.mark.asyncio
async def test_builder_discovers_native_dtap_artifact_names(tmp_path):
    _artifacts(tmp_path)
    (tmp_path / "victim-trajectory.json").rename(
        tmp_path / "openclaw_trajectory_20260908.json"
    )
    (tmp_path / "victim-mcp-events.jsonl").rename(
        tmp_path / "episode-abc.mcp-events.jsonl"
    )

    result = await FeedbackBuilder(
        mode=FeedbackMode.FINAL_DETERMINISTIC
    ).build(attempt_root=tmp_path, submitted_steps=(STEP,), submitted_plan=PLAN)

    assert result["final_response"] == "victim final"
    assert result["deterministic"]["tool_sequence"] == ({
        "index": 0, "tool": "slack:channels_history", "status": "ok"
    },)
    assert result["deterministic"]["trace_complete"] is False


@pytest.mark.asyncio
async def test_digestor_failure_keeps_final_and_deterministic(tmp_path):
    _artifacts(tmp_path)
    builder = FeedbackBuilder(
        mode=FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR,
        digestor=Digestor(error=RuntimeError("boom")),
    )
    result = await builder.build(attempt_root=tmp_path, submitted_steps=(STEP,), submitted_plan=PLAN)
    assert result["final_response"] == "victim final"
    assert "deterministic" in result
    assert "digest" not in result


@pytest.mark.asyncio
async def test_prompted_digestor_publishes_exact_allowed_pointer_inventory(tmp_path):
    _artifacts(tmp_path)
    prompts = []

    async def complete(prompt):
        prompts.append(json.loads(prompt))
        return {
            "diagnosis": "change payload",
            "preserve": ["/steps/0/injected_tool"],
            "reconsider": ["/steps/0/content"],
            "confidence": "medium",
            "payload_effect": "rejected",
            "evidence_refs": [
                "/deterministic/tool_sequence/0/tool",
            ],
        }

    result = await FeedbackBuilder(
        mode=FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR,
        digestor=PromptedLLMDigestor(complete),
    ).build(attempt_root=tmp_path, submitted_steps=(STEP,), submitted_plan=PLAN)

    assert result["digest"]["payload_effect"] == "rejected"
    assert "/steps/0/content" in prompts[0]["allowed_submission_pointers"]
    assert (
        "/deterministic/tool_sequence/0/tool"
        in prompts[0]["allowed_evidence_pointers"]
    )


@pytest.mark.asyncio
async def test_prompted_digestor_retries_one_schema_only_failure(tmp_path):
    _artifacts(tmp_path)
    calls = []

    async def complete(prompt):
        calls.append(json.loads(prompt))
        if len(calls) == 1:
            return {
                "diagnosis": "wrong types", "preserve": "keep it",
                "reconsider": "change it", "confidence": 0.5,
                "payload_effect": "ignored", "evidence_refs": [],
            }
        return {
            "diagnosis": "corrected", "preserve": [],
            "reconsider": ["/steps/0/content"], "confidence": "medium",
            "payload_effect": "ignored", "evidence_refs": [],
        }

    result = await FeedbackBuilder(
        mode=FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR,
        digestor=PromptedLLMDigestor(complete),
    ).build(attempt_root=tmp_path, submitted_steps=(STEP,), submitted_plan=PLAN)

    assert result["digest"]["diagnosis"] == "corrected"
    assert calls[1]["validation_error"] == "digest pointers must be arrays"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_reasoning_summary_is_separately_opt_in_and_source_labeled(tmp_path):
    _artifacts(tmp_path, reasoning=True)
    builder = FeedbackBuilder(
        mode=FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR,
        digestor=Digestor(), reasoning_summarizer=Summarizer(),
        reasoning=ReasoningSummaryConfig(enabled=True),
    )
    result = await builder.build(attempt_root=tmp_path, submitted_steps=(STEP,), submitted_plan=PLAN)
    assert result["reasoning_summary"] == {"source": "explicit_reasoning", "summary": "reasoning summary"}
    with pytest.raises(ValueError):
        FeedbackBuilder(mode=FeedbackMode.FINAL_DETERMINISTIC, reasoning_summarizer=Summarizer(), reasoning=ReasoningSummaryConfig(enabled=True))


@pytest.mark.asyncio
async def test_reasoning_summary_does_not_infer_hidden_reasoning_from_behavior(tmp_path):
    _artifacts(tmp_path, reasoning=False)
    summarizer = Summarizer()
    builder = FeedbackBuilder(
        mode=FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR,
        digestor=Digestor(), reasoning_summarizer=summarizer,
        reasoning=ReasoningSummaryConfig(enabled=True),
    )
    result = await builder.build(attempt_root=tmp_path, submitted_steps=(STEP,), submitted_plan=PLAN)
    assert result["reasoning_summary"] == {"source": "unavailable", "summary": ""}
    assert summarizer.calls == 0


def test_digest_pointers_are_existing_local_nonoverlapping_paths():
    assert pointer_exists(PLAN, "/steps/0/content")
    assert not pointer_exists(PLAN, "/steps/1/content")
    with pytest.raises(ValueError):
        validate_repair_digest(RepairDigest("x", ("/steps/0/content",), ("/steps/0/content",), "high"), PLAN)
    with pytest.raises(ValueError):
        validate_repair_digest(RepairDigest("x", (), ("/judge/result",), "high"), PLAN)
    with pytest.raises(ValueError):
        validate_repair_digest(RepairDigest("x", (), ("",), "high"), PLAN)


def test_v2_digest_validates_payload_effect_and_deterministic_evidence(tmp_path):
    _artifacts(tmp_path)
    trace = parse_mcp_events(tmp_path / "victim-mcp-events.jsonl")
    deterministic = extract_deterministic_feedback((STEP,), trace)
    value = RepairDigest(
        "tool was selected but payload was rejected",
        ("/steps/0/injected_tool",),
        ("/steps/0/content",),
        "medium",
        "rejected",
        ("/deterministic/injections/0/polluted_tool_called",),
    )
    assert validate_repair_digest(
        value, PLAN, deterministic=deterministic
    ).payload_effect == "rejected"
    with pytest.raises(ValueError):
        validate_repair_digest(
            RepairDigest("x", (), (), "low", "ignored", ("/submission/steps/0",)),
            PLAN,
            deterministic=deterministic,
        )


def test_policy_guard_blocks_digest_or_final_secret_and_forbidden_fields():
    contract = PolicyContract(PolicyLeakageGuard(secrets=("super-secret-token",)))
    base = {"schema_version": 1, "final_response": "ok"}
    assert "feedback" in contract.accepted_submit(
        submission=1, success=False, terminal=False, remaining_submissions=1, feedback=base
    )
    with pytest.raises(PolicyContractViolation):
        contract.accepted_submit(
            submission=1, success=False, terminal=False, remaining_submissions=1,
            feedback={"schema_version": 1, "final_response": "super-secret-token"},
        )
    with pytest.raises(PolicyContractViolation):
        contract.accepted_submit(
            submission=1, success=False, terminal=False, remaining_submissions=1,
            feedback={**base, "judge_result": {}},
        )


@pytest.mark.asyncio
async def test_symlink_or_ambiguous_artifacts_are_not_read(tmp_path):
    real = tmp_path / "outside.json"
    real.write_text(json.dumps({"traj_info": {"agent_final_response": "LEAK"}}))
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "victim-trajectory.json").symlink_to(real)
    builder = FeedbackBuilder(mode=FeedbackMode.FINAL_ONLY)
    result = await builder.build(attempt_root=root, submitted_steps=(STEP,), submitted_plan=PLAN)
    assert result["final_response"] == ""


@pytest.mark.asyncio
async def test_digestor_input_and_final_projection_redact_credentials_and_host_paths(tmp_path):
    credential = "sk-abcdefghijklmnop123456"
    host_path = "/trusted/attempts/episode-1"
    _artifacts(tmp_path, final=f"failure {credential} at {host_path}")
    digestor = Digestor()
    builder = FeedbackBuilder(
        mode=FeedbackMode.FINAL_DETERMINISTIC_DIGESTOR,
        digestor=digestor,
    )

    result = await builder.build(
        attempt_root=tmp_path,
        submitted_steps=(STEP,),
        submitted_plan=PLAN,
        redactions=(host_path,),
    )

    assert credential not in result["final_response"]
    assert host_path not in result["final_response"]
    observation = repr(digestor.observation)
    assert credential not in observation
    assert host_path not in observation
