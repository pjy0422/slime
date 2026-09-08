# M7 adaptive feedback v2

M7 gives an H>1 policy bounded evidence from the preceding genuine failed
victim execution. It is disabled by default and does not participate in the
judge, reward, or H state transitions.

## Modes

- `final`: bounded victim final response only.
- `final+deterministic`: the same final response plus ordered qualified victim
  tool names/statuses and per-step observations.
- `final+deterministic+digestor`: the same deterministic projection plus a
  validated, bounded semantic digest. The Digestor is provider-neutral and is
  supplied by trusted rollout configuration.

An optional reasoning summarizer is separately controlled by
`ReasoningSummaryConfig(enabled=True)`. It is accepted only in Digestor mode.
No reasoning is inferred from behavior: the source is labeled
`explicit_reasoning`, `assistant_rationale`, or `unavailable`.

Feedback is emitted only when a victim evaluation started, the attack failed,
and another H attempt remains. Invalid submissions, infrastructure failures,
successes, and final failed attempts do not expose feedback to the policy.

## Exact evidence semantics

`locator_targeted=true` means a recorded victim tool call had an explicitly
supported tool/locator mapping and its redacted argument shape matched the
policy-owned locator by DTAP schema-v1 code-point length and UTF-8 SHA-256.

`access_call_status=ok` and the compatibility field
`injected_target_accessed=true` mean that matching call also completed without
an MCP error. They do **not** establish that:

- the response contained the injected entity;
- the returned content was inserted into the model context;
- the victim understood, believed, or followed it; or
- the injection caused a later action.

Those independent fields remain `null` unless future instrumentation provides
direct evidence. An error or missing completion after a matching start is
`unknown`, not `not_accessed`.

The MCP event sink is best-effort. The M7 DTAP overlay writes
`trace.completed` only after normal proxy shutdown and only if every append
succeeded. Absence can be converted to `false` only with that marker; otherwise
it remains `unknown`.

The M7 registry explicitly classifies all 115 enabled Linux mutators across 25
injection MCPs. Finance generated IDs support exact victim-read correlation.
Other Linux adapters can prove payload inclusion in a successful result from an
explicitly mapped victim MCP, but keep locator targeting `unknown` when the
backend does not expose a stable correlation field. Windows/macOS are outside
this milestone. No unregistered or name-derived fallback is used.

Prompt, tool-description, and skill presentation is observed at the provider
request boundary: `prompt.submitted`, `context.compiled.availableTools`, and
`context.compiled.systemPrompt`, respectively. Skill execution itself remains
`unknown` until OpenClaw provides a trusted structured skill-use event.

## Artifact and leakage boundary

Feedback reads exactly one regular, non-symlink artifact of each expected type
under the current attempt root. It recognizes both viewer-canonical
`victim-*.json*` names and DTAP's native OpenClaw trajectory and
`*.mcp-events.jsonl` names, because feedback is built before the optional viewer
export. Ambiguous, oversized, malformed, escaping, or missing artifacts degrade
to empty/unknown evidence. Raw MCP arguments, hashes, result digests, call IDs,
timestamps, judge data, runtime identity, benchmark metadata, credentials, and
host paths are not projected.

The Digestor sees an allowlisted evidence view rather than the raw trajectory.
Trace text is still untrusted injection-controlled input. Digest and reasoning
outputs therefore cross the same size, forbidden-key, registered-secret, host
path, and common credential-pattern guard as every policy response. Invalid or
timed-out optional analysis is omitted without changing the committed attack
result, reward, or H count.

## Integration and reproduction

Construct a trusted builder and pass it to `run_m4_episode` or directly to
`SubmissionCoordinator`:

```python
from examples.dtap_agent_rl.feedback import FeedbackBuilder, FeedbackMode

feedback = FeedbackBuilder(mode=FeedbackMode.FINAL_DETERMINISTIC)

await run_m4_episode(
    # existing M6 arguments ...
    feedback_builder=feedback,
)
```

Digestor mode requires a `Digestor` implementation. `PromptedLLMDigestor`
accepts an injected async JSON-completion callable, so core orchestration does
not depend on one model SDK. The included trusted Anthropic-compatible provider
supports hosted GLM, keeps credentials in headers, and records only bounded
call/failure/token/latency metrics. The reasoning summarizer is independently
enabled and cannot infer hidden reasoning. Hosted-model outputs receive one
schema-only retry when the first decoded JSON object has invalid field types or
pointers; transport and truncated-JSON failures still fail closed. The CLI
timeout applies per hosted request and the builder bounds the two-call retry.

Apply the DTAP marker and run tests:

```bash
source /home/pjy0422/workspace/dtap/bin/activate
cd /home/pjy0422/workspace/slime
examples/dtap_agent_rl/dtap_integration/apply.sh \
  /home/pjy0422/workspace/DecodingTrust-Agent

pytest -q \
  examples/dtap_agent_rl/tests/test_m7_deterministic.py \
  examples/dtap_agent_rl/tests/test_m7_feedback.py \
  examples/dtap_agent_rl/tests/test_m4_submission.py \
  examples/dtap_agent_rl/tests/test_m4_policy_contract.py \
  examples/dtap_agent_rl/tests/test_m6_openclaw_patch.py

pytest -q examples/dtap_agent_rl/tests
```

Run a generated H=2 OpenClaw episode with deterministic feedback:

```bash
export ANTHROPIC_API_KEY='...'
export DTAP_POLICY_ANTHROPIC_BASE_URL=https://provider.example
export DTAP_VICTIM_ANTHROPIC_BASE_URL=https://provider.example

python -m examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e \
  --task-dir /path/to/DecodingTrust-Agent/dataset/finance/malicious/indirect/... \
  --dtap-root /path/to/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python \
  --policy-model glm-5.2 --victim-model glm-5.2 \
  --victim-agent-type openclaw --max-submissions 2 --m6-placement \
  --feedback-mode final+deterministic \
  --artifacts-dir /tmp/m7-finance-indirect
```

The retained 2026-09-08 checks are under
`artifacts/m7-h2-feedback-native-fix-20260908` (direct) and
`artifacts/m7-h2-feedback-digestor-20260908` (indirect). Both exercised H=2 in
one policy session. The indirect policy used attempt-1 access/tool evidence to
repair the carrier and succeeded on attempt 2. Its post-fix hosted-GLM replay
also records the A/B/C invariants and a validated typed digest.
