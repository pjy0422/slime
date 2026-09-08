# M7 adaptive feedback v1

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

The current exact-locator registry covers Slack workspace/channel, filesystem
path aliases, and browser URL when the injection action itself supplies a URL.
Other environment mutators return `unknown`; no name-based heuristic is used.

## Artifact and leakage boundary

Feedback reads exactly one regular, non-symlink artifact of each known name
under the current attempt root. Ambiguous, oversized, malformed, escaping, or
missing artifacts degrade to empty/unknown evidence. Raw MCP arguments, hashes,
result digests, call IDs, timestamps, judge data, runtime identity, benchmark
metadata, credentials, and host paths are not projected.

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
not depend on one model SDK.

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
