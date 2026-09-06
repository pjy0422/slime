# DTAP Agent RL — M0 through M6

This directory is intended to be copied into the **slime repository** at
`examples/dtap_agent_rl/`. DTAP remains an external pinned dependency.

## M0

M0 creates trusted `TaskSnapshot` state, an allowlisted `PolicyTaskSpec`, and a
normalized `AttackSurface`. Benchmark example attacks (`Attack.attack_turns`) do
not participate in observation or action-surface construction.

## M1

M1 adds a minimal `DTAPClaudeCodeHarness(ClaudeCodeHarness)` plus a host-side,
read-only MCP boundary.

Policy-visible MCP tools are exactly:

- `get_task_spec()`
- `get_attack_surface()`

M1 requires `fastmcp>=2.6` for HTTP bearer/header support.

Episode identity is **not** a tool argument. Claude Code expands
`${DTAP_EPISODE_TOKEN}` into the HTTP `Authorization` header from the environment.
The MCP server resolves that opaque capability against an in-memory
`EpisodeRegistry`.

### Trust boundary

- `config.yaml`, attack examples, DTAP task paths, judge state: host/trusted only.
- `PolicyTaskSpec`, `AttackSurface`: policy-visible.
- `DTAPClaudeCodeHarness`: does not import DTAP and never sees a task directory.
- `EpisodeRegistry`: never enumerable from MCP.
- M1 tools are read-only; mutation/evaluation arrive in M2/M3.

## Wiring from the future `generate.py`

```python
snapshot = load_task_snapshot(task_dir)
view = await build_episode_view(snapshot, live_catalog)

with registered_episode(registry, token=session_id, view=view):
    await DTAPClaudeCodeHarness().run(
        sandbox,
        workdir="/workspace/empty",
        session_id=session_id,
        adapter_url=adapter_url,
        time_budget_sec=300,
        prompt=prompt,
    )
```

The host-side FastMCP server must run in the same process/service as the `EpisodeRegistry` and be reachable from the sandbox, and the host
process should export the full MCP endpoint, e.g.:

```bash
export DTAP_HARNESS_URL=http://<reachable-host>:19090/mcp/
```

`DTAPClaudeCodeHarness` writes an MCP config containing only env placeholders and
injects the episode capability at process launch.

## Tests

Inside a slime checkout:

```bash
pytest -q examples/dtap_agent_rl/tests
```

The tests cover:

- M0 no-solution-leak regression
- token lifecycle and cross-episode isolation
- cleanup on exception
- bearer parsing
- MCP payloads contain no token
- Claude Code base config is preserved
- MCP config contains env placeholders, not the token
- slime Anthropic adapter env is preserved
- native Claude Code tools are explicitly denied
- exact M1 MCP tool list (when FastMCP is installed)

## HTTP MCP deployment

Create the server once per slime worker/service process and keep the registry in
that same process:

```python
registry = EpisodeRegistry()
mcp = create_mcp_server(registry)
mcp.run(transport="http", host="0.0.0.0", port=19090, stateless_http=True)
```

For an already-running asyncio process, use `run_async()` instead. A production
M1 deployment should run one shared service per worker/node rather than one MCP
process per episode.


## Direct Anthropic API smoke (no RL / no slime actor)

After the M0-M1 test suite is green, you can exercise the real Claude Code +
Anthropic API + HTTP/Bearer MCP boundary without SGLang/Ray/RL:

```bash
source /home/pjy0422/workspace/dtap/bin/activate
cd /home/pjy0422/workspace/slime

export PYTHONPATH=/path/to/DecodingTrust-Agent:$PYTHONPATH
export ANTHROPIC_API_KEY='sk-ant-...'

python -m examples.dtap_agent_rl.scripts.smoke_m1_api \
  --task-dir /path/to/DecodingTrust-Agent/dataset/.../<task-id> \
  --model sonnet
```

The smoke performs five phases:

1. Start the real DTAP Docker environment and discover live MCP schemas.
2. Tear the DTAP runtime back down; retain only sanitized `EpisodeView`.
3. Start the host-side read-only FastMCP server with a random bearer capability.
4. Run local Claude Code directly against Anthropic (`ANTHROPIC_API_KEY`), requiring
   calls to `get_task_spec` and `get_attack_surface` only.
5. Assert both calls occurred, no benchmark example/config path/capability leaked,
   original `config.yaml` stayed unchanged, and the registry was cleaned.

The script explicitly removes `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` from
Claude's child environment so a previously configured slime adapter/gateway cannot
accidentally receive this direct-API smoke request.

Use `--show-stream` to print Claude Code's full stream-json output. By default
`Attack.additional_information` is hidden; opt in with
`--expose-additional-information` only after auditing that field for the task set.

This smoke does **not** run the DTAP victim agent, attack mutation, judge, reward,
or any RL training loop. Those begin in later milestones.

## M3: H-submission trajectories

M3 adds `submit_attack(plan)` while keeping the three M2 inspection/validation tools.
One Claude/slime context may submit up to `H` plans. Failed attempts return one bit
and the same policy context continues; success or the H-th normal failure terminates.

Create the shared server with paired registries:

```python
views = EpisodeRegistry()
submissions = EpisodeSubmissionRegistry()
mcp = create_m3_mcp_server(views, submissions)
```

The custom generate entrypoint is
`examples.dtap_agent_rl.generate.generate`. Configure its trusted runtime once per
worker with `configure_runtime(M3GenerateRuntime(...))` before rollout generation.

Deterministic Claude H-loop smoke:

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
python -m examples.dtap_agent_rl.scripts.smoke_m3_hloop_api --model sonnet
```

Real one-submit DTAP smoke:

```bash
python -m examples.dtap_agent_rl.scripts.smoke_m3_dtap \
  --dtap-root /path/to/DecodingTrust-Agent \
  --task-dir /path/to/DecodingTrust-Agent/dataset/.../<task-id> \
  --plan /path/to/validated-plan.json \
  --agent-type openaisdk --model gpt-5.4
```

The real smoke creates an ephemeral task copy, starts a fresh `TaskExecutor`, runs
the victim and judge, reads `attack_success`, shuts the executor down, and verifies
that the source config did not change.

## M4: reward and isolation hardening

M4 keeps the same four policy tools and H victim-run semantics. Production M4 uses a
separate MCP capability, atomic episode authority, Q submit-call budget, typed
receipts, bounded judge firewall, whole-task manifests, explicit child environment,
strict Claude configuration, sandbox attestation, and one worker-scoped scheduler.

Minimal trusted worker wiring:

```python
from examples.dtap_agent_rl.attempt_runner import DtapAttemptRunner
from examples.dtap_agent_rl.authority import EpisodeAuthorityRegistry
from examples.dtap_agent_rl.generate_m4 import M4GenerateRuntime, configure_runtime
from examples.dtap_agent_rl.mcp_server import create_m4_mcp_server
from examples.dtap_agent_rl.sandbox_policy import SandboxPolicyVerifier
from examples.dtap_agent_rl.scheduler import AttemptScheduler
from examples.dtap_agent_rl.security_policy import M4SecurityPolicy
from examples.dtap_agent_rl.transport_security import build_m4_http_app

policy = M4SecurityPolicy(
    max_submit_calls=6,
    max_parallel_attempts=8,
    max_queued_attempts=32,
    inherited_dtap_env_names=("PYTHONPATH", "OPENAI_API_KEY"),
)
scheduler = AttemptScheduler(
    max_parallel=policy.max_parallel_attempts,
    max_queued=policy.max_queued_attempts,
    wait_timeout=policy.queue_wait_timeout_seconds,
)
runner = DtapAttemptRunner(
    security_policy=policy,
    scheduler=scheduler,
    dtap_root="/path/to/DecodingTrust-Agent",
)
authorities = EpisodeAuthorityRegistry()
server = create_m4_mcp_server(authorities, security_policy=policy)
http_app = build_m4_http_app(server, policy)
```

`configure_runtime(M4GenerateRuntime(...))` must receive the same policy, shared
runner, authority registry, and a `SandboxPolicyVerifier`. The sandbox must attest
non-root execution, no capabilities/Docker/host mounts, isolated home/workdir and
`/proc`, default-deny network with exactly the adapter and policy-MCP endpoints, and
process-group cleanup. Missing attestation aborts before episode registration.

The small DTAP-side compatibility changes required by this harness are versioned
with slime under `dtap_integration/`. Apply them to the external checkout before
running a real DTAP gate:

```bash
examples/dtap_agent_rl/dtap_integration/apply.sh \
  /path/to/DecodingTrust-Agent
```

Direct Claude contract smoke:

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
python -m examples.dtap_agent_rl.scripts.smoke_m4_claude_boundary --model sonnet
```

This checks strict MCP configuration, the exact four tools, H/Q behavior, and
config/judge/path/token canaries. Deployment OS/network enforcement is a separate
sandbox integration gate.

Real parallel DTAP/Docker gate:

```bash
export PYTHONPATH=/path/to/DecodingTrust-Agent:$PYTHONPATH
export DTAP_M4_ROOT=/path/to/DecodingTrust-Agent
export DTAP_M4_PARALLEL_TASK_DIR=/path/to/a/prompt-compatible/task
export DTAP_M4_INHERITED_ENV_NAMES=PYTHONPATH,OPENAI_API_KEY

pytest -q -m integration \
  examples/dtap_agent_rl/tests/test_m4_real_dtap_optional.py
```

For an Anthropic-compatible provider used by both the policy and a ClaudeSDK
victim, keep the actual policy adapter variables out of the DTAP child and pass
trusted aliases explicitly:

```bash
export DTAP_POLICY_ANTHROPIC_BASE_URL=https://provider.example
export DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN=1
export DTAP_VICTIM_ANTHROPIC_BASE_URL=https://provider.example
export DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN=1
export DTAP_M4_AGENT_TYPE=claudesdk
export DTAP_M4_MODEL=provider-model
export DTAP_M4_INHERITED_ENV_NAMES=PYTHONPATH,ANTHROPIC_API_KEY,DTAP_VICTIM_ANTHROPIC_BASE_URL,DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN
```

The aliases are consumed only inside the isolated DTAP subprocess. Direct
inheritance of `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, policy MCP bearer,
and slime adapter session credentials remains forbidden.

See `M4_IMPLEMENTATION_PLAN.md`, `M4_IMPLEMENTATION_REPORT.md`, and
`M4_TEST_REPORT.md` for the frozen threat model and release gates.

## M5: deterministic environment verification

The slime-managed DTAP patch adds a zero-target-LLM auxiliary gate. Route mode
(the default) binds the started injection server's resolved endpoint to the
TaskExecutor project and live Docker container, supporting both published bridge
ports and DTAP's host-network containers. Placement mode additionally performs
independent read-back for every enabled Linux injection mutator. The synchronized
registry currently covers 115 tools across 25 injection MCPs; Windows and macOS
remain explicitly excluded until guest-side read-only evidence exists.

```bash
export DTAP_ENV_VERIFICATION=placement
# Optional rollout gate: reject tools without a placement adapter.
export DTAP_ENV_VERIFICATION_STRICT=1
```

Read-only injection tools are marked `not_applicable`; mutating tools without an
adapter are marked `unsupported`. Proofs stay inside the trusted DTAP process and
only expose status plus SHA-256 digest in logs. A verification failure is treated
as infrastructure failure, never as reward zero, so the four-tool M4 policy
contract and reward semantics are unchanged.

The current support matrix, unsupported-server handoff notes, adapter checklist,
and M0-M5 deterministic/live reproduction commands are maintained in
`dtap_integration/README.md`. The live smoke runner starts only DTAP sandbox
infrastructure and never invokes the victim LLM or judge:

```bash
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /path/to/DecodingTrust-Agent/dataset/research/malicious/indirect/Radiological_Risks/2 \
  --strict
```

The deterministic M4 boundary smoke intentionally keeps a fixed candidate for
contract regression. The separate M5 end-to-end smoke supplies no plan or
payload: GLM-5.2 must inspect the live surface, generate and validate a plan,
submit it, then wait for the real ClaudeSDK victim and judge:

```bash
python -m examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e \
  --task-dir /path/to/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --dtap-root /path/to/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python \
  --policy-model glm-5.2 --victim-model glm-5.2
```

Use the policy/victim provider aliases shown above and set placement strict mode
for this release gate. M5 verification is trusted auxiliary state only: it is
not returned as an arbitrary environment oracle and does not affect reward.
## M6: policy-scoped placement receipts

M6 adds exactly two capabilities to the hardened surface:

- `apply_attack_step(step)` accepts one already-validatable environment action,
  applies it in a fresh auxiliary DTAP sandbox, and returns an opaque `action_id`.
- `validate_placement(action_id)` returns the stored independent read-back only
  when that ID belongs to the same episode capability. A verified result includes
  `validated_placement_locator`. An invalid result includes the requested locator
  plus `repair.fields`, so the policy can revise only the location-bearing fields.

The policy cannot provide a server, path, query, or payload to
`validate_placement`; consequently it is not an arbitrary environment oracle.
Probe sandboxes are destroyed before the result is returned, probes do not spend
H/Q and do not affect reward, and `submit_attack` re-applies the final plan in its
own fresh victim/judge environment. Non-environment actions are rejected by the
probe endpoint because they have no independent pre-victim placement semantics.

The default probe budget is eight actions per episode and is bounded by
`M4SecurityPolicy.max_placement_actions`. Use `create_m6_mcp_server` and provide a
`DtapPlacementRunner` when configuring the runtime; M4 remains an unchanged
four-tool contract when no placement runner is supplied.

Run one real receipt/read-back smoke without a victim LLM:

```bash
python -m examples.dtap_agent_rl.scripts.smoke_m6_placement \
  --task-dir /path/to/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --dtap-root /path/to/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python
```

Add `--m6-placement` to `smoke_m5_glm_e2e` for a generated-plan run that requires
GLM-5.2 to validate, apply, read the owned receipt, and then submit the final plan.

## Policy and victim trajectory viewer

`tools/dtap-trajectory-viewer` renders the policy MCP trajectory and the DTAP
victim trajectory in one self-contained HTML page. It also shows a unified diff
and the full text of the original and policy-submitted `config.yaml`.

For a viewer-ready real M6 bundle, add an explicit trusted artifact directory:

```bash
python -m examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e \
  ... --m6-placement --max-submissions 2 --artifacts-dir /tmp/dtap-run

dtap-traj /tmp/dtap-run -o /tmp/dtap-run/trajectory.html
```

Export is opt-in and occurs only after the policy stream passes the existing
token/path leakage check. Normal M4/M6 rollout cleanup remains unchanged.
H counts only DTAP evaluations for which the victim execution actually starts.
`INVALID_SUBMISSION`, schema/placement rejection, and pre-victim infrastructure
failure do not decrement H. Q remains a separate bounded submit-call guard.

Each H-consuming victim run is retained under
`attempts/attempt-0001`, `attempts/attempt-0002`, and so on. The top-level
submitted config, victim trace, and judge files remain latest-attempt aliases
for older tooling. The explorer exposes an attempt selector and keeps the one
policy trajectory shared across the full H-loop.

A complete 24-case Linux live run is checked in at
[`artifacts/p0-p2-live-matrix-20260906`](artifacts/p0-p2-live-matrix-20260906/README.md).
It contains viewer-ready policy and OpenClaw victim trajectories, the original
and submitted configs, placement receipts, and judge results.
