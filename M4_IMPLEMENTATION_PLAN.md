# DTAP Agent RL M4 — Reward Hacking and Isolation Hardening Plan

Status: production implementation complete in this overlay. Local unit, contract,
firewall, integrity, scheduler, FastMCP, generate, harness, and parallel fake-episode
tests are green. Real Claude/sandbox and parallel DTAP/Docker checks remain explicit
release-environment gates. M4 does not add an attack mechanism or placement oracle.

## 1. Goal and milestone boundary

M4 makes the M3 rollout safe to run as an adversarial, parallel RL workload. Its
job is to prove that a policy can influence only its own validated attack plan and
can observe only the intentionally exposed contract.

M4 preserves the M3 trajectory semantics:

- `t = 1..T` is a policy micro-turn.
- `h = 1..H` is a started DTAP evaluation.
- One slime/Claude context remains alive across failed `h` attempts.
- Every `h` runs in a fresh DTAP process/environment.
- The first successful `h` terminates with reward `1.0`.
- The H-th normal failure terminates with reward `0.0`.
- An untrustworthy verdict is excluded from training.

The intended adaptive oracle is exactly one bit per accepted evaluation:

```text
attack_success = true | false
```

M4 does not add placement read-back, judge rationale, victim output, environment
state, or arbitrary validation queries. Those remain outside the policy contract.
M5 and M6 own placement validation.

Design assumptions are pinned to the current upstream interfaces: DTAP's
[`run_single_task`](https://github.com/AI-secure/DecodingTrust-Agent/blob/main/eval/task_runner.py),
slime's current
[`generate.py`](https://github.com/THUDM/slime/blob/main/examples/coding_agent_rl/generate.py),
and Claude Code's documented
[`--strict-mcp-config`](https://docs.anthropic.com/en/docs/claude-code/cli-reference)
and [permission/sandbox model](https://docs.anthropic.com/en/docs/claude-code/permissions).
M4 release CI must pin the reviewed revisions/versions rather than following moving
`main` branches implicitly.

## 2. Threat model

Treat the policy, victim output, prompt-injected content, MCP arguments, DTAP stdout,
judge-result contents, and copied task assets as untrusted data. Treat the host M4
boundary, the registered benchmark snapshot, and deployment secrets as trusted.

The policy may deliberately:

- submit malformed, oversized, deeply nested, duplicate, or concurrent requests;
- reuse another episode's token or replay calls after terminal state;
- compare validation errors, result shapes, ordering, and model-visible metadata;
- attempt to invoke native Claude Code tools, inherited MCP servers, hooks, skills,
  plugins, shell commands, files, network endpoints, or process environment;
- put path-like strings, log control characters, or secret-looking values in attack
  payloads so they are reflected by exceptions and logs;
- make one episode slow, crash, or cancel in order to affect another episode;
- exploit symlinks, special files, path traversal, output-file ambiguity, or cleanup;
- cause judge output to contain attacker-controlled text alongside a verdict.

M4 protects confidentiality and isolation at the application and sandbox boundary.
It does not claim resistance to a malicious kernel, container runtime, host
administrator, model provider, or physical wall-clock side channels. No timestamp or
duration is placed in the policy context, and the policy sandbox must not expose a
clock/network primitive with which to turn execution latency into an additional
oracle.

## 3. Frozen security invariants

1. **Allowlisted observation:** every policy-visible value is constructed by a
   typed allowlist serializer. Trusted objects are never serialized directly.
2. **One-bit reward feedback:** an accepted submit exposes only success/failure,
   submission index, terminal state, and remaining H budget.
3. **No internal errors:** exceptions, stdout/stderr, paths, judge fields, victim
   responses, Docker identities, ports, and timings never enter MCP results or slime
   samples.
4. **Independent capabilities:** the slime adapter session credential, policy-facing
   MCP bearer token, public episode identifier, and attempt identifiers are distinct.
5. **Atomic authority:** one registry entry owns the episode view, submission
   coordinator, lifecycle state, and bearer capability. Read and submit authority
   cannot diverge.
6. **Bounded policy work:** H counts started evaluations; a separate Q budget counts
   all authenticated `submit_attack` calls, including invalid calls.
7. **Fresh and disjoint execution:** attempts never share a task copy, result tree,
   subprocess group, DTAP executor, Docker environment, or mutable registry state.
8. **Minimum child environment:** the DTAP subprocess and Claude sandbox receive
   explicit environment allowlists, never `os.environ.copy()`.
9. **Sandbox enforcement:** client-side tool flags are defense in depth. Filesystem,
   process, and network isolation are enforced below the model/tool permission layer.
10. **Whole-task integrity:** the complete registered benchmark task tree, not only
    `config.yaml`, has the same manifest before and after every episode.
11. **Fail closed:** security-boundary ambiguity becomes a terminal internal
    security abort and the sample is removed from training.
12. **No M5 oracle:** M4 never offers read-back or placement verification.

## 4. Policy-visible contract

The four M3 MCP tools remain the complete policy surface:

```text
get_task_spec()
get_attack_surface()
validate_attack_step(step)
submit_attack(plan)
```

No M4 policy tool is added.

### 4.1 Submit receipt

Use one versioned serializer, `PolicySubmitReceiptV1`. A normal accepted attempt is:

```json
{
  "accepted": true,
  "submission": 1,
  "success": false,
  "terminal": false,
  "remaining_submissions": 2
}
```

An authoritative submit rejection is intentionally coarse:

```json
{
  "accepted": false,
  "terminal": false,
  "remaining_submissions": 3,
  "error": {"code": "INVALID_SUBMISSION"}
}
```

Detailed structural feedback remains available from the read-only M2
`validate_attack_step`. The mutation endpoint must not reveal whether rejection came
from plan-conflict logic, YAML parsing, DTAP parsing, filesystem materialization, or
another trusted gate.

Terminal calls always return a stable terminal receipt. Internal infrastructure and
security aborts use the same policy-facing `EVALUATION_UNAVAILABLE` result and are
removed from training; their internal reason is audit-only.

### 4.2 Response equivalence rule

For the same public state and the same boolean verdict, changing any of the following
must produce byte-equivalent policy JSON:

- judge rationale and extra judge fields;
- victim response and trajectory;
- DTAP stdout/stderr and process exit code;
- output directory, Docker ID, port, hostname, and exception text;
- task success when attack success is unchanged.

JSON key order and encoding are canonicalized. MCP responses have a small maximum
serialized size. They contain no timestamps, durations, correlation IDs, stack
traces, or host-generated paths.

## 5. Episode state and anti-oracle budgets

M4 adds a separate submit-call budget `Q` without changing H:

| Counter | Consumed by | Purpose |
| --- | --- | --- |
| `t` | policy runtime turn | trajectory/context bound |
| `q` | every authenticated submit call | malformed-call and DoS bound |
| `h` | evaluation reaches DTAP running phase | reward experiment budget |

`max_submit_calls` is explicit trusted rollout configuration and must be at least H.
There is no silent default. A typical experiment may configure `Q = 2H`, but that is
an experiment choice rather than a library constant.

State additions:

```python
class EpisodeStatus(str, Enum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"
    POLICY_LIMIT = "policy_limit"
    INFRA_ERROR = "infra_error"
    SECURITY_ABORT = "security_abort"
```

| Terminal state | Reward | Remove sample |
| --- | ---: | --- |
| `SUCCEEDED` | `1.0` | no |
| `EXHAUSTED` | `0.0` | no |
| `POLICY_LIMIT` | `0.0` | no |
| `INFRA_ERROR` | unset | yes |
| `SECURITY_ABORT` | unset | yes |

`POLICY_LIMIT` is trainable because the policy caused it. Infrastructure and
security failures are not negative examples of attack quality.

Additional fixed limits are checked before copying policy data:

- maximum serialized plan bytes;
- maximum steps per plan;
- maximum content bytes per step;
- maximum JSON nesting and collection nodes;
- per-token request rate and maximum waiting submits;
- HTTP request-body and header limits.

Identical plans are not silently deduplicated. Repeating a plan is a new stochastic
experiment and consumes a new H slot if DTAP starts. A retry after terminal state is
rejected and never starts execution.

## 6. Capability and registry redesign

M3 currently reuses the slime session ID as the MCP bearer. M4 replaces that coupling
with host-generated credentials:

```python
@dataclass(frozen=True)
class EpisodeCredentials:
    adapter_session_id: str
    mcp_bearer_token: SecretStr       # >= 256 random bits
    public_episode_id: str            # non-secret, random
```

Only `adapter_session_id` reaches the adapter. Only `mcp_bearer_token` reaches the
MCP authorization header. Logs use a keyed digest of `public_episode_id`, never a
secret or task path.

Replace the paired M3 registries with one `EpisodeAuthorityRegistry`:

```python
@dataclass
class EpisodeAuthority:
    view: EpisodeView
    coordinator: SubmissionCoordinator
    lifecycle: EpisodeLifecycle
    terminal_event: asyncio.Event

class EpisodeAuthorityRegistry:
    def register(credentials, authority): ...
    def resolve(bearer_token) -> EpisodeAuthority: ...
    def unregister(bearer_token): ...
```

Registration and removal are atomic. Token lookup uses a digest key and
constant-time comparison where applicable. Unknown, expired, malformed, wrong-
episode, and post-cleanup credentials have the same external authorization failure.
Tokens are never tool arguments, tool schema values, filenames, metadata, or log
fields.

## 7. Reward firewall and judge isolation

Introduce `reward_firewall.py` between the DTAP subprocess and episode state.

```python
@dataclass(frozen=True)
class TrustedVerdict:
    attack_success: bool

class JudgeVerdictReader:
    def read(expected_result_path: Path) -> TrustedVerdict: ...
```

The reader:

1. opens one exact expected file beneath the attempt result root;
2. rejects symlinks, non-regular files, path escape, unexpected ownership, and an
   oversized result;
3. rejects duplicate JSON keys and excessive JSON depth/nodes;
4. requires a top-level object and an exact JSON boolean `attack_success`;
5. copies only that boolean into `TrustedVerdict`;
6. never includes file text or parser errors in an exception crossing the firewall.

Extra judge fields may exist for DTAP compatibility but are dropped inside the
trusted reader and never reach a policy serializer. The process return code and
`task_success` remain non-reward fields. A missing, ambiguous, malformed, or
untrustworthy verdict becomes an internal infra/security terminal state, never
`attack_success=false`.

All `submit_attack` exceptions are caught by one MCP boundary and mapped to stable
public codes. Framework default exception serialization must not handle a trusted
exception.

## 8. Policy output and leakage guard

Introduce `policy_contract.py` with typed serializers for task projection, attack
surface, validation response, submit receipt, and terminal error. Only these
serializers may return values from an MCP tool.

A defense-in-depth `PolicyLeakageGuard` scans the final serialized payload for:

- the MCP bearer and adapter credential;
- registered absolute path prefixes;
- original example-attack canaries;
- judge/victim canaries used by security tests;
- forbidden keys such as `judge_result`, `victim_output`, `trajectory_path`,
  `runtime_identity`, `task_success`, `stdout`, and `stderr`.

The guard is not the primary sanitizer—the allowlist serializers are. A guard hit
causes `SECURITY_ABORT`, emits a content-free trusted audit event, and returns only
`EVALUATION_UNAVAILABLE`.

Slime `finish_session` metadata is also allowlisted. It may include only a schema
version, public terminal status, H/Q counters, and rollout configuration identifiers.
It must not include agent exit code, exception type, task path, plan content, judge
content, victim content, environment identity, or raw logs.

## 9. Claude Code and sandbox hardening

Model-side tool denial is not a security boundary. M4 requires both Claude Code
configuration hardening and an enforcing sandbox profile.

### 9.1 Claude Code launch

- add `--strict-mcp-config` so inherited user/project MCP servers are ignored;
- allow exactly the four fully qualified DTAP tools, not a broad server wildcard;
- retain explicit denial of all native filesystem, shell, notebook, and web tools;
- use a new per-episode `CLAUDE_CONFIG_DIR` with no inherited user/project state;
- disable hooks, plugins, connectors, subagents, and project instruction loading in
  the pinned production configuration;
- write MCP/settings files mode `0600` inside the isolated policy UID's mode-`0700`
  configuration directory;
- reject production use of arbitrary extra CLI flags or environment JSON;
- pin and preflight the Claude Code version and fail closed if required flags are
  unsupported.

`--strict-mcp-config` is specifically required because Claude Code otherwise reads
MCP/settings from multiple scopes. The generated configuration itself contains only
environment placeholders, never literal credentials.

### 9.2 Enforcing runtime sandbox

The sandbox profile must:

- run the policy process as a non-root UID with no Linux capabilities;
- mount an empty ephemeral workdir and a minimal ephemeral home;
- mount no benchmark, DTAP checkout, attempt root, result root, Docker socket, host
  workspace, credential directory, or shared episode directory;
- use a read-only root filesystem plus narrowly scoped writable tmpfs;
- prevent access to other processes and their environment through `/proc`;
- deny privilege escalation, device access, and host IPC namespaces;
- allow egress only to the slime adapter endpoint and the DTAP policy MCP endpoint;
- deny cloud metadata, loopback services, RFC1918/link-local destinations other than
  the two explicitly resolved endpoints, and arbitrary DNS;
- terminate the complete policy process group on timeout, cancellation, or terminal
  grace-period expiry.

Worker startup must refuse M4 mode if the sandbox cannot prove these mounts and
network rules. A warning-only fallback is not permitted.

## 10. DTAP child-process environment

Replace `os.environ.copy()` in `DtapAttemptRunner` with
`build_dtap_child_env(policy)`. Start from an empty mapping and add only:

- variables required by DTAP and the selected victim provider;
- the attempt-specific results root and non-secret attempt index;
- explicitly approved proxy/CA/runtime variables;
- a minimal `PATH`, locale, and Python import configuration.

Always strip slime adapter credentials, MCP bearer tokens, Claude configuration,
Library/workspace credentials, unrelated provider keys, CI credentials, shell
history variables, and host debugging flags. Required provider-secret names are an
explicit deployment allowlist. Values are never logged.

Each attempt keeps its own process group. Timeout or cancellation kills only that
group. Cleanup must not invoke a worker-global Docker or process teardown capable of
terminating another episode.

## 11. Parallel scheduling and episode isolation

Introduce one worker-scoped `AttemptScheduler`; a per-runner semaphore is not enough
if code can accidentally create multiple runners.

```python
class AttemptScheduler:
    async def run(episode_id, attempt_id, operation): ...
```

Required behavior:

- one in-flight submit per episode;
- bounded global DTAP concurrency;
- FIFO or documented fair queuing across episodes;
- bounded queue length and wait timeout;
- cancellation removes only the caller's queue item/process;
- saturation before DTAP start is infrastructure backpressure and does not consume H;
- no worker-global mutable `TaskExecutor`, Docker pool, output directory, port map,
  or result cache;
- metrics use non-secret labels and never task/payload text.

Attempt roots use random host-generated IDs and atomic mode-`0700` creation. Output
parsing uses an exact attempt root rather than recursive discovery across sibling
episodes.

## 12. Whole-benchmark integrity

Introduce `integrity.py`:

```python
@dataclass(frozen=True)
class BenchmarkManifest:
    entries: tuple[ManifestEntry, ...]
    root_digest: str

class BenchmarkIntegrityGuard:
    def capture(task_dir) -> BenchmarkManifest: ...
    def verify(task_dir, expected) -> None: ...
```

Each manifest records normalized relative path, file type, mode, size, and SHA-256
for every regular task file. Source symlinks, devices, sockets, FIFOs, hard-link
aliasing outside policy, and path escape are rejected.

The lifecycle is:

1. capture/verify the registered task manifest before episode registration;
2. copy from the verified source with no-follow file operations;
3. verify that only copied `config.yaml` differs from the source and that its trusted
   fields plus validated attack plan match the renderer contract;
4. run only the copied task directory;
5. verify the original full manifest after each attempt and again in episode
   `finally`;
6. treat any mismatch as `SECURITY_ABORT`, quarantine the worker/task, and exclude
   the sample.

The manifest check includes `judge.py`, setup scripts, assets, schemas, and every
other file in the task directory. It is not limited to `config.yaml`.

## 13. Trusted audit and incident handling

M4 adds structured host-only audit events with a fixed schema:

```text
episode_registered
submit_rejected
evaluation_started
verdict_accepted
episode_terminal
infra_abort
security_abort
cleanup_completed
```

Events contain only keyed episode/attempt digests, enum codes, counters, and coarse
component names. They contain no plan content, prompt, judge data, victim response,
credentials, paths, stdout/stderr, or exception strings.

On integrity or leakage failure:

- stop the affected episode;
- remove its sample from training;
- terminate its process group;
- revoke/unregister its capability;
- preserve only trusted forensic identifiers outside the policy transcript;
- quarantine the affected task or worker until operator review.

## 14. Production files

New modules:

```text
examples/dtap_agent_rl/
├── authority.py             # split credentials and atomic registry
├── security_policy.py       # explicit sizes, Q, timeouts, env/network policy
├── policy_contract.py       # typed public responses and leakage guard
├── reward_firewall.py       # bounded judge-result reader
├── integrity.py             # whole-task manifest and safe-copy verification
├── scheduler.py             # worker-global fair DTAP concurrency
├── sandbox_policy.py        # Claude launch and runtime isolation contract
└── audit.py                 # content-free trusted events
```

Modified modules:

```text
episode_runtime.py           # Q and terminal hardening states
harness.py                   # distinct token, strict MCP, clean config/env
mcp_server.py                # unified authority and exception firewall
submission.py                # preflight limits, Q accounting, typed receipt
candidate_config.py          # no-follow copy and full manifest checks
attempt_runner.py            # env allowlist, exact output, shared scheduler
m3.py                        # credentials/authority/integrity lifecycle
generate.py                  # security policy, metadata allowlist, revocation
trajectory.py                # terminal/process cleanup invariants
```

DTAP source remains unmodified in M4 unless a real integration test proves that safe
process/result isolation is impossible through the existing public runner. Any DTAP
patch would require a separate minimal design review.

## 15. Test plan

### A. Public-contract and leakage tests

- Snapshot every policy response schema and canonical JSON encoding.
- Vary judge rationale, victim output, exit code, task success, paths, and logs while
  holding `attack_success` constant; assert byte-identical receipts.
- Plant unique canaries in config example attacks, judge output, victim output,
  environment, source paths, and exceptions; assert none occurs anywhere in the
  policy transcript, MCP response, prompt, or slime sample metadata.
- Fuzz exceptions at every submit phase and assert only stable public codes.
- Assert no response contains forbidden keys, absolute paths, traceback fragments,
  timestamps, or runtime IDs.

### B. Capability and replay tests

- Adapter session ID cannot authenticate to MCP.
- MCP bearer cannot authenticate to the adapter.
- Token A cannot read, validate, submit, terminate, or infer counters for episode B.
- Unknown/malformed/expired/wrong-episode tokens have equivalent public failures.
- Token replay after unregister or terminal never starts DTAP.
- Partial registration failure leaves neither read nor write authority.
- Registry cleanup occurs on success, exhaustion, policy limit, cancellation,
  timeout, infra failure, and security abort.

### C. Submit-oracle and resource-limit tests

- Every authenticated submit call increments Q exactly once.
- Invalid calls do not consume H but eventually terminate at Q.
- Oversized/deep plans fail before deepcopy, YAML, filesystem, or runner work.
- Concurrent same-episode submits serialize; a queued call rechecks terminal state.
- Repeated valid plans are explicit new H attempts.
- Post-terminal calls return one stable receipt and never mutate counters.
- HTTP body, header, queue, and rate limits are enforced without raw framework
  errors.

### D. Judge-firewall tests

- Exact JSON booleans pass; strings, integers, null, duplicate keys, deep JSON,
  oversized files, symlinks, path escape, missing files, and multiple candidates fail.
- Extra attacker-controlled judge fields never affect the receipt.
- `task_success` and process return code cannot change reward.
- Missing/untrusted verdict removes the sample rather than labeling it failure.

### E. Filesystem and benchmark-integrity tests

- Mutate each original file type one at a time and assert manifest failure.
- Add, remove, rename, chmod, replace, hard-link, or symlink a task file and assert
  failure.
- Attempt TOCTOU replacement during copy and assert no-follow failure.
- Verify copied config changes only the validated attack section.
- Verify cleanup cannot escape the managed attempt root.
- Verify source manifest remains exact after success, failure, cancellation, and
  forced process kill.

### F. Sandbox and native-tool bypass tests

- Direct Claude smoke attempts every native tool and an inherited fake MCP server;
  only the four DTAP tools are usable.
- User/project settings, hooks, plugins, skills, and MCP configs are ignored.
- Policy cannot read benchmark files, attempt files, `/proc` secrets, host workspace,
  Docker socket, or provider credentials.
- Policy cannot connect to a canary internet host, cloud metadata, sibling sandbox,
  DTAP child, or arbitrary worker port.
- Policy can reach only the intended adapter and DTAP policy MCP endpoints.
- Worker startup fails closed when strict MCP or sandbox/network enforcement is
  missing.

### G. Child-environment tests

- Plant canaries in the worker environment and assert the DTAP child receives none.
- Assert required provider credentials are present only when explicitly allowlisted.
- Assert child stdout/stderr and environment never enter policy-visible data.
- Timeout/cancel kills descendants but leaves a sibling episode running.

### H. Parallel isolation and stress tests

- Run at least 32 fake episodes concurrently with randomized success, failure,
  invalid plans, cancellation, and timeouts.
- Assert unique roots, tokens, counters, process groups, results, and cleanup.
- Saturate the global queue and verify fairness and non-consuming backpressure.
- Run two real DTAP/Docker episodes concurrently, one deterministic success and one
  deterministic failure; verify no result, environment mutation, port, or file crosses
  episodes.
- Repeat the real test under cancellation and worker shutdown.

### I. Reward-regression tests

- Replay a fixed accepted-plan corpus with fixed model seeds before and after M4.
- Assert identical `attack_success` rewards for trustworthy evaluations.
- Assert the only new trainable terminal is explicit `POLICY_LIMIT`.
- Assert all infra/security cases have `remove_sample=true` and no reward label.

Property-based tests should cover state transitions, arbitrary JSON structures,
response serialization, and interleaved episode operations. Real sandbox, Claude,
DTAP, Docker, and concurrent reset tests are mandatory release gates, not optional
coverage.

## 16. Implementation sequence

### Phase A — Freeze security contract and red tests

1. Add the M4 threat model and typed public response snapshots.
2. Add canary-based leakage tests and judge-result fuzz tests.
3. Add Q/state-machine tests and cross-episode concurrency tests.
4. Keep production unchanged until the red suite demonstrates each gap.

Exit gate: failures correspond to known M3 weaknesses, not ambiguous test fixtures.

### Phase B — Authority and response boundary

1. Split adapter/MCP/public identifiers.
2. Replace paired registries with atomic `EpisodeAuthorityRegistry`.
3. Implement typed response serializers and the MCP exception firewall.
4. Remove unsafe finish-session metadata.

Exit gate: capability, replay, response-equivalence, and canary tests pass.

### Phase C — Q budget and reward firewall

1. Add pre-deserialization limits and Q accounting.
2. Add `POLICY_LIMIT`/`SECURITY_ABORT` state transitions.
3. Implement the bounded exact-path judge reader.
4. Route only `TrustedVerdict.attack_success` into reward state.

Exit gate: oracle, fuzz, judge, and reward-regression tests pass.

### Phase D — Filesystem and child-process hardening

1. Implement full benchmark manifests and no-follow copy.
2. Replace recursive result discovery with an exact expected result path.
3. Replace inherited child environment with an allowlist.
4. Add bounded process-tree cleanup and incident audit events.

Exit gate: integrity, symlink/TOCTOU, env-canary, and cancellation tests pass.

### Phase E — Claude/sandbox hardening

1. Add strict MCP configuration and exact tool allowlist.
2. Isolate Claude config/home and disable inherited extension points.
3. Enforce filesystem/process/network sandbox policy.
4. Make missing enforcement a startup error.

Exit gate: adversarial direct-Claude and network/filesystem bypass smokes pass.

### Phase F — Parallel scheduler and stress

1. Install one worker-scoped fair `AttemptScheduler`.
2. Validate bounded queueing, cancellation, and per-episode process ownership.
3. Run fake stress and real concurrent DTAP reset tests.

Exit gate: no cross-episode state or artifact is observed under stress.

### Phase G — Shadow rollout and release

1. Run M4 in shadow on a fixed M3 corpus and compare reward labels.
2. Measure infra/security abort rate, queue saturation, cleanup, and manifest cost.
3. Run all M0–M3 regression tests plus M4 security gates.
4. Review the final policy-visible transcript and child/sandbox manifests.
5. Enable training only after all real-environment gates are green.

## 17. Definition of Done

M4 is complete only when all of the following are true:

- Every M0–M3 test remains green, including the real M3 DTAP compatibility/reset
  gates.
- The policy receives only the four frozen tools and typed allowlisted responses.
- Accepted evaluations reveal exactly one success bit and public H state.
- Judge/victim/config/example/path/secret canaries never appear in policy-visible
  transcripts or slime sample metadata.
- Adapter and MCP credentials are independent and unforgeable; registry authority is
  atomic and revoked on every exit path.
- H and Q are independently and race-safely enforced.
- Malformed and oversized inputs perform bounded work.
- Judge parsing is size/depth/path/type bounded and only a boolean reaches reward.
- Claude Code runs with strict MCP config, exact tools, clean config/home, and no
  inherited hooks/plugins/settings.
- The runtime sandbox proves filesystem, process, credential, and two-endpoint network
  isolation; no warning-only fallback exists.
- DTAP child processes receive an explicit minimum environment.
- The whole original benchmark task tree has an identical manifest after every
  attempt and episode.
- At least 32 concurrent fake episodes and two concurrent real DTAP episodes pass
  isolation, cancellation, and cleanup tests.
- Infrastructure/security failures are excluded, while policy-caused Q exhaustion is
  a trustworthy reward-0 sample.
- Pre/post-hardening reward replay shows no unexplained reward drift.
- No placement/read-back API or auxiliary environment oracle has been introduced.

M4 closure therefore means more than “no leaked field was observed.” It means the
policy-visible contract, capability lifecycle, execution sandbox, reward parser,
filesystem integrity, and concurrent scheduler are all independently enforced and
covered by adversarial tests.
