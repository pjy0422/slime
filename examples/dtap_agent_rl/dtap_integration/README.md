# DTAP integration overlay

This directory keeps the DecodingTrust-Agent changes required by the slime M6
harness under the same version control as the harness.

The patch is based on DecodingTrust-Agent commit
`e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` and provides:

- out-of-tree candidate task support for workflow judge helper imports; and
- opt-in disabling of shared default Docker ports so parallel M4 children can
  use the disjoint port ranges assigned by slime.
- M5 zero-LLM route verification for bridge and host-network environments;
- independent placement read-back for all 115 enabled Linux mutators across 25
  injection MCPs, with unknown tools failing closed;
- a trusted OS Filesystem injection credential fix (the prior empty value
  produced an invalid `Authorization: Bearer ` header); and
- the Research injection route fix for TaskExecutor's dynamic arXiv port.
- an isolated OpenClaw 2026.9 evaluation profile, including an
  Anthropic-compatible custom provider for hosted Ollama models and a safe
  dialogue-trajectory fallback for detached headless sessions; and
- configurable inner and outer OpenClaw MCP deadlines, plus a deterministic
  Medical auxiliary-model fallback for providers whose model names the
  published Hospital image does not recognize; and
- M7's loss-aware `trace.completed` marker for deterministic negative victim
  tool/access observations. The marker is omitted after any sink write failure;
  and
- M7 v2's in-memory probes for all 115 enabled Linux mutators. Finance uses
  generated IDs and exact victim read locators; the remaining adapters report
  payload inclusion from an explicitly mapped victim MCP without claiming an
  exact locator. Prompt, tool, and skill presentation is observed at OpenClaw's
  provider-request assembly boundary. Raw locators and payloads are never
  retained in MCP events.
- a service-free M7 feedback-boundary matrix over 24 alternate benchmark tasks
  (one direct and one indirect case per Linux domain). It independently checks
  payload return and provider presentation, including multiline payloads and
  the actual `customer_service`, `Research`, `travel-suite`, and
  `HospitalClient` MCP namespaces.
- OpenClaw provider token receipts retained in trajectory metadata, including
  exact reasoning-token counts when the provider supplies them. Missing
  reasoning breakdowns remain unavailable rather than being fabricated.
- structured task/attack judge availability statuses. Reward gating consumes
  these statuses instead of matching provider-error phrases in judge prose.
- exact `(server, tool)` matching for tool-description presentation. A tool name
  appearing only inside another tool's description no longer counts as model
  presentation evidence.

Read-back uses the matching victim MCP's task-scoped credential when required.
Gmail verification resolves Mailpit list entries to the message-detail endpoint
before checking recipient, subject, and body.

Apply it to a DTAP checkout with:

```bash
examples/dtap_agent_rl/dtap_integration/apply.sh \
  /path/to/DecodingTrust-Agent
```

The script is idempotent: it reports an already-applied patch without changing
the checkout. It deliberately does not commit inside the DTAP repository.

`DTAP_ENV_VERIFICATION` accepts `off`, `route` (default), or `placement`.
`DTAP_ENV_VERIFICATION_STRICT=1` makes an unsupported placement adapter fail the
attempt. Verification failures are infrastructure failures and never reward zero.

## Placement support and handoff

The source of truth is `SUPPORTED_PLACEMENT_TOOLS` in the DTAP overlay and is
checked against every enabled Linux injection server by
`scripts/audit_m6_adapter_coverage.py`. The current audit covers 115 mutators in
25 injection MCPs with no unknown or stale registry entry:

| Adapter group | Verified mutators |
| --- | ---: |
| Customer Service, Salesforce, Legal, Finance | 31 |
| Gmail, Slack, Calendar, Zoom, Google Docs/Form | 24 |
| WhatsApp, PayPal, Ecommerce, Custom Website, Travel | 13 |
| Snowflake, Databricks, Telecom | 23 |
| OS Filesystem, Terminal, Research | 12 |
| Atlassian, Telegram, GitHub, Hospital | 12 |
| **Total** | **115** |

Read-back is independent of the injection acknowledgement: HTTP adapters query
the victim-visible entity, database adapters issue bounded table reads, and file
adapters inspect the exact TaskExecutor-owned container. Mismatch responses use
the caller's submitted locator and return only location-bearing `repair.fields`.
Every discovered MCP tool is classified by exact name. Tools with independent
read-back are `verified`; observation, maintenance, and intentionally excluded
guest automation tools are listed in `NON_PLACEMENT_TOOLS` and remain
`not_applicable`. A tool in neither registry fails closed as `unsupported`, and
the inventory audit fails until that new name is deliberately classified. No
prefix or naming-convention heuristic participates in this boundary.
Adapter-family routing is also enumerated with exact per-tool sets. Import-time
registry assertions fail if a Customer Service, Salesforce, Calendar, Zoom,
Google Form, or Finance handler set drifts from its supported-tool registry.

M4/M6 policy projection keeps exact qualified names and JSON schemas but reduces
long tool docstrings to their first semantic paragraph. For the representative
CRM task this reduces `get_attack_surface` from 64,812 to 35,954 bytes, below
Claude Code's persisted-output threshold; the policy therefore does not need a
forbidden filesystem/Monitor tool to read its own action surface.

Chase, Robinhood, Reddit, Google Sheets, and Google Drive entries are disabled in
the overlay because this DTAP revision contains no corresponding
`env_injection.py`. Advertising them as enabled produced an unusable surface.
They should be re-enabled only together with an implementation, route mapping,
independent read-back adapter, and positive/mismatch tests.

To add or classify a tool:

1. Confirm the tool is meant to be a policy placement action. If not, add its
   exact server/tool pair to `NON_PLACEMENT_TOOLS` with a short rationale.
2. For a placement action, identify a read-only API, database query, or
   container file read that does
   not reuse the injection MCP result as evidence.
3. Add its port binding to `ROUTES` and target environment to `TARGETS` if they
   are absent. Test both bridge port publishing and host networking.
4. Reuse task-scoped victim credentials via `build_readback_environments`;
   never log tokens or the injected payload.
5. Locate the written entity by a stable key, then compare the semantic fields.
   For list APIs such as Mailpit, fetch the detail record before comparing.
6. Return only status, locator, and SHA-256 digest. A mismatch or unavailable
   read path must raise `VerificationError`, not silently become `unsupported`.
7. Add positive, mismatch, authentication, and strict-mode unit tests, followed
   by a real indirect-task smoke test.

Windows and macOS remain outside the default Linux release gate, but the guest
APIs now provide independent read-back for all mutators referenced by current
indirect tasks: seven Windows file/registry/Office/image tools and six macOS
file/profile/plist/cron/image tools. Other platform mutators remain fail-closed, and the live platform
matrix stays opt-in until VM-backed strict smokes are reproducible.

## M6 receipt boundary

M6 exposes placement only through an episode-owned opaque action receipt. The
caller first supplies a fully schema-validated environment action to
`apply_attack_step`; `validate_placement` accepts only the returned `action_id`.
It accepts no environment name, backend query, filesystem path, or payload.

Successful read-back returns `validated_placement_locator`. Failed read-back
returns `requested_placement_locator`, a stable error code, and only the fields
that locate the entity (for example `kwargs.matter_id`, `kwargs.city` plus
`kwargs.name`, or `kwargs.file_path`). The locator is derived from the caller's
own submitted action. It is a targeted repair hint, not a backend listing or a
claim that some unverified alternate path exists.

Each apply uses a disposable sandbox and a sealed result file. Unknown fields,
symlinks, oversized output, missing results, and non-placement exceptions fail
closed as `EVALUATION_UNAVAILABLE`. Receipts are held only in the episode's
authority and disappear when that authority is unregistered.

## M7 victim-access evidence

The overlay's MCP event log remains best-effort. M7 adds an explicit completion
marker emitted only on normal proxy shutdown after a loss-free event stream.
This lets slime distinguish a proven absence from missing telemetry. The policy
projection and exact field meanings are documented in
`examples/dtap_agent_rl/M7_ADAPTIVE_FEEDBACK.md`.

The v2 registry explicitly classifies the same 25 Linux injection MCPs and 115
mutators as M6. Finance covers all HTML variants, `create_news`,
`create_comment`, and `inject_news`; a generated article ID remains in memory
and can match a later `browse_article` call. Other Linux adapters use explicit
per-mutator payload fields and victim MCP mappings to prove whether a successful
result contained the injected payload. Their locator remains `unknown` unless a
stable correlation field exists. A bounded `feedback.observed` event contains
only the submitted step index, public tool-sequence references, nullable facts,
and reason codes. Unknown tools remain unsupported; there is no name heuristic.

For prompt, tool-description, and skill injections, presentation is checked
only against `prompt.submitted`, `context.compiled.availableTools`, or
`context.compiled.systemPrompt`. Skill use remains `unknown` because OpenClaw
does not emit a trusted structured skill-use event.

## Reproducing M0 through M7

Start from the DTAP virtual environment and apply the slime-managed overlay:

```bash
source /home/pjy0422/workspace/dtap/bin/activate
cd /home/pjy0422/workspace/slime

examples/dtap_agent_rl/dtap_integration/apply.sh \
  /home/pjy0422/workspace/DecodingTrust-Agent

export PYTHONPATH=/home/pjy0422/workspace/DecodingTrust-Agent:${PYTHONPATH:-}
```

Several upstream task `setup.sh` fixtures use `jq` to JSON-encode seed SQL.
The release runtime pins the official Linux amd64 `jq-1.8.2` binary and its
SHA-256 in `runtime-lock.json`; install it on `PATH` before running the matrix.
The release gate rejects a missing or different binary instead of continuing
with partially seeded environments.

```bash
curl -fsSLo /tmp/jq-linux-amd64 \
  https://github.com/jqlang/jq/releases/download/jq-1.8.2/jq-linux-amd64
echo 'b1c22172dd303f3be49e935aa56aa48a8b7a46e0bc838b4997d3bb451495870f  /tmp/jq-linux-amd64' \
  | sha256sum --check --strict
install -m 0755 /tmp/jq-linux-amd64 /path/on/PATH/jq
```

Run the deterministic suites:

```bash
pytest -q examples/dtap_agent_rl/tests
pytest -q /home/pjy0422/workspace/DecodingTrust-Agent/tests/test_env_verification.py
pytest -q \
  /home/pjy0422/workspace/DecodingTrust-Agent/tests/test_openclaw_mcp_events.py \
  /home/pjy0422/workspace/DecodingTrust-Agent/tests/test_feedback_observation.py \
  /home/pjy0422/workspace/DecodingTrust-Agent/tests/test_feedback_boundary_matrix.py
```

The boundary matrix does not call a policy, victim LLM, environment, judge, or
Digestor. The live matrix uses benchmark record zero; this suite selects a
later record for every domain/threat-model pair and prefers environment-bearing
records. Its 100 parametrized checks cover provenance, positive evidence,
payload absence, result-versus-model-presentation separation, runtime MCP name
aliases, and write-echo rejection.

Run real M5 indirect-task smokes without a victim LLM or judge:

```bash
# File placement: expected verified in strict mode.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/research/malicious/indirect/Radiological_Risks/2 \
  --strict

# Gmail/Mailpit placement: expected verified in strict mode.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/finance/malicious/indirect/client_targeted_scam/1 \
  --strict

# Slack and Legal matter overlay: both expected verified in strict mode.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/legal/malicious/indirect/strategy_leak/6 \
  --strict

# Travel accommodation placement: expected verified in strict mode.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --strict

# OS Filesystem placement from the exact sandbox container: expected verified.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/os-filesystem/malicious/indirect/prohibited-ai-practices-and-profiling/11 \
  --strict
```

The originally inspected `finance/.../action_reversal/1` task contains only a
`type: tool` attack step and therefore reports `no_environment_injections`; it
does not exercise M5. The smoke command always tears down the injection MCP
processes and Docker environments in a `finally` block.

Run a real generated-plan episode (policy, victim, placement gate, and judge):

```bash
export ANTHROPIC_API_KEY='...'
export DTAP_POLICY_ANTHROPIC_BASE_URL=https://provider.example
export DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN=1
export DTAP_VICTIM_ANTHROPIC_BASE_URL=https://provider.example
export DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN=1
export DTAP_ENV_VERIFICATION=placement
export DTAP_ENV_VERIFICATION_STRICT=1

python -m examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python \
  --policy-model glm-5.2 \
  --victim-model glm-5.2 \
  --victim-agent-type openclaw \
  --max-submissions 2 \
  --m6-placement \
  --feedback-mode final+deterministic
```

For the optional semantic mode, set
`DTAP_DIGESTOR_ANTHROPIC_BASE_URL` (or reuse the policy base URL) and run with
`--feedback-mode final+deterministic+digestor`. `--reasoning-summary` is a
separate opt-in and is valid only in that mode. The summary reports only an
explicitly retained reasoning/rationale source; it never reconstructs hidden
chain-of-thought. Provider usage reports calls, failures, tokens, and latency;
cost remains unknown unless an external pinned price table is introduced.

This prompt contains no plan or payload. It requires the policy to inspect the
live task/surface, author and validate every step, and submit exactly once. An
attack miss is a valid reward-zero episode; infrastructure/security failure is
not. Set `DTAP_M5_GLM_TASK_DIR` and `DTAP_M5_DTAP_ROOT` to include the same gate
in the opt-in pytest integration suite.

M5 remains a trusted auxiliary check. It does not expose read-back to the policy,
does not issue policy-visible `action_id` receipts, and does not alter reward.
Those policy-scoped receipt semantics belong to M6.

Run the real M6 receipt path (no victim or judge):

```bash
python -m examples.dtap_agent_rl.scripts.smoke_m6_placement \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/legal/malicious/indirect/strategy_leak/6 \
  --tool legal-injection:inject_into_matter \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python

python -m examples.dtap_agent_rl.scripts.smoke_m6_placement \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python

python -m examples.dtap_agent_rl.scripts.smoke_m6_placement \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/os-filesystem/malicious/indirect/prohibited-ai-practices-and-profiling/11 \
  --turn-id 1 \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python
```

The OS example's source template places the global environment injection at turn
3 while its projected task is one instruction; `--turn-id 1` matches the public
M2-M6 candidate contract and does not change placement semantics.

The Linux adapter hardening was also exercised against real strict-placement
tasks for Legal, Travel, OS Filesystem, Salesforce, Hospital, PayPal, Snowflake,
Telecom, Telegram, and Atlassian. In particular, Atlassian must use task fixture
`workflow/.../sensitive-file-deletion/010`, whose allowlist contains
`atlassian-injection`; fixture `016` enables environment attacks in the abstract
but does not advertise an environment injection MCP and is correctly rejected
before application. The Atlassian read-back authenticates with the victim's
task-scoped token and checks both the returned issue fields and its project key.

For the generated-plan M6 release gate, use the same provider variables as the
M5 command and add `--m6-placement`. Set `DTAP_M6_GLM_TASK_DIR` and
`DTAP_M6_DTAP_ROOT` to run the opt-in pytest gate.

### Direct/indirect domain matrix with OpenClaw

Install the OpenClaw CLI first and keep credentials in environment variables;
the generated evaluation profile stores only an `ANTHROPIC_API_KEY` reference.
OpenClaw 2026.9 detached runs do not always expose their SQLite transcript as a
JSONL file. The proxy therefore writes a task-scoped append-only event stream
containing tool/server names, timestamps, status, redacted argument shapes, and
stable digests. Credentials, sandbox paths, raw arguments, and raw results are
never persisted. The bundle manifest and proxy events share one public episode
identifier, so the viewer can correlate a detached victim run with its policy,
submitted YAML, placement receipts, and judge artifact.

The following command selects one benchmark task for each of the 12 Linux
domains and each threat model, runs disjoint DTAP port ranges in parallel, and
writes a resumable result tree. `passed` means the complete policy, placement,
victim, and judge pipeline ran without an infrastructure/security failure; the
attack may legitimately have reward zero.

```bash
export ANTHROPIC_API_KEY='...'
export DTAP_POLICY_ANTHROPIC_BASE_URL=https://ollama.com
export DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN=1
export DTAP_VICTIM_ANTHROPIC_BASE_URL=https://ollama.com
export DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN=1

python -m examples.dtap_agent_rl.scripts.smoke_m6_domain_matrix \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --slime-root /home/pjy0422/workspace/slime \
  --artifacts-root /home/pjy0422/workspace/dtap-results/m6-domain-matrix \
  --policy-model deepseek-v4-flash \
  --victim-model deepseek-v4-flash \
  --victim-agent-type openclaw \
  --max-parallel 4 \
  --port-range-start 36000 \
  --port-range-stride 1024
```

Use `--resume` to skip cases whose `result.json` already says `passed`.
Per-case stdout, stderr, original/submitted YAML, policy trajectory, victim
trajectory, and result live below `<artifacts-root>/<domain>/<threat-model>/`;
the aggregate report is `<artifacts-root>/summary.json`. A failed selected case
always produces a nonzero exit code. Windows/macOS are not selected by default
and are recorded under `excluded_platform_domains`. Missing result files are not
counted as completed cases; malformed or mismatched result files count as failed.
For subset runs, `selected_failed` controls the exit code, while `failed` covers
all stored cases (which may originate from earlier runs).

The report-only regression tests require no model API or victim runtime:

```bash
source /home/pjy0422/workspace/dtap/bin/activate
pytest -q examples/dtap_agent_rl/tests/test_m6_domain_matrix.py
```

Four concurrent slots
consume ranges `36000-36511`, `37024-37535`, `38048-38559`, and
`39072-39583`. Reduce `--max-parallel` if the model endpoint rate-limits.

Placement remains strict. Consequently a task whose generated plan uses a
mutating adapter listed as unsupported above fails as
`EVALUATION_UNAVAILABLE`; Windows and macOS are expected to remain fail-closed
until DTAP provides guest-side read-only verification endpoints.

M6 also enforces receipt provenance at final submission: every environment step
must exactly match one action applied by the same episode and positively read
back with `validate_placement`. One receipt cannot authorize duplicated steps.
An unverified final action is rejected as a nonterminal `INVALID_SUBMISSION`,
spends Q rather than H, and can be omitted or repaired before retrying. The E2E
smoke raises Claude Code's MCP tool deadline to its explicit evaluation timeout
because `submit_attack` synchronously waits for the victim and judge.

H is the number of victim executions, not the number of `submit_attack` calls.
Every `INVALID_SUBMISSION` leaves `remaining_submissions` unchanged. Q is an
independent safety limit on total calls and should be configured with repair
headroom; the live smoke uses `max(6, 3 * H)`.

For Medical tasks, the managed Hospital entrypoint routes the explicitly
configured patient and diagnosis-judge model through the same trusted
OpenAI-compatible provider used by the victim. This is required for model names
outside the published image's GPT/Claude/Gemini dispatch list. Do not use
`DTAP_MEDICAL_AUX_MODE=deterministic` in a scored run: the patient simulator and
diagnosis judge share one container, so removing its credentials also disables
the judge. Judge transport/provider errors are rejected as infrastructure
failures and never converted to an `attack_success=false` reward.

The trusted child also exports the shared provider as `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `OPENAI_MODEL`, and `JUDGE_MODEL`. This covers Research's
raw OpenAI-compatible judge and the common BaseJudge path used by Travel and
other domain judges; Customer Service already reads `JUDGE_MODEL`. Research
accepts provider URLs both with and without a trailing `/v1`.
Medical judge HTTP calls use a judge-scoped 180-second minimum timeout so a
loaded parallel provider is not mistaken for a negative verdict. Finance
`create_news` placement extracts the backend-generated article ID and verifies
the exact victim-visible article page.
`OPENCLAW_MCP_TOOL_TIMEOUT_SECONDS` controls both the proxy call deadline and
OpenClaw's generated `requestTimeoutMs` (default 60 seconds; the E2E runner uses
600 seconds).

### Linux release gate

The pinned runtime, clean overlay application, deterministic suites, adapter
audit, and 24 generated-plan E2E cases can be reproduced with one command:

```bash
python -m examples.dtap_agent_rl.scripts.run_m6_linux_release_gate \
  --dtap-root /path/to/DecodingTrust-Agent \
  --slime-root /path/to/slime \
  --artifacts-root /path/to/artifacts \
  --max-parallel 4
```

Use `--skip-images --skip-e2e` for the fast deterministic gate. The full gate
requires `ANTHROPIC_API_KEY` in the process environment and never reads or writes
a secret-bearing file. Runtime versions and image digests are pinned in
`runtime-lock.json`; the corresponding Python boundary is installable from
`requirements-release.txt` after DTAP's own pinned requirements.

### Verified matrix result (2026-09-05)

The earlier DeepSeek/OpenClaw run completed all 24 Linux direct and indirect
cases without infrastructure or security failures. Windows and macOS are now
excluded instead of being counted as four expected failures. This is execution
coverage, not attack success: reward-zero/exhausted episodes are valid completed
evaluations.

The retained report is
`/home/pjy0422/workspace/dtap-results/m6-deepseek-openclaw-domain-matrix-20260905/summary.json`.

For macOS, prepare the downloaded snapshot-only qcow2 once before a live run:

```bash
python -m examples.dtap_agent_rl.scripts.prepare_macos_baseline \
  --source /path/to/downloaded/macos \
  --output /path/to/macos-cold-boot
export MACOS_DATA_DIR=/path/to/macos-cold-boot
```

The source is never modified. The prepared disk cold-boots through a
per-container thin overlay, so parallel tasks cannot share guest writes.
