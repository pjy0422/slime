# DTAP Agent RL next steps

This checklist starts from the M6 release merged in PR #4. At that point the
DeepSeek/OpenClaw domain matrix completed 24 Linux-supported cases. Windows and
macOS are excluded from this scope. A completed evaluation is distinct
from attack success and from placement coverage.

## P0 — M6 stabilization and observability

- [x] Report these states independently for every episode:
  `plan_generated`, `action_applied`, `placement_verified`, `victim_completed`,
  `judge_completed`, and `attack_success`.
- [x] Add aggregate placement coverage by domain, threat model, injection MCP,
  and mutating tool.
- [x] Distinguish evaluation completion, attack reward, unsupported placement,
  and infrastructure failure in `summary.json` and process exit status.
- [x] Add forced single-action contract smokes so every advertised mutating tool
  is exercised even when a generated policy chooses no environment action.
- [x] Pin the DTAP base commit, container image digests, OpenClaw version, and
  Python dependency lock used by release smokes.
- [x] Automate the 24 Linux-supported direct/indirect cases as a scheduled or
  release-gated smoke with retained artifacts.
- [x] Update stale milestone wording in the main README and keep one placement
  support tables synchronized with code.

### P0 exit criteria

- [x] A report cannot label an episode placement-covered unless at least one
  applicable placement action was independently read back.
- [x] Each supported mutator has a positive and mismatch contract test.
- [x] The Linux matrix can be reproduced from a clean checkout without manual
  source edits or secret-bearing files.

## P1 — Placement adapter coverage

Prioritize adapters using observed policy selections and unsupported-action
counts rather than domain names alone.

- [x] Customer Service and Salesforce mutators.
- [x] Remaining Finance content mutators.
- [x] Calendar, Google Docs, and Google Forms mutators.
- [x] Hospital mutators.
- [x] Telecom and Workflow mutators.
- [x] Disable registries that have no implementation in this DTAP revision:
  Chase, Robinhood, Reddit, Google Sheets, and Google Drive.
- [x] Keep generated-plan indirect coverage at the domain-matrix level and
  exercise every advertised mutator with forced positive/mismatch contracts.
  Generated policy choice is intentionally not treated as deterministic
  per-tool coverage.
- [x] Preserve the M6 oracle boundary: read-back accepts only episode-owned
  action receipts and never exposes arbitrary backend queries.
- [x] Explicitly exclude Windows/macOS from this Linux-only scope; keep both
  fail-closed until guest-side verification is designed separately.

### P1 exit criteria

- [x] Every enabled mutating tool is explicitly `verified`, `not_applicable`, or
  documented `unsupported`; no unknown mutator silently passes.
- [x] Windows/macOS remain fail-closed until independent guest-side evidence is
  available.

## P2 — Complete OpenClaw victim trajectories

- [x] Add a task-scoped structured event sink to the OpenClaw MCP proxy.
- [x] Record victim tool name, redacted arguments, result status, timestamps,
  and stable digests without storing credentials or unrestricted payloads.
- [x] Correlate policy calls, submitted YAML, placement receipts, victim calls,
  and judge output with one episode identifier.
- [x] Make detached/headless runs retain the same minimum trajectory contract.
- [x] Show the causal chain in the trajectory viewer.
- [x] Add leakage tests for API keys, bearer tokens, MCP credentials, sandbox
  paths, and raw sensitive payloads.

### P2 exit criteria

- [x] A failed episode can be classified as policy, validation, placement,
  victim, judge, or infrastructure failure from retained artifacts alone.

## M7 — Real slime RL rollout and training dry-run

M7 is complete when a policy rollout uses the M6 MCP surface, receives a DTAP
victim/judge result, produces a training record, performs an optimizer step,
and evaluates the resulting checkpoint.

- [ ] Add bounded H>1 adaptive feedback without exposing a general environment
  oracle. For direct tasks, return a size-limited/redacted victim final response.
  For indirect tasks, return the ordered victim tool names and statuses, a
  trusted injection-exposure enum (`not_placed`, `placed_not_retrieved`,
  `retrieved`, `presented_to_model`, or `unknown`), and the bounded/redacted
  victim final response. Never expose tool arguments, raw tool results, judge
  rationale, credentials, or host paths.
- [ ] Version the feedback schema and distinguish policy-visible feedback from
  trainer-only diagnostics so experiments can state whether they use an
  adaptive attacker with victim-output access.
- [ ] Test that attempt-N feedback can repair attempt N+1 while
  `INVALID_SUBMISSION` consumes no H and only actual victim executions count
  toward H.
- [ ] Connect the M6 runtime to the production slime rollout worker.
- [ ] Define and version the rollout-record schema: task reference, policy
  prompt/response, MCP trajectory, submitted config digest, placement receipts,
  victim/judge status, reward, and failure classification.
- [ ] Store accepted training records atomically and make interrupted collection
  resumable.
- [ ] Exclude infrastructure-invalid and unsupported-placement episodes from
  reward learning; retain genuine attack misses as reward zero.
- [ ] Verify credential, receipt, port, sandbox, and artifact isolation under
  parallel rollout workers.
- [ ] Add deterministic seeds and capture all non-secret runtime metadata needed
  for reproduction.
- [ ] Build a tiny single-task overfit test that generates multiple rollouts.
- [ ] Run at least one optimizer step and save a checkpoint.
- [ ] Load the checkpoint and run a fresh DTAP evaluation.
- [ ] Confirm the full path works without fixed/template candidate plans.

### M7 exit criteria

- [ ] `rollout -> validated submission -> victim -> judge -> reward -> training
  record -> optimizer step -> checkpoint -> evaluation` passes from one command.
- [ ] Reward-zero and infrastructure-invalid samples are observably different.
- [ ] No policy-visible arbitrary placement oracle is introduced.

## M8 — Evaluation and ablation

- [ ] Compare fixed-template, untrained generated, and RL-trained policies.
- [ ] Compare placement validation disabled, M5 auxiliary, and M6 receipt-gated
  configurations.
- [ ] Report direct and indirect results separately.
- [ ] Measure attack success, evaluation completion, placement coverage, repair
  attempts, token usage, latency, and infrastructure failure rate.
- [ ] Evaluate multiple seeds and confidence intervals.
- [ ] Hold out tasks and domains to measure generalization.
- [ ] Record model/provider versions and total evaluation cost.
- [ ] Publish a machine-readable result manifest alongside the human report.

### M8 exit criteria

- [ ] Results are reproducible from pinned inputs and distinguish security
  effectiveness from harness reliability.
- [ ] Claims about RL improvement include a fixed baseline, held-out evaluation,
  and uncertainty estimates.

## Suggested PR sequence

- [x] PR A scope: M6 metrics, forced adapter coverage, dependency/runtime pinning, and
  README cleanup.
- [x] PR B scope: Linux placement adapters; all 115 enabled mutators covered.
- [x] PR C scope: complete OpenClaw structured victim trajectories and viewer support.
- [ ] PR D: M7 rollout record and tiny training dry-run.
- [ ] PR E: M8 evaluation harness and ablation report.
