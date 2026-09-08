# DTAP Agent RL next steps

This checklist starts from the M6 release merged in PR #4. At that point the
DeepSeek/OpenClaw domain matrix completed 24 Linux-supported cases. Windows and
macOS guest-platform support is now tracked separately. A completed evaluation is distinct
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

## P3 — Windows and macOS guest placement

- [x] Confirm the private DTAP repository has no hidden platform branch and use
  the Windows/macOS implementations already present on `main`.
- [x] Add policy-target-scoped guest read-back for every environment mutator
  currently referenced by Windows/macOS indirect tasks: seven Windows tools and
  macOS `inject_file`; also cover all four additional target-addressable macOS
  mutators plus typography images.
- [x] Cover Windows files, prompt files, registry values, Word paragraphs,
  Excel workbook content, PowerPoint notes, and typography artifact hashes.
- [x] Preserve the receipt boundary: paths and registry keys come only from the
  owned submitted action; responses expose only locator/status/digest.
- [x] Add positive/mismatch contracts and an audit that scans the platform
  datasets for unregistered mutators.
- [x] Run real strict placement smokes against live Windows and macOS guests.
- [x] Run direct/indirect generated-plan E2E for both platforms and retain the
  trajectories. Keep these domains opt-in until VM images and acceleration are
  available on the runner.
- [x] Evaluate and either implement or explicitly retain fail-closed status for
  platform mutators not referenced by the current indirect datasets.

### P3 exit criteria

- [x] All four Windows/macOS representative E2E cases pass with independent
  guest-side evidence and no policy-visible arbitrary oracle.
- [ ] The default matrix may include the guest platforms only after the real
  VM gate is green and reproducible.

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

## M7 — Adaptive feedback completion

M7 is complete when an H>1 policy can use bounded evidence from one genuine
failed victim execution to make a local repair on its next submission. This
milestone covers feedback observability, projection, and live repair behavior;
production slime rollout and optimizer integration are tracked separately in
M8.

- [x] Add bounded H>1 adaptive feedback without exposing a general environment
  oracle. For direct tasks, return a size-limited/redacted victim final response.
  For indirect tasks, return the ordered victim tool names and statuses, a
  per-step split between locator targeting, call completion, response inclusion,
  model presentation, and `unknown`, plus the bounded/redacted victim final
  response. Never expose tool arguments, raw tool results, judge rationale,
  credentials, or host paths.
- [x] Version the policy-visible feedback schema and keep its projection
  separate from raw/internal evidence.

### M7 adaptive-feedback follow-up

The v1 extractor currently has exact access mappings for Slack channel history,
OS/filesystem paths, and browser navigation when the submitted action contains
the URL. M6 placement coverage does not automatically provide M7 victim-access
coverage: placement proves that an injection was written, while M7 needs the
corresponding victim read tool and locator arguments.

- [ ] Add exact M7 access adapters for Finance first: map injected quote,
  analysis, and news identifiers to `browse_stock`, `browse_article`, and other
  read paths, including generated record IDs where the submitted action alone
  cannot name the eventual victim locator.
- [ ] Add exact adapters for Gmail, Legal, Travel, Research, CRM/Salesforce,
  Customer Service, Hospital, Telecom, and the remaining enabled Linux
  environments. Keep ambiguous search/name based matches as `unknown` until a
  stable correlation field exists.
- [ ] Correlate an injection receipt with a returned result item so
  `response_contains_injection` can be set without storing raw tool results.
  Cover empty results, filtering, pagination, duplicate names, and stale
  records.
- [ ] Instrument the victim message assembly boundary so `presented_to_model`
  reflects the actual provider request rather than a successful MCP call.
- [ ] Add structured skill-use events before making skill injection access
  observable. Until then skill observations remain `unknown`.
- [ ] Extend loss-aware trace completion beyond the managed OpenClaw MCP proxy
  if another victim harness becomes part of release runs.
- [ ] Wire a configured GLM/hosted Digestor and the independently switchable
  reasoning summarizer into the live runner, including token/cost/timeout
  accounting and sanitized retained outputs.
- [ ] Run H=2 live direct/indirect cases across representative domains and save
  attempt-paired artifacts. Compare `final`, `final+deterministic`, and
  `final+deterministic+digestor` with identical attempt-1 inputs.
- [ ] Add a live repair contract showing that attempt-1 feedback reaches the
  same policy session for attempt 2 while invalid submissions consume Q only.
- [ ] Test that attempt-N feedback can repair attempt N+1 while
  `INVALID_SUBMISSION` consumes no H and only actual victim executions count
  toward H.
- [ ] Introduce policy-visible feedback schema v2 while retaining v1 input
  compatibility. Represent response matching, model presentation, structured
  skill use, evidence references, and explicit unknown reasons without exposing
  raw locators, call IDs, hashes, tool results, credentials, or host paths.
- [ ] Extend the Digestor result with a typed payload-effect assessment
  (`followed`, `partially_followed`, `rejected`, `ignored`, or `unclear`) while
  keeping semantic interpretation out of the deterministic extractor.
- [ ] Audit all 115 enabled Linux environment mutators into explicit M7
  `supported` or reasoned `unsupported` states. Add positive and mismatch
  contracts for every supported victim-access adapter; Windows/macOS feedback
  adapters remain outside this milestone.

### M7 exit criteria

- [ ] Locator access, successful result inclusion, and actual model presentation
  are independently observable and never inferred from one another.
- [ ] Deterministic facts and Digestor semantic judgments remain separate, and
  feedback failure cannot change reward, H, Q, or terminal state.
- [ ] Representative direct and indirect H=2 runs demonstrate a same-session
  attempt-1 to attempt-2 repair, with A/B/C feedback modes built from identical
  attempt-1 evidence.
- [ ] Every enabled Linux mutator has an explicit M7 support classification; no
  unregistered or ambiguous mutator silently produces positive evidence.
- [ ] No policy-visible arbitrary placement oracle is introduced.

## M8 — Real slime RL rollout and training dry-run

- [ ] Add trainer-only feedback diagnostics to the rollout record so experiments
  can state whether they use an adaptive attacker with victim-output access.
- [ ] Connect the M6/M7 runtime to the production slime rollout worker.
- [ ] Define and version the rollout-record schema: task reference, policy
  prompt/response, MCP trajectory, submitted config digest, placement receipts,
  victim/judge status, reward, failure classification, and feedback mode.
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

### M8 exit criteria

- [ ] `rollout -> validated submission -> victim -> judge -> reward -> training
  record -> optimizer step -> checkpoint -> evaluation` passes from one command.
- [ ] Reward-zero and infrastructure-invalid samples are observably different.
- [ ] The saved checkpoint can be loaded for a fresh DTAP evaluation.

## M9 — Evaluation and ablation

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

### M9 exit criteria

- [ ] Results are reproducible from pinned inputs and distinguish security
  effectiveness from harness reliability.
- [ ] Claims about RL improvement include a fixed baseline, held-out evaluation,
  and uncertainty estimates.

## Suggested PR sequence

- [x] PR A scope: M6 metrics, forced adapter coverage, dependency/runtime pinning, and
  README cleanup.
- [x] PR B scope: Linux placement adapters; all 115 enabled mutators covered.
- [x] PR C scope: complete OpenClaw structured victim trajectories and viewer support.
- [ ] PR D: M7 feedback observability, schema v2, and Linux access adapters.
- [ ] PR E: M7 live H=2 repair and A/B/C feedback gate.
- [ ] PR F: M8 rollout record, optimizer step, and checkpoint evaluation.
- [ ] PR G: M9 evaluation harness and ablation report.
