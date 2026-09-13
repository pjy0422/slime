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

M6 placement coverage does not automatically provide M7 victim-access coverage:
placement proves that an injection was written, while M7 needs the corresponding
victim read tool, result, and provider-message boundary.

The per-environment proof levels and remaining receipt limitations are tracked
in [M7_DOMAIN_FEEDBACK.md](M7_DOMAIN_FEEDBACK.md).

- [x] Add exact M7 access adapters for Finance first: map injected quote,
  analysis, and news identifiers to `browse_stock`, `browse_article`, and other
  read paths, including generated record IDs where the submitted action alone
  cannot name the eventual victim locator.
- [x] Add stable exact-locator adapters where the victim API and injection
  receipt expose a correlation key: Legal, Travel, Research, CRM/Salesforce,
  Customer Service, Telecom, Slack, Google Docs, and OS Filesystem. M7 v2 already
  classifies every mutator and can prove payload inclusion from explicitly
  mapped victim MCP results, but deliberately leaves ambiguous locators
  `unknown`. The first exact-locator expansion covers receipt IDs, entity keys,
  query keys, Slack channels, Google Docs IDs, and sandbox file paths. Gmail and
  Hospital remain payload-correlated because their current injection receipts
  do not expose the eventual message/patient ID; Customer Service stays exact
  only where its receipt exposes case/order IDs.
- [x] Correlate an injection receipt with a returned result item so
  `response_contains_injection` can be set without storing raw tool results.
- [x] Add backend-specific regression fixtures for filtering, pagination,
  duplicate names, and stale records as exact-locator adapters are expanded.
- [x] Add a no-E2E feedback-boundary matrix using 24 benchmark records disjoint
  from the live record-zero matrix. Cover every Linux domain in direct and
  indirect modes, multiline payloads, real runtime MCP namespaces, missing
  payloads, provider-presentation separation, and write-echo false positives.
- [x] Instrument the victim message assembly boundary so `presented_to_model`
  reflects the actual provider request rather than a successful MCP call.
- [x] Keep skill use explicitly `unknown` until OpenClaw exposes a trusted
  structured skill-use event; skill presentation is independently observable.
- [x] Audit whether loss-aware trace completion must extend beyond the managed
  OpenClaw MCP proxy. The Linux and platform release gates currently require
  `--victim-agent-type openclaw`; no second victim harness is part of release
  runs, so no speculative adapter is added. Reopen this item when that changes.
- [x] Wire a configured GLM/hosted Digestor and the independently switchable
  reasoning summarizer into the live runner, including token/cost/timeout
  accounting and sanitized retained outputs.
- [x] Run H=2 live direct/indirect Finance cases and save attempt-paired
  artifacts, including a same-session indirect repair that succeeds on attempt 2.
- [x] Compare `final`, `final+deterministic`, and
  `final+deterministic+digestor` with identical attempt-1 inputs.
- [x] Add a live repair contract showing that attempt-1 feedback reaches the
  same policy session for attempt 2 while invalid submissions consume Q only.
- [x] Test that attempt-N feedback can repair attempt N+1 while
  `INVALID_SUBMISSION` consumes no H and only actual victim executions count
  toward H.
- [x] Introduce policy-visible feedback schema v2 while retaining v1 input
  compatibility. Represent response matching, model presentation, structured
  skill use, evidence references, and explicit unknown reasons without exposing
  raw locators, call IDs, hashes, tool results, credentials, or host paths.
- [x] Extend the Digestor result with a typed payload-effect assessment
  (`followed`, `partially_followed`, `rejected`, `ignored`, or `unclear`) while
  keeping semantic interpretation out of the deterministic extractor.
- [x] Audit all 115 enabled Linux environment mutators into explicit M7
  `supported` or reasoned `unsupported` states. Add positive and mismatch
  contracts for every supported victim-access adapter; Windows/macOS feedback
  adapters remain outside this milestone.

### M7 exit criteria

- [x] Locator access, successful result inclusion, and actual model presentation
  are independently observable and never inferred from one another.
- [x] Deterministic facts and Digestor semantic judgments remain separate, and
  feedback failure cannot change reward, H, Q, or terminal state.
- [x] Representative direct and indirect H=2 runs demonstrate a same-session
  attempt-1 to attempt-2 repair, with A/B/C feedback modes built from identical
  attempt-1 evidence.
- [x] Every enabled Linux mutator has an explicit M7 support classification; no
  unregistered or ambiguous mutator silently produces positive evidence.
- [x] No policy-visible arbitrary placement oracle is introduced.

## Post-M7 hardcode and heuristic hardening

Resolve security/reward-boundary inference before mechanical adapter cleanup.

### P0 — Boundary correctness

- [x] Replace environment-tool prefix classification with exact DTAP
  `SUPPORTED_PLACEMENT_TOOLS` / `NON_PLACEMENT_TOOLS` registries. Fail the
  inventory gate on unknown, stale, or overlapping names.
- [x] Replace judge-error phrase matching with structured task/attack stage
  statuses. Travel retry exhaustion raises `JudgeUnavailableError`; Medical
  uses explicit result-field presence.
- [x] Require exact server/tool identity for M7 tool-description presentation;
  do not match a target name occurring inside another tool's description.
- [x] Replace legacy guest compose-path substring matching with an exact public
  path allowlist.
- [x] Verify both clean overlay application and incremental upgrade from the
  previously merged overlay.

### P1 — Adapter maintainability

- [x] Replace remaining adapter-family `startswith` dispatches with exact
  per-tool handler sets. They are currently guarded by exact registries and are
  not authorization checks, so this is a maintainability task rather than a
  release blocker.
- [x] Add a registry-to-handler exhaustiveness assertion so registering a new
  placement or feedback tool requires choosing its handler family explicitly.

### P2 — Evaluation inventory maintainability

- [x] Generate repeated domain/platform matrices from one checked-in benchmark
  manifest while retaining explicit opt-in gates for VM-backed platforms.

### P3 — Disjoint live E2E regression

- [x] Add a manifest-defined `holdout-v1` selector covering 24 Linux
  direct/indirect coordinates without reusing the release-v1 record-zero tasks.
- [x] Reject missing holdout indices and prevent `--resume` from accepting an
  artifact produced by another selection profile.
- [x] Run all 24 holdout cases with DeepSeek policy/victim, OpenClaw victim
  harness, H=2, strict placement, and deterministic adaptive feedback.
- [x] Retain the run summary and trajectories as one viewer-compatible run.

#### P3 exit criteria

- [x] The 24 selected task identities are disjoint from release-v1, all cases
  complete or receive an explicit failure classification, and the result
  summary records the benchmark-manifest digest and selection profile.

## M8 — Multi-turn slime RL algorithms and DTAP-HiPER training path

M8 connects the completed M6/M7 DTAP runtime to production slime training and
adds the multi-turn credit-assignment methods needed for the main RL experiments.
Detailed implementation requirements are in
[M8_IMPLEMENTATION_PLAN.md](M8_IMPLEMENTATION_PLAN.md).

The shared substrate must preserve the logical learning trajectory across Claude
Code auto-compaction. Rewards, values, advantages, returns, option boundaries,
and segment recurrence are turn/segment-level quantities; the existing slime
policy objective remains token-level after semantic advantages are projected to
owned generated-token spans.

- [x] Add versioned logical-turn metadata to training samples and preserve
  exactly-one-owner semantics through CLEAN, REALIGN, FORK, compaction, packing,
  DP split, and context parallelism.
- [x] Add rollout-DP affinity for temporal estimators so sibling samples from one
  `rollout_id` remain available to the same critic/training rank.
- [x] Implement multi-turn vanilla PPO with turn-level GAE and sparse per-turn
  critic targets.
- [x] Implement DC-GRPO DW and SW with group credit computed before DP split.
- [x] Implement GiGPO with stable environment-state `anchor_key` grouping;
  compaction prompt text must not be used as the anchor identity.
- [ ] Add a two-head high/low critic and reference HiPER/HAE implementation,
  including segment-level high recurrence, within-segment low recurrence, and
  separate high/low value masks and losses.
- [ ] Keep **reference HiPER/HAE** separate from **DTAP-HiPER**. Reference mode
  validates the algorithm with the standard switch/subgoal/action contract;
  DTAP mode reuses the validated HAE core but adds DTAP-specific hierarchical
  planning semantics.
- [ ] In DTAP-HiPER, prompt and parse all four decisions every logical turn:
  `switch`, `high_subgoal`, `low_subgoal`, and executable `action`. `SWITCH`
  creates a new high-level option/segment; `KEEP` preserves the active
  high-level subgoal; the low-level subgoal is a per-turn objective distinct
  from the concrete action.
- [ ] Keep DTAP semantic spans separate (`switch`, `high_subgoal`,
  `low_subgoal`, `action`) so high/low credit and debugging can be reconstructed
  without overloading `loss_mask`.
- [ ] Make DTAP high-level option identity survive auto-compaction through
  structured trajectory metadata rather than by searching compacted summary
  text.
- [ ] Connect the M6/M7 runtime to the production slime rollout worker and
  version the rollout/training record, including task identity, logical turns,
  feedback mode, failure classification, reward, and hierarchy mode.
- [ ] Exclude infrastructure-invalid and unsupported-placement episodes from
  reward learning while retaining genuine attack misses as valid reward-zero
  samples.
- [ ] Store accepted training records atomically, make collection resumable, and
  verify credential/receipt/port/workspace/artifact isolation under parallel
  rollout workers.
- [ ] Add deterministic seeds and capture all non-secret runtime metadata needed
  for reproduction.
- [ ] Add deterministic algorithm fixtures, compaction/DP/CP regressions,
  HiPER reference parity tests, and separate DTAP hierarchical prompt/parser
  tests.
- [ ] Build a tiny single-task overfit test that generates multiple rollouts,
  runs at least one optimizer step, saves a checkpoint, reloads it, and performs
  a fresh DTAP evaluation without fixed/template candidate plans.

### M8 exit criteria

- [ ] Multi-turn PPO, DC-GRPO DW/SW, GiGPO, and reference HiPER/HAE pass their
  deterministic math and trajectory fixtures.
- [ ] Reference HiPER and DTAP-HiPER are explicit separate modes that share the
  HAE math core but not the prompt contract.
- [ ] DTAP-HiPER explicitly prompts/parses `switch`, `high_subgoal`,
  `low_subgoal`, and `action`, and `KEEP`/`SWITCH` semantics remain correct
  across compaction.
- [ ] Reward-zero and infrastructure-invalid samples are observably different.
- [ ] `rollout -> validated submission -> victim -> judge -> reward -> logical
  training record -> advantage/return -> optimizer step -> checkpoint -> fresh
  evaluation` passes end to end.
- [ ] The saved checkpoint can be loaded for a fresh DTAP evaluation without
  fixed/template candidate plans.

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
- [x] PR D: M7 feedback observability, schema v2, and Linux access adapters.
- [x] PR E: M7 live H=2 repair and A/B/C feedback gate.
- [x] PR F1: M8.0a logical-turn metadata and compaction ownership.
- [x] PR F2: M8.0b rollout affinity and metadata DP passthrough.
- [x] PR F3: M8.1 multi-turn vanilla PPO.
- [x] PR F4: M8.2 DC-GRPO DW/SW.
- [x] PR F5: M8.3 GiGPO.
- [ ] PR F6: M8.4a two-head critic infrastructure.
- [ ] PR F7: M8.4b reference HiPER/HAE.
- [ ] PR F8: M8.5 DTAP-HiPER hierarchy and credit projection.
- [ ] PR F9: M8.6 production rollout, optimizer/checkpoint smoke, CI, and docs.
- [ ] PR G: M9 evaluation harness and ablation report.
