# M8 implementation plan — multi-turn RL, HiPER/HAE, and DTAP hierarchy

## 1. Goal

M8 moves DTAP Agent RL from the M6/M7 evaluation runtime into a production slime training path. The milestone is not complete when a rollout merely produces a reward. It is complete when the same logical multi-turn trajectory survives Claude Code auto-compaction, is converted into correct training data, receives algorithm-specific turn/segment credit, takes an optimizer step, saves a checkpoint, and can be loaded for a fresh DTAP evaluation.

M8 implements four training algorithms on top of one shared multi-turn substrate:

1. multi-turn vanilla PPO;
2. DC-GRPO, both DW and SW variants;
3. GiGPO;
4. HiPER/HAE with a two-head critic.

HiPER has two deliberately separate implementation tracks:

- **Reference HiPER/HAE**: implement and validate the algorithm independently of DTAP-specific prompting. This is the parity/debugging baseline.
- **DTAP-HiPER**: reuse the HAE core, but make the policy hierarchy reflect DTAP planning explicitly. Every turn prompts for a switch decision, a high-level subgoal, a low-level subgoal, and an executable action.

Do not collapse these tracks into one prompt or one test fixture. A failure must be classifiable as an HAE math/parity failure or as a DTAP hierarchy/prompt contract failure.

## 2. Non-negotiable semantic rules

### 2.1 RL time is turn time

The temporal unit for rewards, values, advantages, returns, switching, and HAE segments is a logical agent turn, not an individual token.

For turn `t`:

- `s_t`: logical state/context used to produce the turn;
- `q_t`: switch decision;
- `o_t`: active high-level option/subgoal;
- `l_t`: DTAP low-level subgoal for the current turn, when the DTAP hierarchy is enabled;
- `a_t`: executable low-level environment action;
- `r_t`: turn reward.

Token-level PPO remains the optimizer surface. Turn/segment advantages are projected onto the tokens that represent the corresponding semantic decision.

### 2.2 Do not overload `loss_mask`

`Sample.loss_mask` keeps its existing slime meaning: which response tokens are owned by this sample and are eligible for training after prompt drift, realignment, compaction, or branching.

Algorithm-specific semantics are separate metadata. Replayed or realigned context must never regain training ownership merely because it belongs to a logical historical turn.

The final actor weight is conceptually:

```text
base loss_mask × projected algorithm advantage
```

Do not encode high/low/switch/action roles by mutating the base ownership mask.

### 2.3 Compaction changes context representation, not the learning trajectory

Claude Code auto-compaction is mandatory and supported. Prompt text may be summarized or rewritten, but the learning trajectory is still the ordered set of logical turns identified by `(rollout_id, turn_idx)`.

Every logical turn has exactly one training owner across all sibling `Sample`s with the same `rollout_id`.

### 2.4 Keep credit-assignment granularity separate from PPO ratio granularity

M8 computes credit at turn or segment granularity and continues to use slime's existing token-level PPO/GRPO importance ratio and clipping.

M8 does **not** introduce a span-level aggregate log-probability ratio or span-level PPO clipping objective. That is a separate algorithmic choice and is out of scope for this milestone.

## 3. Canonical multi-turn metadata contract

Use `Sample.train_metadata` for training-only annotations. Add one versioned namespace and keep response spans response-relative.

```python
sample.train_metadata = {
    "multi_turn": {
        "version": 1,
        "context_revision": 0,
        "turns": [
            {
                "turn_idx": 0,
                "response_span": [start, end],
                "reward": 0.0,
                "done": False,
                "truncated": False,
                "anchor_key": None,
                "switch": None,
                "role_spans": {
                    "switch": [],
                    "subgoal": [],
                    "high_subgoal": [],
                    "low_subgoal": [],
                    "action": [],
                },
                "value_positions": {
                    "high": None,
                    "low": None,
                },
                "format_valid": True,
            }
        ],
    }
}
```

Rules:

- only turns owned by this `Sample` appear in its `turns` list;
- the complete rollout is the union of sibling sample metadata with the same `rollout_id`, sorted by `turn_idx`;
- `(rollout_id, turn_idx)` must have exactly one owner;
- all spans are response-relative half-open intervals `[start, end)`;
- semantic spans must lie inside the owned `response_span`;
- replayed historical text has no new turn owner;
- metadata stays in full response coordinates even when context parallelism slices token tensors.

### 3.1 Reward resolution

For one logical rollout:

```text
all turns have explicit reward -> use those rewards
no turn has explicit reward   -> [0, ..., 0, sample/trajectory terminal reward]
partial explicit rewards      -> ValueError
```

Partial reward annotation is too ambiguous to guess.
If the terminal fallback reward appears on more than one sibling sample, every
non-null value must agree or validation fails. Apply that fallback exactly once
to the last complete logical turn, never once per sibling sample.

### 3.2 Truncated turns

If collection truncates part-way through a logical turn:

- do not train the partial turn with a full turn-level advantage;
- remove that turn from logical training ownership;
- zero ownership for the partial generated tokens;
- mark the trajectory truncated at the previous complete turn;
- retain enough diagnostics to explain the truncation.

## 4. M8.0 — multi-turn substrate and compaction correctness

Extend the existing trajectory builder rather than replacing it.

Required behavior:

- assign a monotonic `turn_idx` to every logical agent turn;
- record the generated response span owned by that turn;
- preserve ownership through CLEAN prompt extension;
- preserve the existing REALIGN rule: rewritten historical response becomes context and is masked out;
- preserve FORK behavior and keep sibling samples on the same `rollout_id`;
- convert builder-local offsets to response-relative offsets in `to_sample()`;
- never infer learning history back from compacted prompt text.

Target areas:

- `slime/agent/trajectory.py`;
- `slime/utils/types.py` only when a direct field/helper is materially useful;
- agent/adapters should remain protocol/harness generic.

Do not put DTAP-specific reward logic into `BaseAdapter`.

### 4.1 Existing baseline to extend

At the adoption of this plan, `Sample.train_metadata`, `Sample.rollout_id`,
`Sample.group_index`, and trajectory-node `turn_index` already exist. Reuse
those fields rather than introducing parallel identities. The trajectory
builder does not yet emit logical-turn training metadata. Rollout conversion
copies `train_metadata` to the train-data `metadata` key, but the DP packaging
whitelist does not yet forward that key. The current DP scheduler keeps sibling
samples in one training step, but may distribute their microbatches across DP
ranks.

M8.0 therefore extends these existing boundaries: it emits and validates the
versioned namespace, forwards `metadata`, carries `Sample.group_index` as a
deterministic `group_indices` train-data vector, and adds optional same-rank
affinity without replacing the current default scheduler.

### 4.2 Training-data plumbing

Ensure `Sample.train_metadata` survives all DP splitting and minibatch scheduling paths.

Required fixes:

- pass `metadata` through the DP train-data split whitelist;
- add deterministic logical `group_indices` where group-based estimators need them;
- validate malformed multi-turn metadata before training begins;
- ensure metadata is not silently dropped under packing or CP.

### 4.3 Rollout affinity

Add:

```text
--rollout-dp-affinity
```

When enabled:

1. bundle all samples with the same `rollout_id`;
2. assign the entire bundle to one DP rank;
3. pack minibatches within that rank;
4. keep per-rank minibatch counts compatible with the existing scheduler;
5. fail fast if the requested batch constraints cannot be satisfied without breaking affinity.

Required initially for multi-turn PPO and HAE.

### 4.4 M8.0 tests

Add CPU tests for CLEAN ownership, REALIGN masking, FORK sibling ownership, repeated compaction, exactly-one-owner, response-relative span conversion, partial-turn truncation, metadata DP passthrough, rollout-affinity scheduling, and impossible affinity packing.

M8.0 must land without changing current single-turn policy-loss numerics.

## 5. M8.1 — multi-turn vanilla PPO

Add:

```text
--advantage-estimator multi_turn_ppo
```

Requirements:

- critic required;
- one critic value head;
- `--rollout-dp-affinity` required;
- reuse existing `--gamma`, `--lambd`, and `--normalize-advantages`.

Compute turn GAE:

```text
delta_t = r_t + gamma * (1 - done_t) * V_{t+1} - V_t
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
R_t     = A_t + V_t
```

Use one critic scalar per logical turn. Normalize turn scalars before broadcasting them to tokens. Do not run generic token-count-weighted whitening again afterward.

Project `A_t` across the trainable tokens owned by turn `t`, then call the existing slime token-level policy loss unchanged.

Tests must cover exact GAE, terminal bootstrap, `T=1`, variable token lengths, compaction parity, and CP parity.

## 6. M8.2 — DC-GRPO DW/SW

Add:

```text
--advantage-estimator dcgrpo
--dcgrpo-mode {dw,sw}
--dcgrpo-alpha <float>
```

Perform DC-GRPO credit assignment on the complete rollout/group collection **before DP split**.

Definitions:

- trajectory identity: `rollout_id`;
- comparison group: `(group_index, turn_idx)`;
- suffix return: `R_t = r_t + gamma * R_{t+1}`.

DW computes normalized discounted suffix-return credit. SW combines normalized immediate-reward and future components using `alpha`.

For singleton groups or zero-variance components, define the normalized component as zero.

Attach the final scalar as precomputed turn credit; token projection remains inside the trainer.

Tests: hand exact values, final-turn equivalence, `T=1` GRPO parity, singleton, zero variance, ragged trajectories, permutation invariance, and compaction parity.

## 7. M8.3 — GiGPO

Add:

```text
--advantage-estimator gigpo
--gigpo-step-advantage-weight <float>
--gigpo-normalization {mean,mean_std}
```

Compute group credit before DP split.

Definitions:

- episode group: `group_index`;
- trajectory: `rollout_id` ordered by `turn_idx`;
- step group: `(group_index, anchor_key)`;
- `anchor_key`: stable fingerprint of environment observation/state, not the whole serialized conversation.

The anchor must be invariant to Claude Code compaction. `anchor_key=None` is invalid. M8 uses exact anchor equality only; do not add fuzzy matching initially.

Tests: upstream-derived fixture, episode component, step-relative component, weight zero, normalization modes, repeated anchors, missing-anchor validation, and compaction invariance.

## 8. M8.4 — reference HiPER/HAE

This track exists to make the algorithm independently testable before DTAP prompt semantics are introduced.

### 8.1 Reference prompt contract

Use the Plan-Execute form:

```text
<switch>KEEP|SWITCH</switch>
<subgoal>...</subgoal>
<action>...</action>
```

Semantics:

- a switch decision is emitted each turn;
- on `SWITCH`, a new high-level subgoal starts a new HAE segment;
- on `KEEP`, the active high-level subgoal persists;
- the executable action is the low-level decision for the turn.

Parse roles only inside the turn's owned response span.

### 8.2 Two-head critic

Add:

```text
--critic-value-heads {1,2}
--critic-high-value-loss-coef <float>
```

Default remains one head for existing behavior. For two heads:

- low head: `V_low(s_t, o_t)`;
- high head: `V_high(s_t)` at option boundaries.

Do not build a generic named N-head framework in M8. A one-head checkpoint may reuse the backbone, but the incompatible output head must be reinitialized explicitly.

### 8.3 Value positions

Reference default:

- high value: first response token on a segment-boundary turn;
- low value: first token after `</subgoal>`;
- fallback low: first token after `</switch>`;
- final fallback: first response token.

### 8.4 HAE computation

Segments start at turn 0 and each `SWITCH` turn.

Low HAE:

- GAE is computed within a segment;
- the final turn of a nonterminal segment bootstraps from the next boundary's high value.

High HAE:

- aggregate discounted rewards inside each segment;
- use SMDP discount `gamma ** duration`;
- compute high-level GAE over segments with `--hae-high-lambd`.

Normalize low and high scalar advantages separately before token projection.

Reference actor projection:

- low advantage -> `<action>` tokens;
- high advantage -> boundary `<subgoal>` tokens;
- switch remains a distinct span and is not silently given a fabricated switch advantage.

The requested M8 scope uses two value heads only. A third termination critic is out of scope.

### 8.5 Two-head value loss

Use separate sparse masks and targets for low and high heads. Reduce each head by its own number of active value positions, then combine:

```text
value_loss = low_value_loss + high_value_coef * high_value_loss
```

Do not use actor token count as the critic denominator.

Tests must independently cover segment construction, segment-end bootstrap, high SMDP recurrence, normalization, sparse two-head loss, single-head compatibility, and a checked-in reference fixture.

## 9. M8.5 — DTAP-HiPER hierarchy

DTAP-HiPER is not merely reference HiPER loss attached to the existing DTAP prompt. The hierarchy must be explicit in the policy decisions.

### 9.1 DTAP hierarchical prompt contract

Every agent turn must prompt and parse:

```text
<switch>KEEP|SWITCH</switch>
<high_subgoal>...</high_subgoal>
<low_subgoal>...</low_subgoal>
<action>...</action>
```

Semantics:

1. `switch`
   - `SWITCH`: terminate the previous high-level option and choose a new `high_subgoal`;
   - `KEEP`: continue the active high-level option.

2. `high_subgoal`
   - durable DTAP attack/planning objective spanning one or more turns;
   - on `SWITCH`, it is a newly selected option and starts a new HAE segment;
   - on `KEEP`, the emitted value must represent the same active option. Detect drift instead of treating it as an implicit switch.

The runtime assigns a stable internal option identity when `SWITCH` is
accepted. On `KEEP`, compare the emitted `high_subgoal` with the stored value
after normalizing only line endings and surrounding whitespace. Any other
change is an explicit format/drift failure; it must not silently create a new
option or use an LLM/fuzzy comparison.

3. `low_subgoal`
   - generated every turn;
   - immediate local objective conditioned on the active high-level objective;
   - distinct from the executable action.

4. `action`
   - concrete DTAP environment/tool action;
   - validated by the existing submission/placement boundary.

Canonical flow:

```text
state / observation
      |
      v
[ switch: KEEP / SWITCH ]
      |
      +-- SWITCH --> choose new high_subgoal
      |
      +-- KEEP ----> preserve active high_subgoal
      |
      v
[ low_subgoal conditioned on active high_subgoal ]
      |
      v
[ executable action conditioned on high + low subgoal ]
      |
      v
DTAP environment -> victim -> judge -> reward / feedback
```

### 9.2 Prompting requirements

The DTAP hierarchical system/user prompt must explain the different time horizons. It must state that:

- the high-level subgoal stays stable across `KEEP` turns;
- `SWITCH` is used only when the current strategy is complete, invalidated, or should be replaced;
- the low-level subgoal should be achievable in the current turn and advance the active high-level goal;
- the action must be executable under the current DTAP schema/tool contract;
- M7 feedback is next-turn evidence, not a hidden reward or general oracle.

Keep this prompt separate from the reference HiPER prompt. Add an explicit selector:

```text
--hae-policy-mode {reference,dtap}
```

Do not infer the mode from dataset names or tag presence.

### 9.3 DTAP semantic spans and credit

Record four distinct spans:

```text
switch
high_subgoal
low_subgoal
action
```

Canonical DTAP-HAE projection:

- high advantage -> newly selected boundary `high_subgoal` tokens;
- low advantage -> `low_subgoal` + `action` tokens for the turn;
- carry-forward high-subgoal text on `KEEP` receives no new high-option advantage;
- switch remains a distinct decision span.

If M8 experiments need trainable switch tokens, add one explicit DTAP adaptation that assigns high-level/boundary credit to the switch span together with the boundary high-subgoal span. Report this as a **DTAP adaptation**, not exact paper switch-advantage parity.

A later exact switch-advantage implementation may use `beta_t = pi(q_t=SWITCH | ...)` once switch-probability plumbing is implemented and tested.

### 9.4 Hierarchy state under compaction

Structured metadata must retain:

- current/previous high-level option identity;
- switch boundary turns;
- low-level subgoal per turn;
- action span;
- feedback mode/evidence reference;
- logical turn index and context revision.

Do not recover active options by searching compaction-summary text.

### 9.5 DTAP-HiPER tests

Keep these separate from HAE math tests:

- parser extracts all four fields;
- malformed/missing/duplicate tags fail closed;
- `KEEP` high-subgoal drift is detected;
- `SWITCH` creates exactly one new segment boundary;
- low subgoal and action are distinct;
- low credit covers low-subgoal + action only;
- high credit covers boundary high-subgoal only in canonical mode;
- optional switch-credit adaptation is explicit;
- compaction preserves active high-level option identity;
- M7 feedback can affect the next turn without rewriting prior ownership;
- reference and DTAP modes use separate prompt contracts but the same validated HAE core.

## 10. M8.6 — runtime integration and optimizer dry-run

Connect M6/M7 DTAP execution to the production slime rollout worker.

Version the rollout record with at least:

- task reference;
- rollout/group identity;
- logical turn metadata;
- policy prompt/response references;
- MCP trajectory summary;
- submitted config digest;
- placement receipt summary;
- victim/judge status;
- reward and per-turn reward when available;
- failure classification;
- adaptive feedback mode;
- HiPER policy mode/hierarchy fields when enabled;
- deterministic seeds and non-secret reproduction metadata.

Trainer diagnostics must state whether an adaptive attacker had victim-output access.

Accepted records must be written atomically and collection must be resumable.

Eligibility:

- exclude infrastructure-invalid episodes;
- exclude unsupported-placement episodes;
- keep genuine attack misses as valid reward-zero samples;
- keep reward-zero and infrastructure-invalid states observably distinct.

Verify credential, receipt, port, sandbox, workspace, and artifact isolation under parallel workers.

The milestone-wide E2E path is:

```text
rollout
 -> validated submission
 -> victim
 -> judge
 -> reward
 -> multi-turn training record
 -> advantage/return computation
 -> token projection
 -> optimizer step
 -> checkpoint
 -> fresh DTAP evaluation
```

Build a tiny single-task overfit case with multiple rollouts. Save, reload, and evaluate the checkpoint without fixed/template candidate plans.

## 11. Suggested code layout

Keep implementation direct and slime-native; do not add a plugin registry for four built-ins.

```text
slime/utils/advantages/
    __init__.py
    multi_turn.py
    mt_ppo.py
    dcgrpo.py
    gigpo.py
    hae.py
```

Likely modified areas:

```text
slime/agent/trajectory.py
slime/utils/types.py
slime/ray/rollout.py
slime/utils/dp_schedule.py
slime/utils/arguments.py
slime/backends/megatron_utils/model_provider.py
slime/backends/megatron_utils/loss.py
slime/backends/megatron_utils/actor.py
slime/backends/megatron_utils/data.py
examples/dtap_agent_rl/... hierarchy prompt/parser/runtime integration
```

## 12. CLI surface

```text
--advantage-estimator multi_turn_ppo
--advantage-estimator dcgrpo
--dcgrpo-mode {dw,sw}
--dcgrpo-alpha FLOAT
--advantage-estimator gigpo
--gigpo-step-advantage-weight FLOAT
--gigpo-normalization {mean,mean_std}
--advantage-estimator hae
--hae-high-lambd FLOAT
--hae-policy-mode {reference,dtap}
--critic-value-heads {1,2}
--critic-high-value-loss-coef FLOAT
--rollout-dp-affinity
```

Reuse existing gamma, lambda, advantage normalization, PPO clipping, and KL flags.

Validate combinations explicitly. Multi-turn PPO requires a single-head critic plus affinity. HAE requires a two-head critic plus affinity. DC-GRPO mode must be DW or SW. GiGPO step weight must be nonnegative.

## 13. Context parallelism

Semantic metadata always uses full response coordinates.

When critic tensors are CP-local:

1. gather/reconstruct values in full response order;
2. extract sparse turn/segment values;
3. compute logical advantages/returns;
4. project them onto the full response vector;
5. apply the existing CP slicing convention.

Add CP=1 versus CP>1 logical parity tests.

## 14. Observability and debugging

A retained debug record must reconstruct:

```text
raw turn reward
 -> logical turn/segment credit
 -> normalized scalar advantage
 -> semantic role span
 -> final per-token advantage
 -> policy/value loss contribution
```

Common metrics: turns/rollout, samples/rollout, trainable tokens, replayed tokens masked, context revisions, truncated turns.

Algorithm metrics:

- multi-turn PPO: turn reward/value/advantage/return;
- DC-GRPO: immediate/future/suffix credit;
- GiGPO: anchor group size, singleton rate, episode/step credit;
- HAE: segment count/length, low/high advantages, switch count, boundary ratio, low/high value loss;
- DTAP-HiPER: KEEP/SWITCH rate, high-subgoal duration, low-subgoal count, drift violations, hierarchy format failures.

## 15. Test and CI requirements

Add focused CPU tests using existing conventions and register them in the PR CI template. Regenerate the generated workflow after template changes.

Reference fixtures must be checked into this repository; CI must not clone or execute external HiPER/GiGPO repositories.
Each fixture must record the paper/version and, when derived from reference
code, the exact repository commit and applicable license. Expected values must
remain reviewable without network access.

Required classes: pure math, metadata/schema, compaction ownership, DP affinity, CP parity, critic-head compatibility, reference HiPER prompt/parser, DTAP hierarchy prompt/parser, rollout-record eligibility, and tiny optimizer smoke.

## 16. PR sequence

1. **M8.0a** — logical turn metadata and compaction ownership;
2. **M8.0b** — rollout-affinity scheduler and metadata DP passthrough;
3. **M8.1** — multi-turn vanilla PPO;
4. **M8.2** — DC-GRPO DW/SW;
5. **M8.3** — GiGPO;
6. **M8.4a** — two-head critic infrastructure;
7. **M8.4b** — reference HiPER/HAE;
8. **M8.5** — DTAP-HiPER hierarchical prompt/state/credit projection;
9. **M8.6** — production rollout integration, optimizer/checkpoint smoke, observability, CI, docs.

Do not mix DTAP hierarchy, two-head critic, compaction plumbing, and HAE math into one patch.

## 17. Non-goals

1. span-level/decision-level PPO clipping;
2. a generic RL plugin registry;
3. arbitrary N-head critics;
4. a third termination critic;
5. GiGPO fuzzy grouping;
6. distributed all-gather replacement for rollout affinity;
7. semantic overloading of `loss_mask`;
8. recovering logical options from compaction-summary text;
9. claiming exact paper switch-advantage parity without the required switch-probability implementation.

## 18. Exit criteria

M8 is complete only when:

- compaction preserves learning ownership and logical turn identity;
- multi-turn PPO takes a correct turn-level GAE step;
- DC-GRPO DW/SW pass deterministic fixtures;
- GiGPO passes deterministic episode/step grouping fixtures;
- two-head critic and HAE pass segment/bootstrap/SMDP tests;
- reference HiPER and DTAP-HiPER are separate selectable modes;
- DTAP-HiPER explicitly prompts/parses switch, high-level subgoal, low-level subgoal, and action;
- DTAP KEEP/SWITCH semantics survive compaction without prompt-text reconstruction;
- infrastructure-invalid samples are excluded while genuine reward-zero misses remain trainable;
- an end-to-end DTAP rollout reaches an optimizer step and checkpoint;
- the checkpoint loads and runs a fresh DTAP evaluation;
- no fixed/template candidate plan is required.

> Credit assignment is logical-turn/segment level; slime optimizes selected generated tokens with its existing token-level policy loss; compaction may rewrite context, but it must never rewrite the learning trajectory.
