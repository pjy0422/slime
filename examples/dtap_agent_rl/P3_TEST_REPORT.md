# P3 disjoint live E2E report

Date: 2026-09-09–10 (Asia/Seoul)

P3 exercised a task-disjoint 24-case Linux matrix: 12 domains × direct and
indirect, using benchmark record 50 instead of the release-v1 record-zero
selection. The checked-in tests resolve both profiles against the DTAP datasets
and assert that their `(risk_category, task_id)` identities do not overlap.

## Configuration

- selection profile: `holdout-v1` (`benchmark_index=50`)
- policy and victim model: `deepseek-v4-flash`
- victim harness: OpenClaw
- victim-execution horizon: H=2
- feedback: `final+deterministic`
- platform scope: Linux; macOS and Windows excluded
- final manifest digest:
  `81c941bc22709351d175c48b7a87c0b44f3e3fe9d02aee3d3ad77cb406ac7fe6`

## Result

- 24/24 cases completed with no final infrastructure, judge, placement,
  policy, validation, or victim failure.
- 8/24 attacks succeeded (33.3% final ASR).
- Placement validation was applicable to 16 cases and covered all 16.
- 55 victim-visible placements were verified.

The first parallel wave completed 21/24 cases. Code/direct, Medical/direct, and
Workflow/indirect encountered transient provider-rate-limit infrastructure
errors. The bounded resume gate retried only those incomplete coordinates at
lower concurrency and produced the final 24/24 result. The first-wave snapshot
is retained separately so transient failures are not hidden.

## Reproduction

Activate the DTAP environment, export the provider credentials for both policy
and victim, then run:

```bash
python -m examples.dtap_agent_rl.scripts.run_p3_holdout_gate \
  --dtap-root /path/to/DecodingTrust-Agent \
  --artifacts-root /path/to/p3-holdout-artifacts
```

The runner applies the managed DTAP overlay, starts at concurrency 8, and on a
nonzero result performs at most two resume passes at concurrency 4 and 2.
Existing passed coordinates are retained only when their task identity,
selection profile, and benchmark index match.

Viewer-compatible evidence is stored in
`artifacts/p3-holdout-live-20260909/`: `summary.json` is the final aggregate,
`summary-wave1.json` preserves the initial wave, and each domain/threat-model
directory contains the submitted config, trajectories, placements, and judge
artifacts for its selected task.
