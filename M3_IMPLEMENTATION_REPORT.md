# M3 implementation report

## Outcome

M0, M1, M2, and M3 are merged into one overlay. M3 now supports a single persistent
policy context containing at most `H` authoritative attack submissions. Each
accepted submission is rendered into an ephemeral task copy and executed through a
fresh subprocess containing one fresh DTAP `TaskExecutor`.

## Implemented production files

- `episode_runtime.py`: H budget, success/exhaustion/infra terminal states, rewards
- `candidate_config.py`: strict trusted renderer, safe YAML round trip, DTAP parse,
  semantic comparison, immutable source copy, symlink/special-file rejection
- `attempt_runner.py`: fresh process, global concurrency gate, judge-result parsing,
  timeout/cancel process-group cleanup, no exit-code reward inference
- `scripts/run_dtap_attempt.py`: one `ScheduledTask`, one `TaskExecutor`, one victim
  and judge run, guaranteed executor shutdown
- `submission.py`: strict `SubmissionPlan`, per-episode async lock, authoritative M2
  revalidation, YAML gate, H accounting, safe one-bit receipts, token registry
- `mcp_server.py`: optional fourth `submit_attack(plan)` tool and paired registry
  routing while preserving M1/M2 APIs
- `m3.py`: one harness lifetime and paired registry lifetime per policy episode
- `trajectory.py`: sync `open_session`, async `finish_session`, `drop_session`, and
  infra exclusion compatible with slime's current adapter contract
- `generate.py`: configurable slime custom-generate entrypoint
- `scripts/smoke_m3_hloop_api.py`: Claude + fake fail/succeed evaluator smoke
- `scripts/smoke_m3_dtap.py`: real one-submit victim/judge smoke

## Security and correctness properties

- A policy cannot submit raw YAML or override trusted task/agent/judge metadata.
- `submit_attack` revalidates the complete plan even after advisory validation.
- Unknown envelope and step fields fail closed.
- Invalid plans and candidate configs never call the runner or consume H.
- H increments only for a runner result carrying `evaluation_started=True`.
- Judge `attack_success` is authoritative; process exit and `task_success` are not.
- Judge details, victim output, trajectories, paths, and exceptions stay trusted-side.
- Original task files remain unchanged; every attempt gets a unique copied tree.
- Timeout and cancellation kill the full subprocess group.
- Infrastructure errors terminate with no trusted reward and mark samples for removal.
- Policy context is not reset between failed macro submissions.

## Verification

```text
73 passed
2 skipped
```

Skipped:

1. M2 compatibility against the actual installed DTAP helper implementation.
2. M3 real two-attempt Docker reset sentinel.

These are the only remaining release gates and must be run in the user's DTAP
environment before declaring M3 operationally closed.
