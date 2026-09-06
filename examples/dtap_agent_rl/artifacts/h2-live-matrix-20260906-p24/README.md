# H=2 Linux live matrix (parallel 24)

This run contains one representative direct and indirect task for each of the
12 Linux-supported DTAP domains. Windows and macOS are intentionally excluded.

- Policy model: `deepseek-v4-flash`
- Victim model: `deepseek-v4-flash`
- Victim harness: OpenClaw
- Submission horizon: `H=2`
- Initial worker parallelism: `24`
- Final result: `24/24` evaluation runs completed
- Placement coverage: `17/17` applicable runs

The first parallel wave completed 20 cases. One provider-rate-limited case and
three cases exposing Finance placement and Medical judge-timeout defects were
rerun in place with `--resume --max-parallel 24` after those defects were fixed.
The final `summary.json` therefore describes the complete repaired run.

Each case keeps its full policy trajectory plus per-submission artifacts under
`attempts/attempt-0001`, `attempts/attempt-0002`, as applicable. Successful
first submissions terminate early and consequently have only one attempt.
