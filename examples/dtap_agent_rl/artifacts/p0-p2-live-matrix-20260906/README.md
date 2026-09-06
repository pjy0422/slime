# P0-P2 live matrix (2026-09-06)

This directory preserves the viewer-ready artifacts from the Linux DTAP M6
live matrix. The policy and victim both used `deepseek-v4-flash`, and the
victim ran through the OpenClaw harness.

- Scope: 12 Linux domains, direct and indirect (`24` evaluations)
- Result: `24` passed, `0` failed
- Placement coverage: `18/18` applicable evaluations
- Placement actions: `41` verified from `56` attempts
- Excluded platform domains: Windows and macOS

Each `<domain>/<threat-model>/` directory contains the policy JSONL, victim
trajectory and redacted MCP events, original and submitted configs, placement
episode state, and judge output. `summary.json` is the aggregate
machine-readable report.

Render any case with the repository's trajectory viewer, for example:

```bash
dtap-traj \
  examples/dtap_agent_rl/artifacts/p0-p2-live-matrix-20260906/workflow/indirect \
  -o /tmp/workflow-indirect.html
```

Credentials are not stored in this bundle. The Ollama credential used for the
live run was supplied only through the process environment.
