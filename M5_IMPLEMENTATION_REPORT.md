# M5 implementation report

## Implemented

- Network-mode-aware Docker route proof for bridge and host-network DTAP services.
- Binding of started injection MCP child configuration to TaskExecutor ports and
  compose project identity.
- Dynamic `ARXIV_API_PORT` routing for the Research injection server.
- Independent placement adapters for Slack channel messages, Gmail/Mailpit,
  WhatsApp messages, terminal files, and research files.
- Explicit `verified`, `not_applicable`, and `unsupported` placement states.
- `off`, `route`, and `placement` rollout modes plus optional strict coverage.
- Fail-closed injection failures without feeding placement into RL reward.

## Verification

```text
DTAP verifier tests: 15 passed
14 representative benchmark-domain injection shapes (15 target routes): covered
clean-checkout patch apply + repeat apply: passed
python compile: passed
slime M0-M4 regression: 128 passed, 2 skipped
live WhatsApp host-network route: passed
live WhatsApp placement: correctly rejected backend HTTP failure previously
misreported by DTAP as MCP success
live Research indirect placement: verified by container file read-back
live Legal indirect Slack placement: verified with the task MCP credential
live Finance indirect Gmail placement: verified through Mailpit message detail
live Travel indirect injection: succeeded; placement explicitly unsupported
```

Windows and macOS route shapes are covered by deterministic tests. Their guest
filesystem/Office semantic read-back remains unsupported until those VM services
are available for a live integration gate. Other service mutators currently return
`unsupported` rather than claiming a false placement proof.
