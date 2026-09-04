# DTAP integration overlay

This directory keeps the DecodingTrust-Agent changes required by the slime M4
harness under the same version control as the harness.

The patch is based on DecodingTrust-Agent commit
`e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` and provides:

- out-of-tree candidate task support for workflow judge helper imports; and
- opt-in disabling of shared default Docker ports so parallel M4 children can
  use the disjoint port ranges assigned by slime.
- M5 zero-LLM route verification for bridge and host-network environments;
- independent placement read-back for Slack, Gmail, WhatsApp, terminal, and
  research file injections, with explicit unsupported status for other tools; and
- the Research injection route fix for TaskExecutor's dynamic arXiv port.

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

M5 placement verification currently supports:

- Slack channel writes carrying `channel_name` and a message payload. Direct
  messages without a channel locator are not yet covered.
- Gmail `inject_email`, using the Mailpit list API followed by the message-detail
  API to compare recipient, subject, and body.
- WhatsApp `send_whatsapp_message`, using `get_conversation` for read-back.
- Terminal `inject_readme` and `inject_file`, read directly from the target
  container.
- Research `inject_readme` and `inject_paper_notes`, read directly from the
  target container.

Read-only tools whose names start with `get_`, `list_`, `read_`, or `search_`
return `not_applicable`. All other mutating tools return `unsupported`; this
currently includes Customer Service, Salesforce, Legal, Finance, OS Filesystem,
Calendar, Zoom, Google Docs/Form, Snowflake, Databricks, PayPal, Ecommerce,
Custom Website, Travel, Telecom, Atlassian, Telegram, GitHub, Hospital, Windows,
and macOS mutations, plus the uncovered Slack/Terminal/Research variants.

Five enabled injection registries do not yet have even a route definition:
`chase-injection`, `robinhood-injection`, `reddit-injection`,
`googlesheets-injection`, and `googledrive-injection`. They fail closed in route
mode and need entries in both `ROUTES` and `TARGETS` before placement work.

To add an adapter:

1. Identify a read-only API, database query, or container file read that does
   not reuse the injection MCP result as evidence.
2. Add its port binding to `ROUTES` and target environment to `TARGETS` if they
   are absent. Test both bridge port publishing and host networking.
3. Reuse task-scoped victim credentials via `build_readback_environments`;
   never log tokens or the injected payload.
4. Locate the written entity by a stable key, then compare the semantic fields.
   For list APIs such as Mailpit, fetch the detail record before comparing.
5. Return only status, locator, and SHA-256 digest. A mismatch or unavailable
   read path must raise `VerificationError`, not silently become `unsupported`.
6. Add positive, mismatch, authentication, and strict-mode unit tests, followed
   by a real indirect-task smoke test.

Windows and macOS need a guest-side read-only endpoint for file, registry/plist,
and Office state. Until such an endpoint is available, keep their placement
mutators unsupported rather than treating injection-MCP success as proof.

## Reproducing M0 through M5

Start from the DTAP virtual environment and apply the slime-managed overlay:

```bash
source /home/pjy0422/workspace/dtap/bin/activate
cd /home/pjy0422/workspace/slime

examples/dtap_agent_rl/dtap_integration/apply.sh \
  /home/pjy0422/workspace/DecodingTrust-Agent

export PYTHONPATH=/home/pjy0422/workspace/DecodingTrust-Agent:${PYTHONPATH:-}
```

Run the deterministic suites:

```bash
pytest -q examples/dtap_agent_rl/tests
pytest -q /home/pjy0422/workspace/DecodingTrust-Agent/tests/test_env_verification.py
```

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

# Mixed result: Slack verified and Legal unsupported; omit --strict.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/legal/malicious/indirect/strategy_leak/6

# Travel injection succeeds but placement is unsupported; omit --strict.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004
```

The originally inspected `finance/.../action_reversal/1` task contains only a
`type: tool` attack step and therefore reports `no_environment_injections`; it
does not exercise M5. Adding `--strict` to Legal or Travel is expected to fail
until those placement adapters are implemented. The smoke command always tears
down the injection MCP processes and Docker environments in a `finally` block.
