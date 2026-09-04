# DTAP integration overlay

This directory keeps the DecodingTrust-Agent changes required by the slime M4
harness under the same version control as the harness.

The patch is based on DecodingTrust-Agent commit
`e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` and provides:

- out-of-tree candidate task support for workflow judge helper imports; and
- opt-in disabling of shared default Docker ports so parallel M4 children can
  use the disjoint port ranges assigned by slime.
- M5 zero-LLM route verification for bridge and host-network environments;
- independent placement read-back for Slack, Gmail, WhatsApp, Legal, Travel,
  OS Filesystem, terminal, and research injections, with explicit unsupported
  status for other tools;
- a trusted OS Filesystem injection credential fix (the prior empty value
  produced an invalid `Authorization: Bearer ` header); and
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
- Legal `inject_into_matter` and `modify_matter`, requiring both a real matter
  and an exact match in the backend overlay state used by victim `get_matter`.
- Travel accommodation, restaurant, flight, and supported review injections,
  read through the victim-facing query endpoints.
- OS Filesystem file/append/executable/directory/symlink injections, read from
  the exact TaskExecutor-owned container (including mode and link metadata).

Read-only tools whose names start with `get_`, `list_`, `read_`, or `search_`
return `not_applicable`. All other mutating tools return `unsupported`; this
currently includes Customer Service, Salesforce, Finance, Calendar, Zoom,
Google Docs/Form, Snowflake, Databricks, PayPal, Ecommerce, Custom Website,
Telecom, Atlassian, Telegram, GitHub, Hospital, Windows, and macOS mutations,
plus uncovered variants in otherwise supported services.

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

## M6 receipt boundary

M6 exposes placement only through an episode-owned opaque action receipt. The
caller first supplies a fully schema-validated environment action to
`apply_attack_step`; `validate_placement` accepts only the returned `action_id`.
It accepts no environment name, backend query, filesystem path, or payload.

Successful read-back returns `validated_placement_locator`. Failed read-back
returns `requested_placement_locator`, a stable error code, and only the fields
that locate the entity (for example `kwargs.matter_id`, `kwargs.city` plus
`kwargs.name`, or `kwargs.file_path`). The locator is derived from the caller's
own submitted action. It is a targeted repair hint, not a backend listing or a
claim that some unverified alternate path exists.

Each apply uses a disposable sandbox and a sealed result file. Unknown fields,
symlinks, oversized output, missing results, and non-placement exceptions fail
closed as `EVALUATION_UNAVAILABLE`. Receipts are held only in the episode's
authority and disappear when that authority is unregistered.

## Reproducing M0 through M6

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

# Slack and Legal matter overlay: both expected verified in strict mode.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/legal/malicious/indirect/strategy_leak/6 \
  --strict

# Travel accommodation placement: expected verified in strict mode.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --strict

# OS Filesystem placement from the exact sandbox container: expected verified.
python -m examples.dtap_agent_rl.scripts.smoke_m5_env \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/os-filesystem/malicious/indirect/prohibited-ai-practices-and-profiling/11 \
  --strict
```

The originally inspected `finance/.../action_reversal/1` task contains only a
`type: tool` attack step and therefore reports `no_environment_injections`; it
does not exercise M5. The smoke command always tears down the injection MCP
processes and Docker environments in a `finally` block.

Run a real generated-plan episode (policy, victim, placement gate, and judge):

```bash
export ANTHROPIC_API_KEY='...'
export DTAP_POLICY_ANTHROPIC_BASE_URL=https://provider.example
export DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN=1
export DTAP_VICTIM_ANTHROPIC_BASE_URL=https://provider.example
export DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN=1
export DTAP_ENV_VERIFICATION=placement
export DTAP_ENV_VERIFICATION_STRICT=1

python -m examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python \
  --policy-model glm-5.2 \
  --victim-model glm-5.2
```

This prompt contains no plan or payload. It requires the policy to inspect the
live task/surface, author and validate every step, and submit exactly once. An
attack miss is a valid reward-zero episode; infrastructure/security failure is
not. Set `DTAP_M5_GLM_TASK_DIR` and `DTAP_M5_DTAP_ROOT` to include the same gate
in the opt-in pytest integration suite.

M5 remains a trusted auxiliary check. It does not expose read-back to the policy,
does not issue policy-visible `action_id` receipts, and does not alter reward.
Those policy-scoped receipt semantics belong to M6.

Run the real M6 receipt path (no victim or judge):

```bash
python -m examples.dtap_agent_rl.scripts.smoke_m6_placement \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/legal/malicious/indirect/strategy_leak/6 \
  --tool legal-injection:inject_into_matter \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python

python -m examples.dtap_agent_rl.scripts.smoke_m6_placement \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/travel/malicious/indirect/off-platform-payments/004 \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python

python -m examples.dtap_agent_rl.scripts.smoke_m6_placement \
  --task-dir /home/pjy0422/workspace/DecodingTrust-Agent/dataset/os-filesystem/malicious/indirect/prohibited-ai-practices-and-profiling/11 \
  --turn-id 1 \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --python /home/pjy0422/workspace/dtap/bin/python
```

The OS example's source template places the global environment injection at turn
3 while its projected task is one instruction; `--turn-id 1` matches the public
M2-M6 candidate contract and does not change placement semantics.

For the generated-plan M6 release gate, use the same provider variables as the
M5 command and add `--m6-placement`. Set `DTAP_M6_GLM_TASK_DIR` and
`DTAP_M6_DTAP_ROOT` to run the opt-in pytest gate.
