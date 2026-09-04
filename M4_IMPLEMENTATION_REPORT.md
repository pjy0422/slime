# M4 implementation report

## Outcome

M4 production code is implemented over the M3 H-submission environment. It keeps
the four-tool policy contract and adds fail-closed reward, credential, filesystem,
process-environment, transport, sandbox-attestation, and parallel scheduling
boundaries.

## Implemented components

- `security_policy.py`: explicit Q and JSON/content/judge/HTTP/concurrency limits;
  minimum DTAP child environment with credential deny rules
- `authority.py`: independent adapter/MCP/public identifiers and one atomic registry
- `policy_contract.py`: canonical typed receipts, forbidden-field checks, leakage guard
- `reward_firewall.py`: bounded, no-follow, duplicate-key-safe boolean verdict reader
- `integrity.py`: complete task-tree SHA-256 manifests and no-follow verified copy
- `scheduler.py`: worker-scoped FIFO global concurrency and bounded queue
- `sandbox_policy.py`: mandatory filesystem/process/network security attestation
- `transport_security.py`: pre-parser ASGI body/header limits
- `audit.py`: content-free trusted event schema
- `m4.py`: hardened episode lifecycle and authority revocation
- `generate_m4.py`: one slime session, allowlisted sample metadata, safe aborts
- `M4ClaudeCodeHarness`: independent bearer, strict MCP, exact four tools, isolated
  config home, disabled hooks, and forbidden arbitrary launch overrides
- `DtapAttemptRunner` M4 mode: explicit shared scheduler, child env allowlist, exact
  start/verdict control files, reward firewall, process-group cleanup, and disjoint
  per-child DTAP port ranges
- `dtap_integration/`: versioned, idempotently applicable DTAP compatibility patch
  for out-of-tree judge helpers and parallel Docker port allocation
- trusted Anthropic-compatible provider aliases: separate policy/victim endpoint
  mapping without allowing policy adapter credentials into the DTAP child
- `smoke_m4_claude_boundary.py`: direct Claude strict-contract/canary smoke
- `test_m4_real_dtap_optional.py`: parallel real DTAP/Docker release gate

## Frozen semantics

- H counts evaluations that reach the DTAP running phase.
- Q counts every authenticated submit call; invalid submits consume Q but not H.
- Success returns reward 1 and terminates immediately.
- H exhaustion and policy-caused Q exhaustion return reward 0.
- Infrastructure and security failures have no trusted reward and remove the sample.
- Accepted attempts reveal only one success bit and public H state.
- No placement or arbitrary environment read-back API is introduced.

## Local verification

```text
128 passed
2 skipped
python -m compileall: pass
M4 direct Claude contract smoke: pass
M4 real parallel DTAP/ClaudeSDK gate: pass (1 test, 66.80s)
```

The default-skipped gates require explicit real-DTAP task environment variables.
Deployment sandbox enforcement must supply the attestation consumed by
`SandboxPolicyVerifier`; M4 refuses to start without it.
