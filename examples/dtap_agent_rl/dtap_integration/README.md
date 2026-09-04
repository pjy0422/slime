# DTAP integration overlay

This directory keeps the DecodingTrust-Agent changes required by the slime M4
harness under the same version control as the harness.

The patch is based on DecodingTrust-Agent commit
`e0323a521ba4ef88f8e14c1eccf68d0a3d19a458` and provides:

- out-of-tree candidate task support for workflow judge helper imports; and
- opt-in disabling of shared default Docker ports so parallel M4 children can
  use the disjoint port ranges assigned by slime.

Apply it to a DTAP checkout with:

```bash
examples/dtap_agent_rl/dtap_integration/apply.sh \
  /path/to/DecodingTrust-Agent
```

The script is idempotent: it reports an already-applied patch without changing
the checkout. It deliberately does not commit inside the DTAP repository.
