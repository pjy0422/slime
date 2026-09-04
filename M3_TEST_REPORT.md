# M3 implementation test report

M3 production code is implemented against the frozen contract.

Authored coverage:

- 8 episode-runtime cases (including three invalid `H` values)
- 8 candidate-config and filesystem cases
- 13 submission/coordinator cases (including parameterized strict envelopes)
- 3 FastMCP/HTTP cases
- 4 policy-trajectory cases
- 3 custom-generate cases
- 3 subprocess runner cases
- 1 end-to-end M3 wiring case
- 1 opt-in real DTAP reset case

The exact collected count can vary because parameterized cases are expanded by
pytest. The real test is skipped unless `DTAP_M3_RESET_TASK_DIR` is set. FastMCP
tests skip when FastMCP is absent. In the release environment neither the DTAP
compatibility tests nor FastMCP tests may be skipped.

Validation performed in this environment:

- `python -m compileall`: pass
- complete local suite: `73 passed, 2 skipped`
- FastMCP in-memory and HTTP/Bearer tests: pass
- fresh subprocess, timeout process-group cleanup, H-budget, config, coordinator,
  generate, and wiring tests: pass

The two skips are deliberate external gates: current DTAP helper compatibility and
the real two-attempt Docker reset sentinel. This environment has no DTAP checkout or
Docker benchmark fixture, so those results are not claimed.
