# DTAP Agent RL — M0 + M1

This directory is intended to be copied into the **slime repository** at
`examples/dtap_agent_rl/`. DTAP remains an external pinned dependency.

## M0

M0 creates trusted `TaskSnapshot` state, an allowlisted `PolicyTaskSpec`, and a
normalized `AttackSurface`. Benchmark example attacks (`Attack.attack_turns`) do
not participate in observation or action-surface construction.

## M1

M1 adds a minimal `DTAPClaudeCodeHarness(ClaudeCodeHarness)` plus a host-side,
read-only MCP boundary.

Policy-visible MCP tools are exactly:

- `get_task_spec()`
- `get_attack_surface()`

M1 requires `fastmcp>=2.6` for HTTP bearer/header support.

Episode identity is **not** a tool argument. Claude Code expands
`${DTAP_EPISODE_TOKEN}` into the HTTP `Authorization` header from the environment.
The MCP server resolves that opaque capability against an in-memory
`EpisodeRegistry`.

### Trust boundary

- `config.yaml`, attack examples, DTAP task paths, judge state: host/trusted only.
- `PolicyTaskSpec`, `AttackSurface`: policy-visible.
- `DTAPClaudeCodeHarness`: does not import DTAP and never sees a task directory.
- `EpisodeRegistry`: never enumerable from MCP.
- M1 tools are read-only; mutation/evaluation arrive in M2/M3.

## Wiring from the future `generate.py`

```python
snapshot = load_task_snapshot(task_dir)
view = await build_episode_view(snapshot, live_catalog)

with registered_episode(registry, token=session_id, view=view):
    await DTAPClaudeCodeHarness().run(
        sandbox,
        workdir="/workspace/empty",
        session_id=session_id,
        adapter_url=adapter_url,
        time_budget_sec=300,
        prompt=prompt,
    )
```

The host-side FastMCP server must run in the same process/service as the `EpisodeRegistry` and be reachable from the sandbox, and the host
process should export the full MCP endpoint, e.g.:

```bash
export DTAP_HARNESS_URL=http://<reachable-host>:19090/mcp/
```

`DTAPClaudeCodeHarness` writes an MCP config containing only env placeholders and
injects the episode capability at process launch.

## Tests

Inside a slime checkout:

```bash
pytest -q examples/dtap_agent_rl/tests
```

The tests cover:

- M0 no-solution-leak regression
- token lifecycle and cross-episode isolation
- cleanup on exception
- bearer parsing
- MCP payloads contain no token
- Claude Code base config is preserved
- MCP config contains env placeholders, not the token
- slime Anthropic adapter env is preserved
- native Claude Code tools are explicitly denied
- exact M1 MCP tool list (when FastMCP is installed)

## HTTP MCP deployment

Create the server once per slime worker/service process and keep the registry in
that same process:

```python
registry = EpisodeRegistry()
mcp = create_mcp_server(registry)
mcp.run(transport="http", host="0.0.0.0", port=19090, stateless_http=True)
```

For an already-running asyncio process, use `run_async()` instead. A production
M1 deployment should run one shared service per worker/node rather than one MCP
process per episode.


## Direct Anthropic API smoke (no RL / no slime actor)

After the M0-M1 test suite is green, you can exercise the real Claude Code +
Anthropic API + HTTP/Bearer MCP boundary without SGLang/Ray/RL:

```bash
source /home/pjy0422/workspace/dtap/bin/activate
cd /home/pjy0422/workspace/slime

export PYTHONPATH=/path/to/DecodingTrust-Agent:$PYTHONPATH
export ANTHROPIC_API_KEY='sk-ant-...'

python -m examples.dtap_agent_rl.scripts.smoke_m1_api \
  --task-dir /path/to/DecodingTrust-Agent/dataset/.../<task-id> \
  --model sonnet
```

The smoke performs five phases:

1. Start the real DTAP Docker environment and discover live MCP schemas.
2. Tear the DTAP runtime back down; retain only sanitized `EpisodeView`.
3. Start the host-side read-only FastMCP server with a random bearer capability.
4. Run local Claude Code directly against Anthropic (`ANTHROPIC_API_KEY`), requiring
   calls to `get_task_spec` and `get_attack_surface` only.
5. Assert both calls occurred, no benchmark example/config path/capability leaked,
   original `config.yaml` stayed unchanged, and the registry was cleaned.

The script explicitly removes `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` from
Claude's child environment so a previously configured slime adapter/gateway cannot
accidentally receive this direct-API smoke request.

Use `--show-stream` to print Claude Code's full stream-json output. By default
`Attack.additional_information` is hidden; opt in with
`--expose-additional-information` only after auditing that field for the task set.

This smoke does **not** run the DTAP victim agent, attack mutation, judge, reward,
or any RL training loop. Those begin in later milestones.
