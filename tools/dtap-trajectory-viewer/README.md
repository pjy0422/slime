# DTAP Policy + Victim Trajectory Viewer

This is the standalone DTAP leaderboard-style viewer from
`dtap-trajectory-viewer.zip`, extended for the slime DTAP attack-policy loop.
It produces one self-contained HTML page with three related views:

- the policy trajectory (`get_task_spec` through `submit_attack`),
- the DTAP victim trajectory after the submitted attack is applied, and
- a unified diff plus full text of original and submitted `config.yaml`.

The two trajectories are deliberately separate. A successful policy MCP call or
placement receipt is not presented as victim behavior. Payload highlighting is
derived from the submitted YAML, because that is the configuration evaluated by
DTAP; the original YAML is used only as the comparison baseline.

## Install and run

```bash
cd tools/dtap-trajectory-viewer
pip install -e .

dtap-traj /path/to/artifact-dir -o trajectory.html --open
```

The preferred artifact directory layout is:

```text
run-artifacts/
├── policy.jsonl
├── policy-prompt.txt
├── original-config.yaml
├── submitted-config.yaml
└── victim-trajectory.json
```

These names are auto-detected. Paths can also be supplied explicitly:

```bash
dtap-traj /path/to/run \
  --policy-trace /path/to/policy.jsonl \
  --policy-prompt /path/to/policy-prompt.txt \
  --victim-trace /path/to/victim-trajectory.json \
  --original-yaml /path/to/original/config.yaml \
  --submitted-yaml /path/to/submitted/config.yaml \
  -o trajectory.html
```

`-y/--yaml` remains a backward-compatible alias for `--submitted-yaml`.
Use `--json` to inspect the combined data model without rendering HTML.

## Policy trace format

The policy parser consumes JSONL emitted by Claude Code with:

```bash
claude -p ... --output-format stream-json --verbose
```

It understands `assistant` messages containing `thinking`, `text`, and
`tool_use` blocks and `user` messages containing `tool_result` blocks. DTAP MCP
names such as `mcp__dtap__submit_attack` are rendered as server `dtap`, tool
`submit_attack`, matching the victim viewer's existing server/tool convention.
The victim parser accepts both the original OpenClaw `messagesSnapshot` JSONL
and DTAP's framework-neutral trajectory JSON used by ClaudeSDK and other agents.

## Producing artifacts from the real M6 smoke

The GLM E2E smoke can export a viewer-ready bundle without changing production
cleanup behavior:

```bash
source /home/pjy0422/workspace/dtap/bin/activate
cd /home/pjy0422/workspace/slime
export PYTHONPATH=/home/pjy0422/workspace/DecodingTrust-Agent:$PYTHONPATH

python -m examples.dtap_agent_rl.scripts.smoke_m5_glm_e2e \
  --dtap-root /home/pjy0422/workspace/DecodingTrust-Agent \
  --task-dir /path/to/task \
  --policy-model glm-5.2 --victim-model glm-5.2 \
  --m6-placement --artifacts-dir /tmp/dtap-run

tools/dtap-trajectory-viewer/.venv/bin/dtap-traj /tmp/dtap-run \
  -o /tmp/dtap-run/trajectory.html
```

Artifact export is opt-in. Normal M4/M6 evaluation still destroys attempt
directories and does not expose hidden benchmark configs to the policy.
