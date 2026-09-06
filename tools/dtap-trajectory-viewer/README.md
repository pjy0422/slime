# DTAP Policy + Victim Trajectory Viewer

The package now supports two modes:

1. `dtap-traj serve`: a local indexed explorer for many RL trajectories.
2. Legacy single-file HTML export for sharing one self-contained trajectory.

The explorer keeps the artifact directory as the source of truth. SQLite stores only searchable episode metadata and artifact paths, while the existing parser lazily loads policy/victim/config data for the selected episode.

## Local explorer

```bash
cd tools/dtap-trajectory-viewer
pip install -e .

dtap-traj serve \
  ../../examples/dtap_agent_rl/artifacts/p0-p2-live-matrix-20260906 \
  --watch --open
```

The UI opens at `http://127.0.0.1:8765` by default and supports:

- run/domain/threat-model/status/attack-success filters,
- free-text search over episode IDs, risk categories and artifact paths,
- lazy Policy trajectory view,
- lazy Victim trajectory view,
- side-by-side Combined view,
- DTAP task/attack judge results with deterministic and LLM-as-judge labels,
- the trusted reward-firewall verdict kept separate from raw judge metadata,
- original/submitted Config Diff,
- server-side pagination,
- live re-indexing with `--watch`.

### Judge semantics

The two DTAP judge outcomes are independent:

- **Task judge** asks whether the victim completed the original user task.
- **Attack judge** asks whether the adversary's separate attack objective was
  achieved.

For an indirect attack, `task succeeded / attack failed` is the desired safe
outcome, while `task succeeded / attack succeeded` means the useful task was
completed but the injected objective also took effect. In some direct tasks the
original request is itself the malicious objective, so their task and attack
judge implementations can intentionally share a predicate.

The episode list reports the attack judge as **succeeded**, **failed**, or **not
evaluated**. It does not infer containment from a missing/null verdict.

The explorer has no authentication and is intended for local use. Keep the
default loopback host when trajectories contain sensitive evaluation data.

## Opening the explorer from a remote machine

Start the explorer on the server using its safe loopback default:

```bash
dtap-traj serve /path/to/artifacts --watch
```

When using VS Code Remote SSH, open the **Ports** view, choose **Forward a
Port**, enter `8765`, and then use the forwarded URL shown by VS Code. A plain
SSH client can do the same with:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@server
```

Then open <http://127.0.0.1:8765/> on the client machine.

For a temporary clickable HTTPS URL, install `cloudflared` and run:

```bash
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765
```

The generated `trycloudflare.com` URL is public and has no authentication or
uptime guarantee. Use it only for sanitized/public artifacts, do not share the
URL, and stop `cloudflared` when finished. For persistent or sensitive use,
configure a named Cloudflare Tunnel protected by Cloudflare Access.

`dtap-traj index <artifact-root>` can populate or refresh the SQLite index without starting the server. The default database is `<artifact-root>/.dtap-traj.sqlite3` and uses SQLite WAL mode.

## Legacy standalone HTML export

The existing behavior remains available and unchanged:

```bash
dtap-traj /path/to/artifact-dir -o trajectory.html --open
```

The legacy renderer produces one self-contained HTML page. `--json` still exposes the parsed viewer data model without rendering.

## Artifact layout

A viewer-ready episode directory typically contains:

```text
episode/
├── episode-manifest.json
├── result.json
├── policy.jsonl
├── policy-prompt.txt
├── victim-trajectory.json
├── victim-mcp-events.jsonl
├── original-config.yaml
├── submitted-config.yaml
├── judge-result.json
├── judge-verdict.json
└── attempts/
    ├── attempt-0001/  # submitted config, victim trajectory, judges
    └── attempt-0002/
```

For an H-loop, use the submission-attempt selector above the detail view to
switch the submitted config, victim trajectory, and DTAP judges together. The
policy tab remains the complete episode policy trajectory. Top-level artifact
files are treated as latest-attempt compatibility aliases.

The explorer does not rewrite these files. It indexes metadata and loads the
selected bundle through the existing `dtap_traj.parser` implementation.
