# Portable DTAP development environment

This directory captures the Python environment currently used for the DTAP
agent-RL deterministic tests and API E2E runs. It is a reproducible rebuild,
not a byte-for-byte copy of a virtualenv: virtualenv activation scripts and
editable installs contain absolute paths and cannot safely be moved between
servers.

The snapshot supports Linux x86_64 with CPython 3.12.13 and an NVIDIA driver
capable of running the recorded PyTorch CUDA build. The target host also needs:

- `git` access to the private slime and DecodingTrust-Agent repositories;
- `uv` at the version recorded in `runtime-manifest.json`;
- a working NVIDIA driver and `nvidia-smi`;
- Docker with permission to run `docker info`;
- the recorded Node, Claude Code, and OpenClaw command versions used by the
  current policy/victim E2E harness.

From a fresh slime checkout, rebuild the environment next to the repository:

```bash
cd /path/to/slime

./examples/dtap_agent_rl/scripts/bootstrap_portable_venv.sh \
  --venv ../dtap \
  --dtap-root ../DecodingTrust-Agent

source ../dtap/bin/activate
```

The bootstrap performs these operations:

1. installs the exact Python patch release through `uv`;
2. clones DecodingTrust-Agent at the pinned commit if it is absent;
3. applies the ordered, idempotent slime-managed DTAP overlay;
4. installs all captured third-party distributions from the exact lock;
5. installs slime, DTAP, and the trajectory viewer from their new local paths;
6. verifies the lock digest, package versions, repository identity, overlay,
   Docker access, and CUDA availability.

Use `--dry-run` to inspect paths and commands without creating anything. Use
`--skip-system-checks` only for a CPU validation host; a successful skipped
check does not establish GPU readiness.

The snapshot deliberately does **not** contain API keys, auth tokens, Docker
volumes/images, model checkpoints, datasets outside the repositories, result
artifacts, or the Windows/macOS guest disks. Secrets must be supplied through
environment variables at runtime. Large external assets remain governed by
`dtap_integration/runtime-lock.json` and must be transferred or fetched
separately.

This snapshot reproduces the current DTAP/API development venv. It does not
replace slime's `build_conda.sh` native training stack; F9 optimizer work still
needs that CUDA 12.9, SGLang, and Megatron environment on the target server.

## Refreshing the snapshot

After intentionally changing the source venv, regenerate both tracked files:

```bash
source /path/to/current/dtap/bin/activate
cd /path/to/slime

python -m examples.dtap_agent_rl.scripts.portable_venv export \
  --python "$VIRTUAL_ENV/bin/python" \
  --slime-root "$PWD" \
  --dtap-root /path/to/DecodingTrust-Agent \
  --output-dir examples/dtap_agent_rl/environment
```

Export fails closed on unexpected editable installs, local file requirements,
embedded URL credentials, a missing DTAP overlay marker, or unsupported host
platforms. Review the lock diff before committing it.
