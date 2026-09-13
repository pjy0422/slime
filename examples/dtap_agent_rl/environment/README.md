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
replace slime's native training stack. Keep the two environments isolated:
the DTAP venv currently carries a CUDA 13 PyTorch build, while the validated
training stack uses CUDA 12.9 native extensions.

## Preparing the native training runtime without a free GPU

The immutable image in `training-runtime.json` contains the CUDA 12.9 build of
PyTorch, SGLang, Megatron-LM, Transformer Engine, FlashAttention, Apex, and the
slime kernel dependencies. Pull and validate it without exposing a GPU:

```bash
cd /path/to/slime
python -m examples.dtap_agent_rl.scripts.training_runtime
```

Allow roughly 45 GB of local Docker image storage for the pinned runtime.

The probe disables networking inside the container, sets
`CUDA_VISIBLE_DEVICES` to empty, imports the installed training modules
(including FlashAttention 2), checks the PyTorch CUDA build and pinned Megatron
commit, and exits without allocating a GPU. Re-running is idempotent because
the image is addressed by manifest digest. Use `--skip-pull` after the image is
local or `--dry-run` to inspect the commands.

The pinned image intentionally overrides several upstream package metadata
requirements. In particular, SGLang declares FlashAttention 4 while the CUDA
12.9 training path removes it and installs FlashAttention 2, and Megatron's
NumPy 1.x pin conflicts with optional packages that declare NumPy 2.x. The
reviewed `pip check` output is committed in the manifest. The probe accepts
exactly that set and fails on either a new conflict or a missing expected
override; these entries are compatibility debt, not a claim that `pip check`
is clean.

This establishes dependency readiness only. CUDA execution, NCCL collectives,
VRAM headroom, kernel architecture compatibility, SGLang serving, Megatron
optimizer steps, and throughput measurements remain explicitly deferred until
GPU capacity is available. At that point use this same immutable image and
record the measured setup in the Performance workspace rather than rebuilding
the DTAP venv. The probe prints the canonical `runtime_setup_digest`; copy that
value into the tuning trial and M8 training records so both viewer workspaces
refer to the same installed stack.

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
