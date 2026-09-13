#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
slime_root=$(git -C "$script_dir" rev-parse --show-toplevel)
environment_dir="$slime_root/examples/dtap_agent_rl/environment"
manifest="$environment_dir/runtime-manifest.json"
workspace=$(dirname "$slime_root")
venv="$workspace/dtap"
dtap_root="$workspace/DecodingTrust-Agent"
uv_bin=${UV_BIN:-uv}
skip_system_checks=0
dry_run=0

usage() {
  cat <<'EOF'
usage: bootstrap_portable_venv.sh [options]

Options:
  --venv PATH            destination venv (default: sibling dtap directory)
  --dtap-root PATH       DTAP checkout (default: sibling DecodingTrust-Agent)
  --uv PATH              uv executable (default: $UV_BIN or uv)
  --skip-system-checks   skip Docker and GPU runtime checks
  --dry-run              print the migration plan without changing files
EOF
}

while (($#)); do
  case "$1" in
    --venv) venv=$2; shift 2 ;;
    --dtap-root) dtap_root=$2; shift 2 ;;
    --uv) uv_bin=$2; shift 2 ;;
    --skip-system-checks) skip_system_checks=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

python_version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["platform"]["python_version"])' "$manifest")
dtap_remote=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repositories"]["dtap"]["remote"])' "$manifest")
dtap_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repositories"]["dtap"]["commit"])' "$manifest")
lock_file=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["lock"]["file"])' "$manifest")
uv_version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["bootstrap"]["uv_version"])' "$manifest")

run() {
  if ((dry_run)); then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if ((dry_run)); then
  echo "slime_root=$slime_root"
  echo "dtap_root=$dtap_root"
  echo "venv=$venv"
elif ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "uv $uv_version is required; run: python3 -m pip install --user uv==$uv_version" >&2
  exit 1
elif [[ $("$uv_bin" --version | awk '{print $2}') != "$uv_version" ]]; then
  echo "uv $uv_version is required; run: python3 -m pip install --user uv==$uv_version" >&2
  exit 1
fi

if ((!dry_run)) && [[ -e "$venv" ]]; then
  echo "refusing to overwrite existing venv: $venv" >&2
  exit 1
fi

run "$uv_bin" python install "$python_version"
run "$uv_bin" venv "$venv" --python "$python_version" --seed

if [[ -e "$dtap_root" ]]; then
  if [[ ! -d "$dtap_root/.git" ]]; then
    echo "DTAP destination exists but is not a git checkout: $dtap_root" >&2
    exit 1
  fi
  if [[ $(git -C "$dtap_root" rev-parse HEAD) != "$dtap_commit" ]]; then
    echo "existing DTAP checkout is not at pinned commit $dtap_commit" >&2
    exit 1
  fi
else
  run env GIT_TERMINAL_PROMPT=0 git clone --no-checkout "$dtap_remote" "$dtap_root"
  run git -C "$dtap_root" checkout --detach "$dtap_commit"
fi

run "$slime_root/examples/dtap_agent_rl/dtap_integration/apply.sh" "$dtap_root"
run "$uv_bin" pip install --python "$venv/bin/python" --no-deps -r "$environment_dir/$lock_file"
run "$uv_bin" pip install --python "$venv/bin/python" --no-deps -e "$slime_root"
run "$uv_bin" pip install --python "$venv/bin/python" --no-deps -e "$dtap_root"
run "$uv_bin" pip install --python "$venv/bin/python" --no-deps -e "$slime_root/tools/dtap-trajectory-viewer"

verify_args=(
  -m examples.dtap_agent_rl.scripts.portable_venv verify
  --manifest "$manifest"
  --slime-root "$slime_root"
  --dtap-root "$dtap_root"
  --require-editable-paths
)
if ((skip_system_checks)); then
  verify_args+=(--skip-system-checks)
fi
run "$venv/bin/python" "${verify_args[@]}"

if ((dry_run)); then
  echo "dry-run complete; no files were changed"
else
  echo "portable DTAP environment is ready"
  printf 'activate with: source %q\n' "$venv/bin/activate"
fi
