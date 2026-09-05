#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/DecodingTrust-Agent" >&2
  exit 2
fi

dtap_root=$(realpath "$1")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if ! git -C "$dtap_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not a DTAP git checkout: $dtap_root" >&2
  exit 2
fi

# Later overlay patches intentionally refine lines introduced by earlier ones,
# which makes per-file reverse checks insufficient for a second invocation.
# Record the digest only after the complete ordered series succeeds. Keeping the
# marker under this worktree's git metadata avoids modifying DTAP source files.
patches=(
  "$script_dir/patches/m4-runtime-integration.patch"
  "$script_dir/patches/m5-environment-verification.patch"
  "$script_dir/patches/m6-placement-receipts.patch"
  "$script_dir/patches/m6-domain-placement.patch"
  "$script_dir/patches/m6-openclaw-deepseek.patch"
)
overlay_digest=$(sha256sum "${patches[@]}" | sha256sum | cut -d' ' -f1)
marker=$(git -C "$dtap_root" rev-parse --git-path dtap-agent-rl-overlay.sha256)
if [[ -f "$marker" ]] && [[ $(<"$marker") == "$overlay_digest" ]]; then
  echo "DTAP agent RL overlay already applied"
  exit 0
fi

for patch_file in "${patches[@]}"; do
  if git -C "$dtap_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "Already applied: $(basename "$patch_file")"
    continue
  fi
  git -C "$dtap_root" apply --check "$patch_file"
  git -C "$dtap_root" apply "$patch_file"
  echo "Applied: $(basename "$patch_file")"
done

printf '%s\n' "$overlay_digest" > "$marker"
