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
base_patches=(
  "$script_dir/patches/m4-runtime-integration.patch"
  "$script_dir/patches/m5-environment-verification.patch"
  "$script_dir/patches/m6-placement-receipts.patch"
  "$script_dir/patches/m6-domain-placement.patch"
  "$script_dir/patches/m6-openclaw-deepseek.patch"
  "$script_dir/patches/p0-p2-linux-stabilization.patch"
  "$script_dir/patches/p3-judge-reliability.patch"
  "$script_dir/patches/p4-h2-live-stability.patch"
  "$script_dir/patches/p5-windows-macos-placement.patch"
  "$script_dir/patches/m7-feedback-observability.patch"
  "$script_dir/patches/m7-feedback-v2.patch"
  "$script_dir/patches/m7-exact-locators.patch"
  "$script_dir/patches/m7-domain-feedback.patch"
  "$script_dir/patches/m7-feedback-boundary-matrix.patch"
  "$script_dir/patches/m7-token-observability.patch"
)
incremental_patches=(
  "$script_dir/patches/m7-explicit-tool-capabilities.patch"
  "$script_dir/patches/m7-structured-judge-status.patch"
  "$script_dir/patches/m7-exact-tool-presentation-identity.patch"
  "$script_dir/patches/p6-exact-adapter-dispatch.patch"
  "$script_dir/patches/p7-holdout-e2e-stability.patch"
)
patches=("${base_patches[@]}" "${incremental_patches[@]}")
# Hash contents, not absolute filenames: the checkout may be reached through a
# symlink and must still produce the same idempotency marker.
digest_patches() {
  for patch_file in "$@"; do
    sha256sum "$patch_file" | cut -d' ' -f1
  done | sha256sum | cut -d' ' -f1
}
overlay_digest=$(digest_patches "${patches[@]}")
marker=$(git -C "$dtap_root" rev-parse \
  --path-format=absolute --git-path dtap-agent-rl-overlay.sha256)
stored_digest=""
if [[ -f "$marker" ]]; then
  stored_digest=$(<"$marker")
fi
if [[ "$stored_digest" == "$overlay_digest" ]]; then
  echo "DTAP agent RL overlay already applied"
  exit 0
fi

# Existing overlay checkouts contain later refinements that can make reverse
# checks for earlier patches ambiguous. The marker identifies the exact known
# prefix, so only the suffix introduced after that release is inspected.
prefix_count=-1
if [[ -n "$stored_digest" ]]; then
  for ((count=${#incremental_patches[@]}; count >= 0; count--)); do
    prefix_patches=("${base_patches[@]}" "${incremental_patches[@]:0:count}")
    if [[ $(digest_patches "${prefix_patches[@]}") == "$stored_digest" ]]; then
      prefix_count=$count
      break
    fi
  done
fi
if ((prefix_count >= 0)); then
  for ((index=prefix_count; index < ${#incremental_patches[@]}; index++)); do
    patch_file=${incremental_patches[index]}
    if git -C "$dtap_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
      echo "Already applied: $(basename "$patch_file")"
    else
      git -C "$dtap_root" apply --check "$patch_file"
      git -C "$dtap_root" apply "$patch_file"
      echo "Applied: $(basename "$patch_file")"
    fi
  done
  printf '%s\n' "$overlay_digest" > "$marker"
  exit 0
fi
if [[ -n "$stored_digest" ]]; then
  echo "unrecognized DTAP agent RL overlay marker; refusing ambiguous upgrade" >&2
  exit 1
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
