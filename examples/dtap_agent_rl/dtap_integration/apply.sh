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
  "$script_dir/patches/p0-p2-linux-stabilization.patch"
  "$script_dir/patches/p3-judge-reliability.patch"
  "$script_dir/patches/p4-h2-live-stability.patch"
  "$script_dir/patches/p5-windows-macos-placement.patch"
  "$script_dir/patches/m7-feedback-observability.patch"
  "$script_dir/patches/m7-feedback-v2.patch"
  "$script_dir/patches/m7-exact-locators.patch"
  "$script_dir/patches/m7-domain-feedback.patch"
  "$script_dir/patches/m7-feedback-boundary-matrix.patch"
)
# Hash contents, not absolute filenames: the checkout may be reached through a
# symlink and must still produce the same idempotency marker.
overlay_digest=$(
  for patch_file in "${patches[@]}"; do
    sha256sum "$patch_file" | cut -d' ' -f1
  done | sha256sum | cut -d' ' -f1
)
marker=$(git -C "$dtap_root" rev-parse \
  --path-format=absolute --git-path dtap-agent-rl-overlay.sha256)
if [[ -f "$marker" ]] && [[ $(<"$marker") == "$overlay_digest" ]]; then
  echo "DTAP agent RL overlay already applied"
  exit 0
fi

# Recover idempotency for checkouts patched by an older apply.sh whose marker
# digest depended on the spelling of the slime path.
latest_patch=${patches[${#patches[@]}-1]}
if git -C "$dtap_root" apply --reverse --check "$latest_patch" >/dev/null 2>&1; then
  printf '%s\n' "$overlay_digest" > "$marker"
  echo "DTAP agent RL overlay already applied"
  exit 0
fi

# An existing M4-M6 checkout has refinements that make replaying intermediate
# patches ambiguous. If the new delta applies cleanly, advance just that delta.
if git -C "$dtap_root" apply --check "$latest_patch" >/dev/null 2>&1; then
  git -C "$dtap_root" apply "$latest_patch"
  printf '%s\n' "$overlay_digest" > "$marker"
  echo "Applied: $(basename "$latest_patch")"
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
