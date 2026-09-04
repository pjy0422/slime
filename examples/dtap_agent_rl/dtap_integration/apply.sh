#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/DecodingTrust-Agent" >&2
  exit 2
fi

dtap_root=$(realpath "$1")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -d "$dtap_root/.git" ]]; then
  echo "not a DTAP git checkout: $dtap_root" >&2
  exit 2
fi

for patch_file in "$script_dir"/patches/*.patch; do
  if git -C "$dtap_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "Already applied: $(basename "$patch_file")"
    continue
  fi
  git -C "$dtap_root" apply --check "$patch_file"
  git -C "$dtap_root" apply "$patch_file"
  echo "Applied: $(basename "$patch_file")"
done
