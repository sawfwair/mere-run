#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
policy_manifest_dir="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-package-policy.XXXXXX")"
trap 'rm -rf "$policy_manifest_dir"' EXIT
swift scripts/check-package-policy.swift --self-test
for variant in darwin linux-source linux-cuda-prebuilt; do
  case "$variant" in
    darwin) platform=darwin; linkage=source ;;
    linux-source) platform=linux; linkage=source ;;
    linux-cuda-prebuilt) platform=linux; linkage=cuda-prebuilt ;;
  esac
  MERERUN_PACKAGE_PLATFORM="$platform" MERERUN_MLX_SWIFT_LINKAGE="$linkage" \
    swift package dump-package > "$policy_manifest_dir/$variant.json"
  swift scripts/check-package-policy.swift "$policy_manifest_dir/$variant.json" scripts/package-policy.json "$variant"
done
