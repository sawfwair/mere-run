#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -gt 1 ]]; then
  echo "Usage: scripts/check-model-boundaries.sh [package-manifest.json]" >&2
  exit 64
fi

if [[ $# -eq 1 ]]; then
  package_json="$1"
else
  package_json="$(mktemp "${TMPDIR:-/tmp}/mere-run-model-package.XXXXXX")"
  trap 'rm -f "$package_json"' EXIT
  swift package dump-package >"$package_json"
fi

swift scripts/check-model-boundaries.swift "$package_json"
