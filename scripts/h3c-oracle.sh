#!/usr/bin/env bash

set -euo pipefail

readonly H3C_REPOSITORY="https://github.com/antirez/h3.c.git"
readonly H3C_REVISION="f0dbe7699250c4943ec148ed7c2c16031fee8d05"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
oracle_root="${MERERUN_H3C_ORACLE_ROOT:-$repo_root/.build/h3c-oracle}"
source_root="$oracle_root/source"
mode="${1:-setup}"
if [[ $# -gt 0 ]]; then
  shift
fi

usage() {
  cat <<'EOF'
usage: scripts/h3c-oracle.sh [setup|test|pin|run] [h3 arguments...]

  setup  Fetch the pinned h3.c revision and build its CLI and static library.
  test   Build the oracle and run h3.c's deterministic test suite.
  pin    Print the repository, immutable revision, and local checkout path.
  run    Build the oracle, then pass all remaining arguments to its h3 CLI.

Environment:
  MERERUN_H3C_ORACLE_ROOT  Override the ignored local oracle directory.
  MERERUN_H3C_JOBS         Override the build parallelism.
EOF
}

checkout_oracle() {
  mkdir -p "$oracle_root"
  if [[ ! -d "$source_root/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$H3C_REPOSITORY" "$source_root"
  fi

  git -C "$source_root" fetch --depth 1 origin "$H3C_REVISION"
  git -C "$source_root" checkout --detach --force "$H3C_REVISION"

  local resolved_revision
  resolved_revision="$(git -C "$source_root" rev-parse HEAD)"
  if [[ "$resolved_revision" != "$H3C_REVISION" ]]; then
    echo "error: h3.c resolved to $resolved_revision, expected $H3C_REVISION" >&2
    exit 1
  fi
}

build_oracle() {
  checkout_oracle
  local jobs="${MERERUN_H3C_JOBS:-$(sysctl -n hw.logicalcpu)}"
  make -C "$source_root" -j"$jobs"
}

case "$mode" in
  setup)
    build_oracle
    ;;
  test)
    build_oracle
    make -C "$source_root" test
    ;;
  pin)
    printf 'repository=%s\nrevision=%s\npath=%s\n' \
      "$H3C_REPOSITORY" "$H3C_REVISION" "$source_root"
    ;;
  run)
    build_oracle
    exec "$source_root/h3" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown mode: $mode" >&2
    usage >&2
    exit 64
    ;;
esac
