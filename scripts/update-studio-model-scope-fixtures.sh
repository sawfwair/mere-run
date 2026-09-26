#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MERERUN_UPDATE_MODEL_SCOPE_FIXTURES=1 swift test \
  --filter 'StudioKitTests.StudioModelScopeGoldenTests'

cat <<'EOF'
Re-recorded apps/macos/StudioKitTests/Fixtures/model-scope/: one file per routed command,
listing the flags each Studio surface shows, validates, and sends for every runtime family.
Review the diff: it is what a contract change does to the app's controls and command lines.
EOF
