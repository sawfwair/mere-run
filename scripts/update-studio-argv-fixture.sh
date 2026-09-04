#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MERERUN_UPDATE_STUDIO_ARGV_FIXTURE=1 swift test \
  --filter 'StudioKitTests.CommandArgumentGoldenTests|StudioKitTests.CommandDefaultDraftTests'

cat <<'EOF'
Re-recorded apps/macos/StudioKitTests/Fixtures/command-argv.txt and
command-default-drafts.txt. Review the diff: it is the command line Studio runs, so
every changed line is a change to what the app does.
EOF
