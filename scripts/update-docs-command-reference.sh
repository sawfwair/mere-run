#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MERERUN_UPDATE_DOCS=1 swift test \
  --filter DocumentationContractTests/testGeneratedCommandInventoriesMatchCLI

echo "Updated the generated CLI inventories in docs/index.md, docs/getting-started.md, and docs/cli.md."
