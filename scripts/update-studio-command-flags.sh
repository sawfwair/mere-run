#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MERERUN_UPDATE_STUDIO_FLAGS=1 swift test \
  --filter MereRunAppTests.CommandFlagsGenerationTests/testGeneratedFlagConstantsMatchTheContract

echo "Regenerated apps/macos/MereRunStudio/Catalog/CommandFlags.swift from MereRunCapabilityCatalog."
