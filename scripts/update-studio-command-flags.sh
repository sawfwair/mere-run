#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

MERERUN_UPDATE_STUDIO_FLAGS=1 swift test \
  --filter StudioKitTests.CommandFlagsGenerationTests/testGeneratedFlagConstantsMatchTheContract

echo "Regenerated apps/macos/StudioKit/Catalog/CommandFlags.swift from MereRunCapabilityCatalog."
