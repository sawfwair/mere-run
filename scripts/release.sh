#!/usr/bin/env bash
set -euo pipefail

# Full mere.run release: build → sign → DMG → notarize.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_NUMBER="$(git -C "$ROOT_DIR" rev-list --count HEAD)"
GIT_SHORT_HASH="$(git -C "$ROOT_DIR" rev-parse --short HEAD)"
echo "[release] mere.run build $BUILD_NUMBER (commit $GIT_SHORT_HASH)"

echo ""
echo "[release] step 1/3: build + sign..."
"$ROOT_DIR/scripts/build_release.sh"

echo ""
echo "[release] step 2/3: create DMG..."
"$ROOT_DIR/scripts/make_dmg.sh"

echo ""
echo "[release] step 3/3: notarize..."
"$ROOT_DIR/scripts/notarize.sh"

DMG_PATH="$ROOT_DIR/dist/mere-run.dmg"
DMG_SIZE="$(du -h "$DMG_PATH" | cut -f1 | xargs)"

echo ""
echo "[release] done."
echo "  artifact: $DMG_PATH ($DMG_SIZE)"
echo "  build:    $BUILD_NUMBER"
echo "  commit:   $GIT_SHORT_HASH"
