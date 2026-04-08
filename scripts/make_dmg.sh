#!/usr/bin/env bash
set -euo pipefail

# Create a mere.run DMG containing the signed binary, Claude Code skill, and installer.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN_PATH="${BIN_PATH:-$(swift build -c release --product mere.run --show-bin-path --package-path "$ROOT_DIR")/mere.run}"
DMG_PATH="${DMG_PATH:-$ROOT_DIR/dist/mere-run.dmg}"
VOLUME_NAME="${VOLUME_NAME:-mere.run}"
SIGN_IDENTITY="${SIGN_IDENTITY:-Developer ID Application: Kyle McCullough (S5JDPCT8RC)}"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "[make_dmg] binary not found at: $BIN_PATH" >&2
  echo "[make_dmg] run scripts/build_release.sh first." >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/dist"

STAGING_DIR="$(mktemp -d)"
TEMP_DMG="$(mktemp -u).dmg"

cleanup() {
  hdiutil detach "/Volumes/$VOLUME_NAME" 2>/dev/null || true
  rm -rf "$STAGING_DIR"
  rm -f "$TEMP_DMG"
}
trap cleanup EXIT

echo "[make_dmg] staging..."

# Binary
cp "$BIN_PATH" "$STAGING_DIR/mere.run"
chmod +x "$STAGING_DIR/mere.run"

# Skill
mkdir -p "$STAGING_DIR/skills/mere-run"
cp -R "$ROOT_DIR/.claude/skills/mere-run/" "$STAGING_DIR/skills/mere-run/"

# Installer
cp "$ROOT_DIR/scripts/install.sh" "$STAGING_DIR/install.sh"
chmod +x "$STAGING_DIR/install.sh"

# README
GIT_SHORT_HASH="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
cat > "$STAGING_DIR/README.txt" <<EOF
mere.run — Local AI Inference Toolkit
A Sawfwair product.

Build: $GIT_SHORT_HASH
Requires: macOS 14+ on Apple Silicon

Install:
  Open Terminal and run:
    cd /Volumes/mere.run
    ./install.sh

  This copies mere.run to /usr/local/bin and installs the
  Claude Code skill to ~/.claude/skills/mere-run.

Quick start:
  mere.run text chat -p "Hello!"
  mere.run image generate -p "a sunset" -o sunset.png
  mere.run --help
EOF

# Calculate DMG size
STAGING_SIZE_MB=$(du -sm "$STAGING_DIR" | cut -f1)
DMG_SIZE_MB=$((STAGING_SIZE_MB + 20))

echo "[make_dmg] creating writable dmg (${DMG_SIZE_MB}MB)..."
rm -f "$TEMP_DMG"
hdiutil create \
  -volname "$VOLUME_NAME" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDRW \
  -size "${DMG_SIZE_MB}m" \
  "$TEMP_DMG" >/dev/null

echo "[make_dmg] converting to compressed dmg..."
rm -f "$DMG_PATH"
hdiutil convert "$TEMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH" >/dev/null

echo "[make_dmg] signing dmg..."
codesign --timestamp --sign "$SIGN_IDENTITY" "$DMG_PATH"

DMG_SIZE="$(du -h "$DMG_PATH" | cut -f1 | xargs)"
echo "[make_dmg] done: $DMG_PATH ($DMG_SIZE)"
