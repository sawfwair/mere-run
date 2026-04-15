#!/usr/bin/env bash
set -euo pipefail

# Create a mere.run DMG containing the signed binary, Claude Code skill, and installer.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN_PATH="${BIN_PATH:-$(swift build -c release --product mere.run --show-bin-path --package-path "$ROOT_DIR")/mere.run}"
DMG_PATH="${DMG_PATH:-$ROOT_DIR/dist/mere-run.dmg}"
VOLUME_NAME="${VOLUME_NAME:-mere.run}"
SIGN_IDENTITY="${SIGN_IDENTITY:-Developer ID Application: Kyle McCullough (S5JDPCT8RC)}"
MODEL_SOURCE_BASE_URL="${MODEL_SOURCE_BASE_URL:-}"
MODEL_SOURCE_CONFIG_FILENAME="mererun-model-source-base-url.txt"
SKILL_SOURCE_DIR="${SKILL_SOURCE_DIR:-}"

sign_code_asset() {
  local path="$1"
  shift || true
  echo "[make_dmg] signing: $path"
  codesign --force --timestamp --sign "$SIGN_IDENTITY" "$@" "$path"
  codesign -v "$path"
}

if [[ ! -x "$BIN_PATH" ]]; then
  echo "[make_dmg] binary not found at: $BIN_PATH" >&2
  echo "[make_dmg] run scripts/build_release.sh first." >&2
  exit 1
fi

if [[ -z "$SKILL_SOURCE_DIR" ]]; then
  if [[ -d "$ROOT_DIR/.agents/skills/mere-run" ]]; then
    SKILL_SOURCE_DIR="$ROOT_DIR/.agents/skills/mere-run"
  else
    SKILL_SOURCE_DIR="$ROOT_DIR/.claude/skills/mere-run"
  fi
fi

if [[ ! -d "$SKILL_SOURCE_DIR" ]]; then
  echo "[make_dmg] skill directory not found at: $SKILL_SOURCE_DIR" >&2
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

# SwiftPM release products can depend on colocated frameworks and resource bundles.
BIN_DIR="$(dirname "$BIN_PATH")"
shopt -s nullglob
runtime_assets=("$BIN_DIR"/*.framework "$BIN_DIR"/*.bundle)
shopt -u nullglob
for asset in "${runtime_assets[@]}"; do
  ditto "$asset" "$STAGING_DIR/$(basename "$asset")"
done

if [[ -n "$MODEL_SOURCE_BASE_URL" ]]; then
  printf '%s\n' "$MODEL_SOURCE_BASE_URL" > "$STAGING_DIR/$MODEL_SOURCE_CONFIG_FILENAME"
fi

for asset in "$STAGING_DIR"/*.framework; do
  [[ -e "$asset" ]] || continue
  sign_code_asset "$asset"
done
sign_code_asset "$STAGING_DIR/mere.run" --options runtime

# Skill
mkdir -p "$STAGING_DIR/skills/mere-run"
cp -R "$SKILL_SOURCE_DIR/" "$STAGING_DIR/skills/mere-run/"

# Installer
cp "$ROOT_DIR/scripts/install.sh" "$STAGING_DIR/install.sh"
chmod +x "$STAGING_DIR/install.sh"

# Public notices
cp "$ROOT_DIR/THIRD_PARTY_NOTICES.md" "$STAGING_DIR/THIRD_PARTY_NOTICES.md"

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
