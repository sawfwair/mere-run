#!/usr/bin/env bash
set -euo pipefail

# Build and sign mere.run release binary.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIGN_IDENTITY="${SIGN_IDENTITY:-Developer ID Application: Kyle McCullough (S5JDPCT8RC)}"

sign_code_asset() {
  local path="$1"
  echo "[build_release] signing: $path"
  codesign --force --timestamp --sign "$SIGN_IDENTITY" "$path"
  codesign -v "$path"
}

echo "[build_release] building release binary..."
swift build -c release --product mere.run --package-path "$ROOT_DIR"

BIN_DIR="$(swift build -c release --product mere.run --show-bin-path --package-path "$ROOT_DIR")"
BIN_PATH="$BIN_DIR/mere.run"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "[build_release] error: binary not found at $BIN_PATH" >&2
  exit 1
fi

echo "[build_release] signing with: $SIGN_IDENTITY"

shopt -s nullglob
runtime_assets=("$BIN_DIR"/*.framework)
shopt -u nullglob
for asset in "${runtime_assets[@]}"; do
  sign_code_asset "$asset"
done

codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$BIN_PATH"
codesign -v "$BIN_PATH"

echo "[build_release] done: $BIN_PATH"
echo "$BIN_PATH"
