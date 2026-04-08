#!/usr/bin/env bash
set -euo pipefail

# Build and sign mere.run release binary.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIGN_IDENTITY="${SIGN_IDENTITY:-Developer ID Application: Kyle McCullough (S5JDPCT8RC)}"

echo "[build_release] building release binary..."
swift build -c release --product mere.run --package-path "$ROOT_DIR"

BIN_PATH="$(swift build -c release --product mere.run --show-bin-path --package-path "$ROOT_DIR")/mere.run"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "[build_release] error: binary not found at $BIN_PATH" >&2
  exit 1
fi

echo "[build_release] signing with: $SIGN_IDENTITY"
codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$BIN_PATH"
codesign -v "$BIN_PATH"

echo "[build_release] done: $BIN_PATH"
echo "$BIN_PATH"
