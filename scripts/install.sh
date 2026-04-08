#!/usr/bin/env bash
set -euo pipefail

# mere.run installer — ships inside the DMG.
# Copies the binary to /usr/local/bin and the Claude Code skill to ~/.claude/skills/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN_SRC="$SCRIPT_DIR/mere.run"
SKILL_SRC="$SCRIPT_DIR/skills/mere-run"
BIN_DEST="/usr/local/bin/mere.run"
SKILL_DEST="$HOME/.claude/skills/mere-run"

if [[ ! -x "$BIN_SRC" ]]; then
  echo "[mere.run] error: binary not found at $BIN_SRC" >&2
  exit 1
fi

if [[ ! -d "$SKILL_SRC" ]]; then
  echo "[mere.run] error: skill not found at $SKILL_SRC" >&2
  exit 1
fi

echo "[mere.run] installing mere.run..."

# Install binary
if [[ -w "$(dirname "$BIN_DEST")" ]] 2>/dev/null; then
  mkdir -p "$(dirname "$BIN_DEST")"
  cp "$BIN_SRC" "$BIN_DEST"
else
  echo "[mere.run] need sudo to write to $(dirname "$BIN_DEST")"
  sudo mkdir -p "$(dirname "$BIN_DEST")"
  sudo cp "$BIN_SRC" "$BIN_DEST"
  sudo chmod +x "$BIN_DEST"
fi

# Install Claude Code skill
mkdir -p "$SKILL_DEST"
cp -R "$SKILL_SRC/" "$SKILL_DEST/"

echo ""
echo "[mere.run] installed:"
echo "  binary: $BIN_DEST"
echo "  skill:  $SKILL_DEST"
echo ""

# Verify
if command -v mere.run >/dev/null 2>&1; then
  echo "[mere.run] verification:"
  mere.run --help | head -3
  echo ""
  echo "[mere.run] ready. Run 'mere.run --help' to get started."
else
  echo "[mere.run] binary installed but not in PATH."
  echo "  Add /usr/local/bin to your PATH, or run directly: $BIN_DEST --help"
fi
