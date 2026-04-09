#!/usr/bin/env bash
set -euo pipefail

# mere.run installer — ships inside the DMG.
# Copies the binary to /usr/local/bin and the Claude Code skill to ~/.claude/skills/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN_SRC="$SCRIPT_DIR/mere.run"
SKILL_SRC="$SCRIPT_DIR/skills/mere-run"
BIN_DEST="/usr/local/bin/mere.run"
SKILL_DEST="$HOME/.claude/skills/mere-run"
MODEL_SOURCE_CONFIG_FILENAME="mererun-model-source-base-url.txt"
SUPPORT_GLOBS=("$SCRIPT_DIR"/*.framework "$SCRIPT_DIR"/*.bundle "$SCRIPT_DIR"/"$MODEL_SOURCE_CONFIG_FILENAME")

if [[ ! -x "$BIN_SRC" ]]; then
  echo "[mere.run] error: binary not found at $BIN_SRC" >&2
  exit 1
fi

if [[ ! -d "$SKILL_SRC" ]]; then
  echo "[mere.run] error: skill not found at $SKILL_SRC" >&2
  exit 1
fi

echo "[mere.run] installing mere.run..."

copy_path() {
  local src="$1"
  local dest_dir="$2"
  local base
  base="$(basename "$src")"

  mkdir -p "$dest_dir"
  rm -rf "$dest_dir/$base"
  ditto "$src" "$dest_dir/$base"
}

# Install binary
if [[ -w "$(dirname "$BIN_DEST")" ]] 2>/dev/null; then
  mkdir -p "$(dirname "$BIN_DEST")"
  copy_path "$BIN_SRC" "$(dirname "$BIN_DEST")"
  chmod +x "$BIN_DEST"
else
  echo "[mere.run] need sudo to write to $(dirname "$BIN_DEST")"
  sudo mkdir -p "$(dirname "$BIN_DEST")"
  sudo rm -f "$BIN_DEST"
  sudo ditto "$BIN_SRC" "$BIN_DEST"
  sudo chmod +x "$BIN_DEST"
fi

# Install colocated runtime assets needed by the CLI.
shopt -s nullglob
support_items=("${SUPPORT_GLOBS[@]}")
shopt -u nullglob

if (( ${#support_items[@]} > 0 )); then
  echo "[mere.run] installing runtime assets..."
  if [[ -w "$(dirname "$BIN_DEST")" ]] 2>/dev/null; then
    for item in "${support_items[@]}"; do
      copy_path "$item" "$(dirname "$BIN_DEST")"
    done
  else
    for item in "${support_items[@]}"; do
      base="$(basename "$item")"
      sudo rm -rf "$(dirname "$BIN_DEST")/$base"
      sudo ditto "$item" "$(dirname "$BIN_DEST")/$base"
    done
  fi
fi

# Install Claude Code skill
mkdir -p "$SKILL_DEST"
cp -R "$SKILL_SRC/" "$SKILL_DEST/"

echo ""
echo "[mere.run] installed:"
echo "  binary: $BIN_DEST"
if (( ${#support_items[@]} > 0 )); then
  echo "  runtime assets:"
  for item in "${support_items[@]}"; do
    echo "    $(dirname "$BIN_DEST")/$(basename "$item")"
  done
fi
echo "  skill:  $SKILL_DEST"
echo ""

# Verify
if [[ -x "$BIN_DEST" ]]; then
  echo "[mere.run] verification:"
  "$BIN_DEST" --help | head -3
  echo ""
  echo "[mere.run] ready. Run 'mere.run --help' to get started."
else
  echo "[mere.run] binary installed but not in PATH."
  echo "  Add /usr/local/bin to your PATH, or run directly: $BIN_DEST --help"
fi
