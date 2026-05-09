#!/usr/bin/env bash
set -euo pipefail

# mere.run installer — ships inside the DMG.
# Copies the binary and packaged runtime assets to /usr/local/bin.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${MERERUN_INSTALL_SOURCE_DIR:-$SCRIPT_DIR}"
if [[ ! -x "$SOURCE_DIR/mere.run" && -x "$SCRIPT_DIR/CLI/mere.run" ]]; then
  SOURCE_DIR="$SCRIPT_DIR/CLI"
fi

BIN_SRC="$SOURCE_DIR/mere.run"
BIN_DEST="${MERERUN_INSTALL_BIN_DEST:-/usr/local/bin/mere.run}"
BIN_DEST_DIR="$(dirname "$BIN_DEST")"

usage() {
  cat <<'USAGE'
Usage: ./install.sh [--help]

Install the packaged mere.run CLI and colocated runtime assets.

Environment:
  MERERUN_INSTALL_BIN_DEST     Override install path (default: /usr/local/bin/mere.run)
  MERERUN_INSTALL_SOURCE_DIR   Override packaged CLI directory (default: ./CLI if present)
USAGE
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "[mere.run] error: unknown argument: $1" >&2
    usage >&2
    exit 64
    ;;
esac

if [[ ! -x "$BIN_SRC" ]]; then
  echo "[mere.run] error: binary not found at $BIN_SRC" >&2
  exit 1
fi

echo "[mere.run] installing mere.run..."
echo "[mere.run] source: $SOURCE_DIR"

copy_path() {
  local src="$1"
  local dest_dir="$2"
  local base
  base="$(basename "$src")"

  mkdir -p "$dest_dir"
  rm -rf "$dest_dir/$base"
  ditto "$src" "$dest_dir/$base"
}

can_install_without_sudo=false
if mkdir -p "$BIN_DEST_DIR" 2>/dev/null && [[ -w "$BIN_DEST_DIR" ]]; then
  can_install_without_sudo=true
fi

# Install binary
if [[ "$can_install_without_sudo" == true ]]; then
  copy_path "$BIN_SRC" "$BIN_DEST_DIR"
  chmod +x "$BIN_DEST"
else
  echo "[mere.run] need sudo to write to $BIN_DEST_DIR"
  sudo mkdir -p "$BIN_DEST_DIR"
  sudo rm -f "$BIN_DEST"
  sudo ditto "$BIN_SRC" "$BIN_DEST"
  sudo chmod +x "$BIN_DEST"
fi

# Install colocated runtime assets needed by the CLI.
shopt -s nullglob
support_items=("$SOURCE_DIR"/*.framework "$SOURCE_DIR"/*.bundle)
shopt -u nullglob

if (( ${#support_items[@]} > 0 )); then
  echo "[mere.run] installing runtime assets..."
  if [[ "$can_install_without_sudo" == true ]]; then
    for item in "${support_items[@]}"; do
      copy_path "$item" "$BIN_DEST_DIR"
    done
  else
    for item in "${support_items[@]}"; do
      base="$(basename "$item")"
      sudo rm -rf "$BIN_DEST_DIR/$base"
      sudo ditto "$item" "$BIN_DEST_DIR/$base"
    done
  fi
fi

# Install MLX Metal shader resources alongside the binary.
# mlx-swift looks for metallib files in a Resources/ directory next to the executable.
MLX_BUNDLE="$BIN_DEST_DIR/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [[ -f "$MLX_BUNDLE" ]]; then
  echo "[mere.run] installing MLX Metal shaders..."
  RESOURCES_DIR="$BIN_DEST_DIR/Resources"
  if [[ "$can_install_without_sudo" == true ]]; then
    mkdir -p "$RESOURCES_DIR"
    cp -f "$MLX_BUNDLE" "$RESOURCES_DIR/default.metallib"
    cp -f "$MLX_BUNDLE" "$RESOURCES_DIR/mlx.metallib"
    cp -f "$MLX_BUNDLE" "$BIN_DEST_DIR/mlx.metallib"
  else
    sudo mkdir -p "$RESOURCES_DIR"
    sudo cp -f "$MLX_BUNDLE" "$RESOURCES_DIR/default.metallib"
    sudo cp -f "$MLX_BUNDLE" "$RESOURCES_DIR/mlx.metallib"
    sudo cp -f "$MLX_BUNDLE" "$BIN_DEST_DIR/mlx.metallib"
  fi
fi

echo ""
echo "[mere.run] installed:"
echo "  binary: $BIN_DEST"
if (( ${#support_items[@]} > 0 )); then
  echo "  runtime assets:"
  for item in "${support_items[@]}"; do
      echo "    $BIN_DEST_DIR/$(basename "$item")"
  done
fi
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
