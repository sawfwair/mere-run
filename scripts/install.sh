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
  MERERUN_INSTALL_SKIP_VERIFY  Skip running the installed binary when set to 1
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
install_platform="${MERERUN_INSTALL_PLATFORM:-$(uname -s)}"

copy_path() {
  local src="$1"
  local dest_dir="$2"
  local base
  base="$(basename "$src")"
  local dest="$dest_dir/$base"

  mkdir -p "$dest_dir"
  rm -rf "$dest"
  if [[ "${MERERUN_INSTALL_DISABLE_DITTO:-0}" != "1" ]] && command -v ditto >/dev/null 2>&1; then
    ditto "$src" "$dest"
  else
    cp -a "$src" "$dest"
  fi
}

copy_path_sudo() {
  local src="$1"
  local dest_dir="$2"
  local base
  base="$(basename "$src")"
  local dest="$dest_dir/$base"

  sudo mkdir -p "$dest_dir"
  sudo rm -rf "$dest"
  if [[ "${MERERUN_INSTALL_DISABLE_DITTO:-0}" != "1" ]] && command -v ditto >/dev/null 2>&1; then
    sudo ditto "$src" "$dest"
  else
    sudo cp -a "$src" "$dest"
  fi
}

copy_to_path() {
  local src="$1"
  local dest="$2"
  local dest_dir
  dest_dir="$(dirname "$dest")"

  mkdir -p "$dest_dir"
  rm -rf "$dest"
  if [[ "${MERERUN_INSTALL_DISABLE_DITTO:-0}" != "1" ]] && command -v ditto >/dev/null 2>&1; then
    ditto "$src" "$dest"
  else
    cp -a "$src" "$dest"
  fi
}

copy_to_path_sudo() {
  local src="$1"
  local dest="$2"
  local dest_dir
  dest_dir="$(dirname "$dest")"

  sudo mkdir -p "$dest_dir"
  sudo rm -rf "$dest"
  if [[ "${MERERUN_INSTALL_DISABLE_DITTO:-0}" != "1" ]] && command -v ditto >/dev/null 2>&1; then
    sudo ditto "$src" "$dest"
  else
    sudo cp -a "$src" "$dest"
  fi
}

can_install_without_sudo=false
if mkdir -p "$BIN_DEST_DIR" 2>/dev/null && [[ -w "$BIN_DEST_DIR" ]]; then
  can_install_without_sudo=true
fi

# Install binary
if [[ "$can_install_without_sudo" == true ]]; then
  copy_to_path "$BIN_SRC" "$BIN_DEST"
  chmod +x "$BIN_DEST"
else
  echo "[mere.run] need sudo to write to $BIN_DEST_DIR"
  copy_to_path_sudo "$BIN_SRC" "$BIN_DEST"
  sudo chmod +x "$BIN_DEST"
fi

# Install colocated runtime assets needed by the CLI.
support_items=()
if [[ "$install_platform" == "Darwin" ]]; then
  for item in \
    "$SOURCE_DIR"/*.framework \
    "$SOURCE_DIR"/*.bundle \
    "$SOURCE_DIR"/*.dylib
  do
    if [[ -e "$item" ]]; then
      support_items+=("$item")
    fi
  done
else
  for item in \
    "$SOURCE_DIR"/*.so \
    "$SOURCE_DIR"/*.so.* \
    "$SOURCE_DIR"/lib
  do
    if [[ -e "$item" ]]; then
      support_items+=("$item")
    fi
  done
fi

if (( ${#support_items[@]} > 0 )); then
  echo "[mere.run] installing runtime assets..."
  if [[ "$can_install_without_sudo" == true ]]; then
    for item in "${support_items[@]}"; do
      copy_path "$item" "$BIN_DEST_DIR"
    done
  else
    for item in "${support_items[@]}"; do
      copy_path_sudo "$item" "$BIN_DEST_DIR"
    done
  fi
fi

# Install the bundled DeepSeek V4 Flash inference binaries.
# The premier agent tier (96 GB+ Macs) spawns vendor/ds4/ds4-server as a
# subprocess. The CLI looks for it next to itself at runtime.
DS4_SRC="$SOURCE_DIR/vendor/ds4"
if [[ ! -d "$DS4_SRC" ]]; then
  DS4_SRC="$SCRIPT_DIR/vendor/ds4"
fi
if [[ -d "$DS4_SRC" ]]; then
  echo "[mere.run] installing DS4 inference binaries..."
  if [[ "$can_install_without_sudo" == true ]]; then
    copy_path "$DS4_SRC" "$BIN_DEST_DIR/vendor"
  else
    copy_path_sudo "$DS4_SRC" "$BIN_DEST_DIR/vendor"
  fi
fi

# Install MLX Metal shader resources alongside the binary.
# mlx-swift looks for metallib files in a Resources/ directory next to the executable.
MLX_BUNDLE="$BIN_DEST_DIR/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [[ "$install_platform" == "Darwin" && -f "$MLX_BUNDLE" ]]; then
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
if [[ "${MERERUN_INSTALL_SKIP_VERIFY:-0}" == "1" ]]; then
  echo "[mere.run] verification skipped."
elif [[ -x "$BIN_DEST" ]]; then
  echo "[mere.run] verification:"
  "$BIN_DEST" --help | head -3
  echo ""
  echo "[mere.run] ready. Run 'mere.run --help' to get started."
else
  echo "[mere.run] binary installed but not in PATH."
  echo "  Add /usr/local/bin to your PATH, or run directly: $BIN_DEST --help"
fi
