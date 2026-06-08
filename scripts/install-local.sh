#!/usr/bin/env bash
set -euo pipefail

# Build mere.run from this checkout and install it globally so the `mere.run`
# command on your $PATH points at the local source tree's latest code.
#
# Typical use:
#   scripts/install-local.sh                       # release build, /usr/local/bin
#   scripts/install-local.sh --debug               # faster build, for hacking
#   MERERUN_INSTALL_BIN_DEST=~/bin/mere.run \
#     scripts/install-local.sh                     # no-sudo prefix
#
# The script:
# 1. Runs `swift build` for the `mere.run` CLI product.
# 2. Stages the built binary + colocated runtime assets (vendor/ds4, llama
#    xcframework, MLX bundle, Resources/) into a temp directory in the layout
#    scripts/install.sh expects.
# 3. Delegates to scripts/install.sh to ditto everything into place. That keeps
#    the actual install logic (sudo handling, MLX shader copy, etc.) in one
#    well-tested place.
#
# After this script returns, `which mere.run` should point at the installed
# binary and `mere.run --help` should reflect the code in your working tree.

usage() {
  cat <<'USAGE'
Usage: scripts/install-local.sh [--debug] [--no-build] [--help]

Build mere.run from the current source checkout and install it globally.

Options:
  --debug       Build with `swift build` (default: --configuration release).
  --no-build    Skip the swift build step and reuse whatever's already at
                `.build/release/mere.run` (or .build/debug if --debug).
  -h, --help    Show this help.

Environment:
  MERERUN_INSTALL_BIN_DEST     Override install path. Default: /usr/local/bin/mere.run
                               Use ~/bin/mere.run or similar to skip sudo.
USAGE
}

configuration="release"
do_build=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      configuration="debug"
      shift
      ;;
    --no-build)
      do_build=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[install-local] error: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if ! command -v swift >/dev/null 2>&1; then
  echo "[install-local] error: swift not found in PATH. Install Xcode or the Swift toolchain." >&2
  exit 127
fi

swift_build_args=(build --product mere.run)
swift_bin_path_args=(build --show-bin-path)
if [[ "$configuration" == "release" ]]; then
  swift_build_args+=(--configuration release)
  swift_bin_path_args+=(--configuration release)
fi

if (( do_build )); then
  echo "[install-local] swift build (--configuration $configuration)"
  swift "${swift_build_args[@]}"
fi

build_dir="$(swift "${swift_bin_path_args[@]}")"
cli_executable="${build_dir}/mere.run"

if [[ ! -x "$cli_executable" ]]; then
  echo "[install-local] error: built CLI not found at $cli_executable" >&2
  echo "[install-local] run without --no-build, or build first with: swift build --configuration $configuration --product mere.run" >&2
  exit 66
fi

staging="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-install.XXXXXX")"
cleanup() {
  rm -rf "$staging"
}
trap cleanup EXIT

cp -a "$cli_executable" "$staging/mere.run"
chmod +x "$staging/mere.run"

# Colocated runtime assets that mere.run looks for next to its binary.
# Each is optional: skipped silently if not built or not present in vendor/.
stage_asset() {
  local asset="$1"
  if [[ -e "$asset" ]]; then
    cp -a "$asset" "$staging/"
  fi
}

if [[ "$(uname -s)" == "Darwin" ]]; then
  for asset in \
    "${build_dir}"/*.framework \
    "${build_dir}"/*.bundle \
    "${build_dir}/Resources" \
    "${build_dir}"/*.dylib
  do
    stage_asset "$asset"
  done
  stage_asset "${repo_root}/vendor/mlx-swift_Cmlx.bundle"
else
  native_lib_dir="${repo_root}/.build/native/linux-$(uname -m)/llama/lib"
  case "$(uname -m)" in
    x86_64|amd64)
      native_lib_dir="${repo_root}/.build/native/linux-x86_64/llama/lib"
      ;;
    aarch64|arm64)
      native_lib_dir="${repo_root}/.build/native/linux-arm64/llama/lib"
      ;;
  esac
  for asset in \
    "${build_dir}"/*.so \
    "${build_dir}"/*.so.*
  do
    stage_asset "$asset"
  done
  if [[ -d "$native_lib_dir" ]]; then
    cp -a "$native_lib_dir" "$staging/lib"
  fi
fi

# vendor/ds4 is the bundled DeepSeek V4 Flash inference engine spawned by the
# premier agent tier. install.sh expects it at vendor/ds4 under SOURCE_DIR.
if [[ -d "${repo_root}/vendor/ds4" ]]; then
  mkdir -p "$staging/vendor"
  cp -a "${repo_root}/vendor/ds4" "$staging/vendor/ds4"
fi

echo "[install-local] staged $(du -sh "$staging" | cut -f1) of payload at $staging"

# Hand off to the existing installer for the actual copy + sudo handling.
MERERUN_INSTALL_SOURCE_DIR="$staging" bash "${repo_root}/scripts/install.sh"
