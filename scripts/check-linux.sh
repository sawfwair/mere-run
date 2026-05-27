#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-linux.sh [--manifest-only] [--help]

Validate the Linux package/native plumbing for the mere.run CLI.

Options:
  --manifest-only  Only validate the Linux Package.swift view.
  -h, --help       Show this help.

Environment:
  MERERUN_CHECK_LINUX_ALLOW_NON_LINUX=1   Allow --manifest-only from macOS.
  MERERUN_CHECK_LINUX_MANIFEST_ONLY=1     Same as --manifest-only.
USAGE
}

manifest_only="${MERERUN_CHECK_LINUX_MANIFEST_ONLY:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-only)
      manifest_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[check-linux] error: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# shellcheck source=scripts/linux-arm64-bf16-toolchain.sh
source "$repo_root/scripts/linux-arm64-bf16-toolchain.sh"

host_os="$(uname -s)"
if [[ "$host_os" != "Linux" ]]; then
  if [[ "${MERERUN_CHECK_LINUX_ALLOW_NON_LINUX:-0}" != "1" ]]; then
    echo "[check-linux] error: this check must run on Linux." >&2
    echo "[check-linux] set MERERUN_CHECK_LINUX_ALLOW_NON_LINUX=1 with --manifest-only for a macOS manifest smoke." >&2
    exit 64
  fi
  manifest_only=1
fi

missing_tools=()
for required_tool in swift rg; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    missing_tools+=("$required_tool")
  fi
done

if (( ${#missing_tools[@]} > 0 )); then
  printf '[check-linux] missing tools: %s\n' "${missing_tools[*]}" >&2
  exit 127
fi

package_json="$(mktemp "${TMPDIR:-/tmp}/mere-run-linux-package.XXXXXX")"
cleanup() {
  rm -f "$package_json"
}
trap cleanup EXIT

check_linux_package_view() {
  MERERUN_PACKAGE_PLATFORM=linux swift package dump-package >"$package_json"

  if rg -q '"name" : "mere.run.app"|vendor/llama\.xcframework|"linkedFramework"' "$package_json"; then
    echo "[check-linux] Linux package view still exposes the app, xcframework llama, or framework links." >&2
    exit 1
  fi
}

if [[ "$manifest_only" == "1" ]]; then
  check_linux_package_view
  bash ./scripts/prepare-linux-native.sh --check
  echo "[check-linux] manifest-only Linux package smoke passed."
  exit 0
fi

missing_tools=()
for required_tool in cc ffmpeg ffprobe gzip pkg-config timeout unzip zip; do
  if ! command -v "$required_tool" >/dev/null 2>&1; then
    missing_tools+=("$required_tool")
  fi
done
if ! printf '#include <cblas.h>\n' | cc -E -xc - >/dev/null 2>&1; then
  missing_tools+=("cblas.h (install libopenblas-dev)")
fi
if ! printf '#include <lapack.h>\n' | cc -E -xc - >/dev/null 2>&1; then
  missing_tools+=("lapack.h (install liblapacke-dev)")
fi
if [[ "$(cc -print-file-name=libgfortran.so)" == "libgfortran.so" ]]; then
  missing_tools+=("libgfortran.so (install gfortran)")
fi

if (( ${#missing_tools[@]} > 0 )); then
  printf '[check-linux] missing tools: %s\n' "${missing_tools[*]}" >&2
  exit 127
fi

case "$(uname -m)" in
  x86_64|amd64)
    arch="x86_64"
    ;;
  aarch64|arm64)
    arch="arm64"
    ;;
  *)
    echo "[check-linux] error: unsupported Linux architecture: $(uname -m)" >&2
    exit 65
    ;;
esac
configure_linux_arm64_bf16_toolchain "$arch" "check-linux"

bash ./scripts/prepare-linux-native.sh
native_root="$repo_root/.build/native/linux-$arch"
export PKG_CONFIG_PATH="$native_root/pkgconfig:${PKG_CONFIG_PATH:-}"
export LIBRARY_PATH="$native_root/llama/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$native_root/llama/lib:${LD_LIBRARY_PATH:-}"

check_linux_package_view
swift build --disable-index-store --product mere.run
swift build --disable-index-store --product MediaIOSmoke
timeout 120s swift run --disable-index-store --skip-build MediaIOSmoke

run_installer_fixture() {
  local fixture_root
  local source_dir
  local dest_dir
  fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-linux-install.XXXXXX")"
  source_dir="$fixture_root/source"
  dest_dir="$fixture_root/dest"
  mkdir -p "$source_dir" "$dest_dir"
  trap "rm -rf '$fixture_root'; cleanup" EXIT

  cp scripts/install.sh "$source_dir/install.sh"
  chmod +x "$source_dir/install.sh"
  cat >"$source_dir/mere.run" <<'EOF'
#!/usr/bin/env bash
echo "mere.run fixture"
EOF
  chmod +x "$source_dir/mere.run"
  printf 'fake shared library\n' >"$source_dir/libmere_native.so"

  MERERUN_INSTALL_BIN_DEST="$dest_dir/mere.run" \
    MERERUN_INSTALL_PLATFORM=Linux \
    MERERUN_INSTALL_DISABLE_DITTO=1 \
    bash "$source_dir/install.sh" >/dev/null

  [[ -x "$dest_dir/mere.run" ]]
  [[ -f "$dest_dir/libmere_native.so" ]]
}

run_installer_fixture

mere_run_bin=".build/debug/mere.run"
if [[ ! -x "$mere_run_bin" ]]; then
  echo "[check-linux] expected built executable at $mere_run_bin." >&2
  exit 1
fi

"$mere_run_bin" --help >/dev/null
"$mere_run_bin" guide --help >/dev/null
"$mere_run_bin" image generate --help >/dev/null
"$mere_run_bin" image validate --help >/dev/null
"$mere_run_bin" text chat --help >/dev/null
"$mere_run_bin" text code --help >/dev/null
"$mere_run_bin" text embed --help >/dev/null
"$mere_run_bin" speech synthesize --help >/dev/null
"$mere_run_bin" speech transcribe --help >/dev/null
"$mere_run_bin" speech profile --help >/dev/null
"$mere_run_bin" vision inspect --help >/dev/null
"$mere_run_bin" vision ocr --help >/dev/null
"$mere_run_bin" music generate --help >/dev/null
"$mere_run_bin" video generate --help >/dev/null
"$mere_run_bin" video export-latents --help >/dev/null
"$mere_run_bin" model --help >/dev/null
"$mere_run_bin" status --help >/dev/null
"$mere_run_bin" api serve --help >/dev/null

model_list_output="$("$mere_run_bin" model list)"
rg -q '^ID +Category +Status +Size$' <<<"$model_list_output"
rg -q '^image-klein-max +image +' <<<"$model_list_output"

status_output="$("$mere_run_bin" status --timeout-seconds 0.1)"
rg -q '^mere\.run status$' <<<"$status_output"
rg -q '^  server: ' <<<"$status_output"
rg -q '^  model store: ' <<<"$status_output"
rg -q '^  installed models: ' <<<"$status_output"
