#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor/llama.xcframework"
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-6d957078270f58d4ea14e8c205f5ef4e49be33f3}"
LLAMA_CPP_URL="${LLAMA_CPP_URL:-https://github.com/ggml-org/llama.cpp.git}"

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool not found in PATH: $tool" >&2
    exit 1
  fi
}

require_tool git
require_tool cmake
require_tool xcodebuild
require_tool xcrun

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/llama-xcframework.XXXXXX")"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

echo "[llama-xcframework] cloning ${LLAMA_CPP_URL} into ${tmpdir}"
git clone --filter=blob:none "${LLAMA_CPP_URL}" "${tmpdir}/llama.cpp"
git -C "${tmpdir}/llama.cpp" checkout "${LLAMA_CPP_COMMIT}"

echo "[llama-xcframework] building upstream xcframework at commit ${LLAMA_CPP_COMMIT}"
(
  cd "${tmpdir}/llama.cpp"
  ./build-xcframework.sh
)

echo "[llama-xcframework] replacing ${VENDOR_DIR}"
rm -rf "${VENDOR_DIR}"
cp -R "${tmpdir}/llama.cpp/build-apple/llama.xcframework" "${VENDOR_DIR}"
find "${VENDOR_DIR}" -type d -name dSYMs -prune -exec rm -rf {} +

echo "[llama-xcframework] done"
