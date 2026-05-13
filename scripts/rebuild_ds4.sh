#!/usr/bin/env bash
set -euo pipefail

# Rebuilds the ds4 (DeepSeek V4 Flash) inference binaries vendored under
# vendor/ds4/. Mirrors rebuild_llama_xcframework.sh: clones the upstream repo at
# a pinned commit, runs `make`, and copies the resulting executables into the
# repo. The vendored binaries are checked in so end users do not need a C
# toolchain at install time.
#
# Usage:
#   scripts/rebuild_ds4.sh
#
# Environment overrides:
#   DS4_COMMIT        Override the pinned upstream commit.
#   DS4_URL           Override the upstream git URL.
#   DS4_LOCAL_SOURCE  If set, build from this local checkout instead of cloning.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor/ds4"
DS4_COMMIT="${DS4_COMMIT:-f8b4ed635d559b3a5b44bf2df6a77e21b3e9178f}"
DS4_URL="${DS4_URL:-https://github.com/antirez/ds4.git}"
DS4_LOCAL_SOURCE="${DS4_LOCAL_SOURCE:-}"

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool not found in PATH: $tool" >&2
    exit 1
  fi
}

require_tool make
require_tool cc

UNAME_S="$(uname -s)"
if [[ "$UNAME_S" != "Darwin" ]]; then
  echo "[ds4] note: this script targets macOS Metal builds; on $UNAME_S you" >&2
  echo "[ds4] will get a CUDA or CPU build depending on the host." >&2
fi

tmpdir=""
cleanup() {
  if [[ -n "$tmpdir" && -d "$tmpdir" ]]; then
    rm -rf "$tmpdir"
  fi
}
trap cleanup EXIT

if [[ -n "$DS4_LOCAL_SOURCE" ]]; then
  if [[ ! -d "$DS4_LOCAL_SOURCE" ]]; then
    echo "[ds4] error: DS4_LOCAL_SOURCE not a directory: $DS4_LOCAL_SOURCE" >&2
    exit 1
  fi
  source_dir="$DS4_LOCAL_SOURCE"
  echo "[ds4] building from local source: $source_dir"
  resolved_commit="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || echo "unknown")"
else
  require_tool git
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ds4-vendor.XXXXXX")"
  source_dir="${tmpdir}/ds4"
  echo "[ds4] cloning ${DS4_URL} into ${tmpdir}"
  git clone --filter=blob:none "${DS4_URL}" "${source_dir}"
  git -C "${source_dir}" checkout "${DS4_COMMIT}"
  resolved_commit="${DS4_COMMIT}"
fi

echo "[ds4] building at commit ${resolved_commit}"
(
  cd "${source_dir}"
  make -j ds4 ds4-server ds4-bench
)

echo "[ds4] replacing ${VENDOR_DIR}"
rm -rf "${VENDOR_DIR}"
mkdir -p "${VENDOR_DIR}"
cp "${source_dir}/ds4" "${VENDOR_DIR}/ds4"
cp "${source_dir}/ds4-server" "${VENDOR_DIR}/ds4-server"
cp "${source_dir}/ds4-bench" "${VENDOR_DIR}/ds4-bench"
cp "${source_dir}/LICENSE" "${VENDOR_DIR}/LICENSE"
chmod +x "${VENDOR_DIR}/ds4" "${VENDOR_DIR}/ds4-server" "${VENDOR_DIR}/ds4-bench"
printf "%s\n" "${resolved_commit}" > "${VENDOR_DIR}/VERSION"

# Metal shader sources are loaded at runtime relative to the process cwd, so we
# ship the metal/ directory next to the binary. mere.run launches ds4-server
# with currentDirectoryURL = vendor/ds4 so these are picked up automatically.
if [[ -d "${source_dir}/metal" ]]; then
  cp -R "${source_dir}/metal" "${VENDOR_DIR}/metal"
fi

cat > "${VENDOR_DIR}/README.md" <<EOF
# vendor/ds4

Prebuilt DeepSeek V4 Flash inference binaries vendored from
[ds4](${DS4_URL}) at commit \`${resolved_commit}\`.

Rebuild with:

    scripts/rebuild_ds4.sh

Binaries:
- \`ds4\`         interactive CLI (not used by mere.run runtime; included for parity)
- \`ds4-server\`  OpenAI-compatible HTTP server (spawned by MereRunCore)
- \`ds4-bench\`  frontier throughput benchmark

The 86 GB GGUF model is **not** vendored. mere.run lazy-downloads it from
\`antirez/deepseek-v4-gguf\` on Hugging Face the first time the premier agent
tier is used on a 96 GB+ Apple Silicon Mac.
EOF

echo "[ds4] vendored binaries:"
ls -la "${VENDOR_DIR}"
echo "[ds4] done"
