#!/usr/bin/env bash
set -euo pipefail

# Build mlx-swift's AOT Metal kernel library (default.metallib) from the
# CURRENT dependency checkout and stamp it with the versions it was built from.
#
# Why this exists: plain `swift build` never compiles the .metal sources —
# there is no metallib rule in the SwiftPM manifest — yet the MLX runtime
# hard-requires the library and loads whatever file it finds with no version
# validation. A metallib left over from an older mlx-swift silently corrupts
# inference: on 2026-07-03 a pre-0.30 metallib paired with the mlx 0.31.1
# host dispatch produced gibberish, nondeterministic decode for every MLX
# text model once the KV length crossed 1024 (2-pass SDPA kernel ABI change).
#
# The produced library is accompanied by a `default.metallib.version` sidecar
# recording the mlx core version, the mlx-swift pin, and a hash of the kernel
# sources. `mere.run` validates the sidecar at startup (MLXBundleSupport) and
# refuses to run against a mismatched library.
#
# Usage:
#   scripts/build_mlx_metallib.sh                       Build and install into
#                                                       .build (debug+release).
#   scripts/build_mlx_metallib.sh --configuration release
#                                                       Install into one config.
#   scripts/build_mlx_metallib.sh --output DIR          Build and write the
#                                                       metallib + sidecar pair
#                                                       into DIR only.
#   scripts/build_mlx_metallib.sh --verify-only PATH    Compare an existing
#                                                       bundle/sidecar against
#                                                       the current checkout;
#                                                       exit 1 if stale.
#
# Environment:
#   MERERUN_MLX_SWIFT_CHECKOUT  Override the mlx-swift checkout location
#                               (default: .build/checkouts/mlx-swift).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="${MERERUN_MLX_SWIFT_CHECKOUT:-$repo_root/.build/checkouts/mlx-swift}"
gen_dir="$checkout/Source/Cmlx/mlx-generated/metal"
version_header="$checkout/Source/Cmlx/mlx/mlx/version.h"
resolved="$repo_root/Package.resolved"
stamp_name="default.metallib.version"

usage() {
  sed -n '3,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

mode="install"
output_dir=""
verify_target=""
configurations=(debug release)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || { echo "error: --output needs a directory" >&2; exit 64; }
      mode="output"; output_dir="$2"; shift 2 ;;
    --configuration)
      [[ $# -ge 2 ]] || { echo "error: --configuration needs debug|release|all" >&2; exit 64; }
      case "$2" in
        debug|release) configurations=("$2") ;;
        all) configurations=(debug release) ;;
        *) echo "error: unknown configuration: $2" >&2; exit 64 ;;
      esac
      shift 2 ;;
    --verify-only)
      [[ $# -ge 2 ]] || { echo "error: --verify-only needs a bundle/sidecar path" >&2; exit 64; }
      mode="verify"; verify_target="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ ! -d "$gen_dir" ]]; then
  cat >&2 <<EOF
error: mlx-swift checkout not found at:
  $checkout

Resolve package dependencies first:
  swift package resolve

(or point MERERUN_MLX_SWIFT_CHECKOUT at an mlx-swift checkout)
EOF
  exit 1
fi

# --- Stamp inputs -----------------------------------------------------------

core_version="$(awk '
  /#define MLX_VERSION_MAJOR/ { maj = $3 }
  /#define MLX_VERSION_MINOR/ { min = $3 }
  /#define MLX_VERSION_PATCH/ { pat = $3 }
  END { if (maj != "" && min != "" && pat != "") printf "%s.%s.%s", maj, min, pat }
' "$version_header")"
if [[ -z "$core_version" ]]; then
  echo "error: could not parse MLX version from $version_header" >&2
  exit 1
fi

read -r swift_pin_version swift_pin_revision < <(python3 - "$resolved" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    doc = json.load(f)
pins = doc.get("pins") or doc.get("object", {}).get("pins", [])
for pin in pins:
    identity = (pin.get("identity") or pin.get("package", "")).lower()
    if identity == "mlx-swift":
        state = pin.get("state", {})
        print(state.get("version", "unknown"), state.get("revision", "unknown"))
        break
else:
    print("unknown", "unknown")
EOF
)

sources_hash="$(cd "$gen_dir" && find . -type f \( -name '*.metal' -o -name '*.h' \) -print0 \
  | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -d' ' -f1)"

# --- Verify mode ------------------------------------------------------------

find_sidecar() {
  local target="$1"
  local candidates=(
    "$target"
    "$target/$stamp_name"
    "$target/Contents/Resources/$stamp_name"
    "$target/Resources/$stamp_name"
  )
  for c in "${candidates[@]}"; do
    if [[ -f "$c" && "$(basename "$c")" == "$stamp_name" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

stamp_field() {
  awk -F': ' -v key="$1" '$1 == key { print $2; exit }' "$2"
}

if [[ "$mode" == "verify" ]]; then
  if ! sidecar="$(find_sidecar "$verify_target")"; then
    echo "STALE: no $stamp_name found under $verify_target" >&2
    echo "       (unstamped metallib — provenance unknown; rebuild with scripts/build_mlx_metallib.sh)" >&2
    exit 1
  fi
  stamped_core="$(stamp_field "mlx-core-version" "$sidecar")"
  stamped_rev="$(stamp_field "mlx-swift-revision" "$sidecar")"
  stamped_hash="$(stamp_field "kernel-sources-sha256" "$sidecar")"
  status=0
  [[ "$stamped_core" == "$core_version" ]] || { echo "STALE: mlx core version $stamped_core != checkout $core_version" >&2; status=1; }
  [[ "$stamped_rev" == "$swift_pin_revision" ]] || { echo "STALE: mlx-swift revision $stamped_rev != pinned $swift_pin_revision" >&2; status=1; }
  [[ "$stamped_hash" == "$sources_hash" ]] || { echo "STALE: kernel source hash mismatch" >&2; status=1; }
  if [[ $status -eq 0 ]]; then
    echo "OK: $sidecar matches mlx-swift $swift_pin_version (mlx core $core_version)"
  fi
  exit $status
fi

# --- Build ------------------------------------------------------------------

workdir="$(mktemp -d -t mlx-metallib)"
trap 'rm -rf "$workdir"' EXIT

metal_sources=("$gen_dir"/*.metal)
if [[ ${#metal_sources[@]} -eq 0 ]]; then
  echo "error: no .metal sources in $gen_dir" >&2
  exit 1
fi

echo "[metallib] compiling ${#metal_sources[@]} kernels from mlx-swift $swift_pin_version (mlx core $core_version)"
# -fno-fast-math is mandatory: Metal defaults to fast math, and mlx's kernels
# require IEEE semantics (mlx's own CMake passes the same flag).
for src in "${metal_sources[@]}"; do
  base="$(basename "$src" .metal)"
  xcrun -sdk macosx metal \
    -x metal -Wall -Wextra -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions \
    -c "$src" -I "$gen_dir" -o "$workdir/$base.air" \
    2> "$workdir/$base.err" &
done
wait

failed=0
for src in "${metal_sources[@]}"; do
  base="$(basename "$src" .metal)"
  if [[ ! -f "$workdir/$base.air" ]]; then
    echo "error: metal compile failed for $base.metal:" >&2
    cat "$workdir/$base.err" >&2
    failed=1
  fi
done
[[ $failed -eq 0 ]] || exit 1

xcrun -sdk macosx metallib "$workdir"/*.air -o "$workdir/default.metallib"

metal_compiler="$(xcrun -sdk macosx metal --version 2>/dev/null | head -1 || echo unknown)"
cat > "$workdir/$stamp_name" <<EOF
mlx-core-version: $core_version
mlx-swift-version: $swift_pin_version
mlx-swift-revision: $swift_pin_revision
kernel-sources-sha256: $sources_hash
built-at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
metal-compiler: $metal_compiler
EOF

install_pair() {
  local dest_dir="$1"
  mkdir -p "$dest_dir"
  cp -f "$workdir/default.metallib" "$dest_dir/default.metallib"
  cp -f "$workdir/$stamp_name" "$dest_dir/$stamp_name"
}

# A bundle created from scratch here (plain `swift build` emits none on CI)
# needs a valid Contents/Info.plist or codesign rejects the enclosing app
# with "bundle format unrecognized, invalid, or unsuitable".
ensure_bundle_info_plist() {
  local bundle_root="$1"
  local plist="$bundle_root/Contents/Info.plist"
  [[ -f "$plist" ]] && return 0
  mkdir -p "$bundle_root/Contents"
  cat > "$plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleIdentifier</key>
	<string>mlx-swift.Cmlx.resources</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>mlx-swift_Cmlx</string>
	<key>CFBundlePackageType</key>
	<string>BNDL</string>
</dict>
</plist>
EOF
}

if [[ "$mode" == "output" ]]; then
  install_pair "$output_dir"
  echo "[metallib] wrote $output_dir/default.metallib (+ $stamp_name)"
  exit 0
fi

# Install into the SwiftPM build tree: the canonical bundle location, plus the
# flat compatibility locations the MLX loader (and MLXTestSupport) probe.
for config in "${configurations[@]}"; do
  build_root="$repo_root/.build/arm64-apple-macosx/$config"
  install_pair "$build_root/mlx-swift_Cmlx.bundle/Contents/Resources"
  ensure_bundle_info_plist "$build_root/mlx-swift_Cmlx.bundle"
  install_pair "$build_root/Resources"
  cp -f "$workdir/default.metallib" "$build_root/Resources/mlx.metallib"
  cp -f "$workdir/default.metallib" "$build_root/mlx.metallib"
  echo "[metallib] installed into .build/arm64-apple-macosx/$config"
done

# The repo vendors a copy (tracked in git) so plain `swift build` users get
# working Metal shaders without the full Xcode metal toolchain. It is the
# FIRST location the bundle/test resolvers consult, so it must be regenerated
# on every mlx-swift bump — commit the refreshed pair when it changes.
vendor_bundle="$repo_root/vendor/mlx-swift_Cmlx.bundle"
if [[ -d "$vendor_bundle" ]]; then
  # The vendor pair is tracked in git; skip the refresh when the existing
  # stamp already matches this exact build (same sources, same pin) so
  # routine script runs don't churn the tracked sidecar's built-at line.
  vendor_stamp="$vendor_bundle/Contents/Resources/$stamp_name"
  if [[ -f "$vendor_stamp" ]] \
    && [[ "$(stamp_field "kernel-sources-sha256" "$vendor_stamp")" == "$sources_hash" ]] \
    && [[ "$(stamp_field "mlx-swift-revision" "$vendor_stamp")" == "$swift_pin_revision" ]] \
    && [[ "$(stamp_field "mlx-core-version" "$vendor_stamp")" == "$core_version" ]]; then
    echo "[metallib] vendored $vendor_bundle already current; left untouched"
  else
    install_pair "$vendor_bundle/Contents/Resources"
    ensure_bundle_info_plist "$vendor_bundle"
    echo "[metallib] refreshed vendored $vendor_bundle (tracked in git — commit if changed)"
  fi
fi

echo "[metallib] stamp: mlx core $core_version, mlx-swift $swift_pin_version ($swift_pin_revision)"
