#!/usr/bin/env bash
set -euo pipefail

# Build distributable Linux CLI artifacts for mere.run.
#
# Outputs:
#   dist/linux/mere-run-<version>-linux-<arch>.tar.gz
#   dist/linux/mere-run_<version>_<deb-arch>.deb   (when dpkg-deb is available)
#
# The tarball uses the same payload shape as the macOS DMG's terminal payload:
# a top-level mere.run executable, install.sh, and colocated runtime assets.
# The deb installs the payload under /usr/lib/mere-run and exposes
# /usr/bin/mere.run as a symlink so $ORIGIN/lib rpaths remain valid.

usage() {
  cat <<'USAGE'
Usage: scripts/package-linux.sh [options]

Build Linux release artifacts for the headless mere.run CLI.

Options:
  --version VERSION       Package version. Default: MERERUN_RELEASE_VERSION or git describe.
  --output-dir DIR        Artifact output directory. Default: dist/linux
  --configuration NAME    Swift configuration: release or debug. Default: release
  --skip-build            Reuse an existing .build/<configuration>/mere.run binary.
  --skip-native           Do not run scripts/prepare-linux-native.sh before building.
  --skip-deb              Do not create a .deb package.
  -h, --help              Show this help.

Environment:
  MERERUN_RELEASE_VERSION       Default package version override.
  MERERUN_BUNDLE_SWIFT_LIBS     Copy libswift*/Foundation runtime .so dependencies
                                reported by ldd into payload lib/. Default: 1.
  MERERUN_PACKAGE_LINUX_DEPS    Override Debian Depends field.
  MERERUN_LINUX_ACCEL           Passed through to prepare-linux-native.sh (cpu/cuda).
  MERERUN_DS4_LINUX_BIN_DIR     Passed through to prepare-linux-native.sh for DS4 staging.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

configuration="release"
output_dir="dist/linux"
version="${MERERUN_RELEASE_VERSION:-}"
do_build=1
do_native=1
do_deb=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      version="${2:?missing value for --version}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --configuration)
      configuration="${2:?missing value for --configuration}"
      shift 2
      ;;
    --skip-build)
      do_build=0
      shift
      ;;
    --skip-native)
      do_native=0
      shift
      ;;
    --skip-deb)
      do_deb=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[package-linux] error: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

case "$(uname -s)" in
  Linux) ;;
  *)
    echo "[package-linux] error: Linux packaging must run on Linux." >&2
    exit 65
    ;;
esac

case "$configuration" in
  release|debug) ;;
  *)
    echo "[package-linux] error: --configuration must be release or debug." >&2
    exit 64
    ;;
esac

case "$(uname -m)" in
  x86_64|amd64)
    platform_arch="x86_64"
    deb_arch="amd64"
    ;;
  aarch64|arm64)
    platform_arch="arm64"
    deb_arch="arm64"
    ;;
  *)
    echo "[package-linux] error: unsupported Linux architecture: $(uname -m)" >&2
    exit 65
    ;;
esac

if [[ -z "$version" ]]; then
  if git describe --tags --always --dirty >/dev/null 2>&1; then
    version="$(git describe --tags --always --dirty)"
  else
    version="0.0.0-local"
  fi
fi

# Debian versions cannot include arbitrary git-describe punctuation. Keep the
# human tag for tar names, and use a policy-safe version for .deb metadata.
deb_version="${version#v}"
deb_version="$(printf '%s' "$deb_version" | tr '/' '-' | sed -E 's/[^0-9A-Za-z.+:~_-]/-/g')"
if [[ -z "$deb_version" ]]; then
  deb_version="0.0.0-local"
fi

native_root="$repo_root/.build/native/linux-$platform_arch"
llama_prefix="$native_root/llama"
pkgconfig_dir="$native_root/pkgconfig"

if (( do_native )); then
  echo "[package-linux] preparing Linux native runtime artifacts"
  bash scripts/prepare-linux-native.sh
fi

export MERERUN_PACKAGE_PLATFORM="linux"
export PKG_CONFIG_PATH="$pkgconfig_dir:${PKG_CONFIG_PATH:-}"
export LIBRARY_PATH="$llama_prefix/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$llama_prefix/lib:${LD_LIBRARY_PATH:-}"

swift_build_args=(build --product mere.run)
swift_bin_path_args=(build --show-bin-path)
if [[ "$configuration" == "release" ]]; then
  swift_build_args+=(--configuration release)
  swift_bin_path_args+=(--configuration release)
fi

if (( do_build )); then
  if ! command -v swift >/dev/null 2>&1; then
    echo "[package-linux] error: swift not found in PATH." >&2
    exit 127
  fi
  echo "[package-linux] swift build --configuration $configuration"
  swift "${swift_build_args[@]}"
fi

build_dir="$(swift "${swift_bin_path_args[@]}")"
cli_executable="$build_dir/mere.run"
if [[ ! -x "$cli_executable" ]]; then
  echo "[package-linux] error: built CLI not found at $cli_executable" >&2
  exit 66
fi

rm -rf "$output_dir"
mkdir -p "$output_dir"
staging="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-linux-package.XXXXXX")"
cleanup() {
  rm -rf "$staging"
}
trap cleanup EXIT

payload_name="mere-run-${version}-linux-${platform_arch}"
payload_dir="$staging/$payload_name"
mkdir -p "$payload_dir"

cp -a "$cli_executable" "$payload_dir/mere.run"
chmod +x "$payload_dir/mere.run"
cp -a scripts/install.sh "$payload_dir/install.sh"
chmod +x "$payload_dir/install.sh"

for doc in README.md LICENSE THIRD_PARTY_NOTICES.md CHANGELOG.md; do
  if [[ -f "$doc" ]]; then
    cp -a "$doc" "$payload_dir/"
  fi
done

stage_asset() {
  local asset="$1"
  if [[ -e "$asset" ]]; then
    cp -a "$asset" "$payload_dir/"
  fi
}

for asset in \
  "$build_dir"/*.so \
  "$build_dir"/*.so.*
do
  stage_asset "$asset"
done

if [[ -d "$llama_prefix/lib" ]]; then
  cp -a "$llama_prefix/lib" "$payload_dir/lib"
fi

# Bundle Swift runtime shared libraries when the official Swift toolchain linked
# them dynamically. This lets the tarball/deb run on clean Ubuntu hosts without
# requiring users to install the Swift compiler toolchain first.
if [[ "${MERERUN_BUNDLE_SWIFT_LIBS:-1}" == "1" ]]; then
  mkdir -p "$payload_dir/lib"
  while IFS= read -r lib_path; do
    [[ -f "$lib_path" ]] || continue
    case "$(basename "$lib_path")" in
      libswift*|libFoundation*|libdispatch*|libBlocksRuntime*)
        cp -a "$lib_path" "$payload_dir/lib/"
        ;;
    esac
  done < <(ldd "$payload_dir/mere.run" 2>/dev/null | awk '/=> \/|^\// { for (i=1; i<=NF; i++) if ($i ~ /^\//) print $i }' | sort -u)
fi

if [[ -d vendor/ds4 ]]; then
  mkdir -p "$payload_dir/vendor"
  cp -a vendor/ds4 "$payload_dir/vendor/ds4"
fi

(
  cd "$staging"
  tar -czf "$repo_root/$output_dir/${payload_name}.tar.gz" "$payload_name"
)

echo "[package-linux] wrote $output_dir/${payload_name}.tar.gz"

if (( do_deb )); then
  if command -v dpkg-deb >/dev/null 2>&1; then
    deb_root="$staging/deb-root"
    install_root="$deb_root/usr/lib/mere-run"
    mkdir -p "$install_root" "$deb_root/usr/bin" "$deb_root/DEBIAN"
    cp -a "$payload_dir"/. "$install_root/"
    ln -s ../lib/mere-run/mere.run "$deb_root/usr/bin/mere.run"

    depends="${MERERUN_PACKAGE_LINUX_DEPS:-ffmpeg, libcurl4, zlib1g, libopenblas0-pthread | libopenblas0, liblapacke}"
    installed_size="$(du -sk "$deb_root/usr" | awk '{print $1}')"
    cat >"$deb_root/DEBIAN/control" <<CONTROL
Package: mere-run
Version: $deb_version
Section: science
Priority: optional
Architecture: $deb_arch
Maintainer: mere.run <hello@mere.run>
Depends: $depends
Installed-Size: $installed_size
Homepage: https://mere.run
Description: Local-first inference CLI
 mere.run is a local-first Swift CLI for text, image, audio, video, and agent
 inference workflows. This Linux package ships the headless CLI and colocated
 runtime assets.
CONTROL

    chmod 0755 "$deb_root/DEBIAN"
    dpkg-deb --root-owner-group --build "$deb_root" "$repo_root/$output_dir/mere-run_${deb_version}_${deb_arch}.deb"
    echo "[package-linux] wrote $output_dir/mere-run_${deb_version}_${deb_arch}.deb"
  else
    echo "[package-linux] warning: dpkg-deb not found; skipping .deb package." >&2
  fi
fi

sha256sum "$output_dir"/* >"$output_dir/SHA256SUMS"
echo "[package-linux] wrote $output_dir/SHA256SUMS"
