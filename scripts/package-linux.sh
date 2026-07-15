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
  --artifact-suffix NAME  Append NAME to tar/deb artifact names, e.g. cuda.
  --configuration NAME    Swift configuration: release or debug. Default: release
  --skip-build            Reuse an existing .build/<configuration>/mere.run binary.
  --skip-native           Do not run scripts/prepare-linux-native.sh before building.
  --skip-deb              Do not create a .deb package.
  -h, --help              Show this help.

Environment:
  MERERUN_RELEASE_VERSION       Default package version override.
  MERERUN_PACKAGE_ARTIFACT_SUFFIX
                                Default artifact suffix. Empty by default.
  MERERUN_BUNDLE_SWIFT_LIBS     Copy libswift*/Foundation runtime .so dependencies
                                reported by ldd into payload lib/. Default: 1.
  MERERUN_PACKAGE_LINUX_DEPS    Override Debian Depends field. For CUDA, an
                                explicit value bypasses automatic toolkit-major
                                dependency gating; the caller owns compatibility.
  MERERUN_LINUX_ACCEL           Passed through to prepare-linux-native.sh (cpu/cuda).
  MERERUN_NATIVE_BUILD_JOBS     Passed through to prepare-linux-native.sh.
  MERERUN_CUDA_ARCHITECTURES    Passed through to prepare-linux-native.sh.
  CUDA_LIBRARY_PATH             Optional CUDA toolkit library directory for
                                SwiftPM MLX CUDA linking. Linux SBSA and lib64
                                defaults are detected when unset.
  MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE=1
                                Allow an arm64 CPU package for local smoke tests.
  MERERUN_DS4_LINUX_BIN_DIR     Passed through to prepare-linux-native.sh for DS4 staging.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# shellcheck source=scripts/linux-arm64-bf16-toolchain.sh
source "$repo_root/scripts/linux-arm64-bf16-toolchain.sh"

configuration="release"
output_dir="dist/linux"
artifact_suffix="${MERERUN_PACKAGE_ARTIFACT_SUFFIX:-}"
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
    --artifact-suffix)
      artifact_suffix="${2:?missing value for --artifact-suffix}"
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
    deb_multiarch="x86_64-linux-gnu"
    ;;
  aarch64|arm64)
    platform_arch="arm64"
    deb_arch="arm64"
    deb_multiarch="aarch64-linux-gnu"
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

artifact_suffix_part=""
if [[ -n "$artifact_suffix" ]]; then
  if [[ ! "$artifact_suffix" =~ ^[a-z0-9][a-z0-9.+-]*$ ]]; then
    echo "[package-linux] error: --artifact-suffix must be Debian-safe: lowercase letters, digits, '.', '+', or '-', starting with a letter or digit." >&2
    exit 64
  fi
  artifact_suffix_part="-$artifact_suffix"
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
linux_accel="${MERERUN_LINUX_ACCEL:-cpu}"

cuda_target_names=(sbsa-linux aarch64-linux x86_64-linux)

cuda_toolkit_root_candidates() {
  local seen=":"
  local candidate
  for candidate in \
    "${CUDA_HOME:-}" \
    "${CUDA_PATH:-}" \
    /usr/local/cuda \
    /usr/local/cuda-* \
    /usr; do
    [[ -n "$candidate" && -d "$candidate" ]] || continue
    local resolved
    resolved="$(cd "$candidate" && pwd -P)"
    case "$seen" in
      *":$resolved:"*) continue ;;
    esac
    seen="$seen$resolved:"
    printf '%s\n' "$resolved"
  done
}

cuda_toolkit_major_from_binary() {
  local binary="$1"
  local linked_libraries
  linked_libraries="$(ldd "$binary" 2>&1 || true)"

  # libcudart's SONAME major follows the CUDA toolkit ABI major. Do not infer
  # from cuFFT, cuDNN, NCCL, or the driver: their SONAMEs use independent
  # version lines and can legitimately differ from the toolkit major.
  local sonames
  sonames="$(
    printf '%s\n' "$linked_libraries" \
      | grep -oE 'libcudart\.so\.[0-9]+' \
      | sort -u \
      || true
  )"
  local count
  count="$(printf '%s\n' "$sonames" | sed '/^$/d' | wc -l | tr -d '[:space:]')"
  if [[ "$count" != "1" ]]; then
    if [[ "$count" == "0" ]]; then
      echo "[package-linux] error: could not determine the built CUDA toolkit major from libcudart in $binary." >&2
    else
      echo "[package-linux] error: conflicting libcudart SONAMEs in $binary: $(printf '%s' "$sonames" | paste -sd, -)" >&2
    fi
    return 1
  fi
  printf '%s\n' "${sonames##*.}"
}

case "$linux_accel" in
  cpu|cuda) ;;
  *)
    echo "[package-linux] error: unsupported MERERUN_LINUX_ACCEL=$linux_accel (expected cpu or cuda)." >&2
    exit 64
    ;;
esac
if [[ "$platform_arch" == "arm64" &&
      "$linux_accel" != "cuda" &&
      "${MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE:-0}" != "1" ]]; then
  echo "[package-linux] error: Linux arm64 release packages must use MERERUN_LINUX_ACCEL=cuda." >&2
  echo "[package-linux] CPU arm64 packages are for local smoke tests only; set MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE=1 to force one." >&2
  exit 65
fi
if (( do_build || do_native )); then
  configure_linux_arm64_bf16_toolchain "$platform_arch" "package-linux"
fi

mlx_swift_cuda_link_flags() {
  local mlx_cmake_build="$native_root/build/mlx-swift-cuda-smoke"
  local local_openblas_root="$native_root/deps/apt-root"
  local cudnn_library_path="${CUDNN_LIBRARY_PATH:-}"
  local cuda_library_path="${CUDA_LIBRARY_PATH:-}"
  local cuda_stub_library_path="${CUDA_STUB_LIBRARY_PATH:-}"
  local flags=()

  flags+=("-L" "$mlx_cmake_build/_deps/mlx-c-build")
  flags+=("-L" "$mlx_cmake_build/_deps/mlx-build")
  flags+=("-L" "$mlx_cmake_build/_deps/mlx-build/mlx/io")
  flags+=("-L" "$mlx_cmake_build/lib")

  if [[ -f "$local_openblas_root/usr/lib/$deb_multiarch/openblas-pthread/libopenblas.so" ]]; then
    flags+=("-L" "$local_openblas_root/usr/lib/$deb_multiarch/openblas-pthread")
    flags+=("-Xlinker" "-rpath" "-Xlinker" "$local_openblas_root/usr/lib/$deb_multiarch/openblas-pthread")
  fi
  if [[ -z "$cudnn_library_path" ]]; then
    local cudnn_library_candidates=("/usr/lib/$deb_multiarch")
    local cuda_root
    while IFS= read -r cuda_root; do
      cudnn_library_candidates+=("$cuda_root/lib64")
      local cuda_target
      for cuda_target in "${cuda_target_names[@]}"; do
        cudnn_library_candidates+=("$cuda_root/targets/$cuda_target/lib")
      done
    done < <(cuda_toolkit_root_candidates)
    for candidate in "${cudnn_library_candidates[@]}"; do
      if [[ -f "$candidate/libcudnn.so" || -f "$candidate/libcudnn.so.9" ]]; then
        cudnn_library_path="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$cudnn_library_path" ]]; then
    flags+=("-L" "$cudnn_library_path")
    flags+=("-Xlinker" "-rpath" "-Xlinker" "$cudnn_library_path")
    for cudnn_lib in \
      libcudnn.so.9 \
      libcudnn_graph.so.9 \
      libcudnn_engines_runtime_compiled.so.9 \
      libcudnn_ops.so.9 \
      libcudnn_cnn.so.9 \
      libcudnn_adv.so.9 \
      libcudnn_engines_precompiled.so.9 \
      libcudnn_heuristic.so.9; do
      if [[ -f "$cudnn_library_path/$cudnn_lib" ]]; then
        flags+=("$cudnn_library_path/$cudnn_lib")
      fi
    done
  fi
  if [[ -z "$cuda_library_path" ]]; then
    local cuda_library_candidates=()
    local cuda_root
    while IFS= read -r cuda_root; do
      cuda_library_candidates+=("$cuda_root/lib64")
      local cuda_target
      for cuda_target in "${cuda_target_names[@]}"; do
        cuda_library_candidates+=("$cuda_root/targets/$cuda_target/lib")
      done
    done < <(cuda_toolkit_root_candidates)
    for candidate in "${cuda_library_candidates[@]}"; do
      if [[ -f "$candidate/libcublasLt.so" || -f "$candidate/libnvrtc.so" || -f "$candidate/libcudart.so" ]]; then
        cuda_library_path="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$cuda_stub_library_path" &&
        ( -z "$cuda_library_path" || ! -f "$cuda_library_path/libcuda.so" ) ]]; then
    local cuda_stub_candidates=()
    local cuda_root
    while IFS= read -r cuda_root; do
      cuda_stub_candidates+=("$cuda_root/lib64/stubs")
      local cuda_target
      for cuda_target in "${cuda_target_names[@]}"; do
        cuda_stub_candidates+=("$cuda_root/targets/$cuda_target/lib/stubs")
      done
    done < <(cuda_toolkit_root_candidates)
    for candidate in "${cuda_stub_candidates[@]}"; do
      if [[ -f "$candidate/libcuda.so" ]]; then
        cuda_stub_library_path="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$cuda_library_path" ]]; then
    flags+=("-L" "$cuda_library_path")
    if [[ "$(basename "$cuda_library_path")" != "stubs" ]]; then
      flags+=("-Xlinker" "-rpath" "-Xlinker" "$cuda_library_path")
    fi
  fi
  if [[ -n "$cuda_stub_library_path" && "$cuda_stub_library_path" != "$cuda_library_path" ]]; then
    flags+=("-L" "$cuda_stub_library_path")
  fi
  if [[ -d /usr/lib/$deb_multiarch ]]; then
    flags+=("-L" "/usr/lib/$deb_multiarch")
  fi

  flags+=("-lswiftSwiftOnoneSupport")
  flags+=(
    "-lcublasLt"
    "-lnvrtc"
    "-lcuda"
    "-lcudart"
    "-lcufft"
    "-lnccl"
    "-lrt"
  )

  printf '%q ' "${flags[@]}"
}

configure_swiftpm_cuda_bridge() {
  if [[ "$linux_accel" != "cuda" ]]; then
    return
  fi

  local mlx_linkage
  mlx_linkage="$(printf '%s' "${MERERUN_MLX_SWIFT_LINKAGE:-}" | tr '[:upper:]' '[:lower:]')"
  if [[ -n "$mlx_linkage" && "$mlx_linkage" != "cuda-prebuilt" ]]; then
    echo "[package-linux] error: MERERUN_LINUX_ACCEL=cuda requires MERERUN_MLX_SWIFT_LINKAGE=cuda-prebuilt." >&2
    exit 64
  fi

  export MERERUN_MLX_SWIFT_LINKAGE="${MERERUN_MLX_SWIFT_LINKAGE:-cuda-prebuilt}"
  export MERERUN_MLX_SWIFT_BUILD_DIR="${MERERUN_MLX_SWIFT_BUILD_DIR:-$native_root/build/mlx-swift-cuda-smoke}"
  export MERERUN_MLX_SWIFT_SOURCE_DIR="${MERERUN_MLX_SWIFT_SOURCE_DIR:-$repo_root/.build/native/src/mlx-swift}"
  export MERERUN_MLX_SWIFT_LINK_FLAGS="${MERERUN_MLX_SWIFT_LINK_FLAGS:-$(mlx_swift_cuda_link_flags)}"
}

if (( do_native )); then
  echo "[package-linux] preparing Linux native runtime artifacts"
  bash scripts/prepare-linux-native.sh
fi

configure_swiftpm_cuda_bridge

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

if [[ "$output_dir" == /* ]]; then
  output_root="$output_dir"
else
  output_root="$repo_root/$output_dir"
fi
rm -rf "$output_root"
mkdir -p "$output_root"
staging="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-linux-package.XXXXXX")"
cleanup() {
  rm -rf "$staging"
}
trap cleanup EXIT

payload_name="mere-run-${version}-linux-${platform_arch}${artifact_suffix_part}"
payload_dir="$staging/$payload_name"
mkdir -p "$payload_dir"

cp -a "$cli_executable" "$payload_dir/mere.run-bin"
chmod +x "$payload_dir/mere.run-bin"
cat >"$payload_dir/mere.run" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
source_path="${BASH_SOURCE[0]}"
while [[ -L "$source_path" ]]; do
  source_dir="$(cd -P "$(dirname "$source_path")" && pwd)"
  link_target="$(readlink "$source_path")"
  if [[ "$link_target" == /* ]]; then
    source_path="$link_target"
  else
    source_path="$source_dir/$link_target"
  fi
done
payload_dir="$(cd -P "$(dirname "$source_path")" && pwd)"
if [[ -f "$payload_dir/.mererun-linux-cuda" ]]; then
  export MERERUN_LINUX_ACCEL="${MERERUN_LINUX_ACCEL:-cuda}"
  if [[ -z "${CUDA_HOME:-}" && -z "${CUDA_PATH:-}" ]]; then
    for cuda_runtime_root in /usr/local/cuda /usr/local/cuda-* /usr; do
      if [[ -f "$cuda_runtime_root/include/cuda.h" ]]; then
        export CUDA_HOME="$cuda_runtime_root"
        break
      fi
    done
  fi
  if [[ -z "${CUDA_HOME:-}" && -n "${CUDA_PATH:-}" ]]; then
    export CUDA_HOME="$CUDA_PATH"
  fi
  if [[ -z "${CUDA_PATH:-}" && -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="$CUDA_HOME"
  fi
fi
if [[ -x "$payload_dir/llama-cli" && -z "${MERERUN_LLAMA_CLI:-}" ]]; then
  export MERERUN_LLAMA_CLI="$payload_dir/llama-cli"
fi
cuda_cccl_candidates=()
for cuda_root in \
  "${CUDA_HOME:-}" \
  "${CUDA_PATH:-}" \
  /usr/local/cuda \
  /usr/local/cuda-* \
  /usr
do
  [[ -n "$cuda_root" && -d "$cuda_root" ]] || continue
  cuda_cccl_candidates+=("$cuda_root/include/cccl")
  for cuda_target in sbsa-linux aarch64-linux x86_64-linux; do
    cuda_cccl_candidates+=("$cuda_root/targets/$cuda_target/include/cccl")
  done
done
cuda_cccl_candidates+=(/usr/include/cccl)
for cuda_cccl_include in "${cuda_cccl_candidates[@]}"; do
  if [[ -d "$cuda_cccl_include/cuda/std" ]]; then
    export MERERUN_CUDA_CCCL_INCLUDE_PATH="$cuda_cccl_include"
    case ":${CPATH:-}:" in
      *":$cuda_cccl_include:"*) ;;
      *) export CPATH="$cuda_cccl_include${CPATH:+:$CPATH}" ;;
    esac
    break
  fi
done
export LD_LIBRARY_PATH="$payload_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$payload_dir/mere.run-bin" "$@"
WRAPPER
chmod +x "$payload_dir/mere.run"
if [[ "$linux_accel" == "cuda" ]]; then
  touch "$payload_dir/.mererun-linux-cuda"
fi
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
  "$build_dir"/*.so.* \
  "$build_dir"/*.resources
do
  stage_asset "$asset"
done

if [[ -d "$llama_prefix/lib" ]]; then
  cp -a "$llama_prefix/lib" "$payload_dir/lib"
fi
if [[ -x "$llama_prefix/bin/llama-cli" ]]; then
  cp -a "$llama_prefix/bin/llama-cli" "$payload_dir/llama-cli"
  chmod +x "$payload_dir/llama-cli"
fi
# Bundle the persistent llama.cpp HTTP server too. `api serve` can spawn it once
# (isolated subprocess, so no in-process llama/MLX CUDA collision on GB10) and
# proxy to it, keeping the GGUF model resident across requests instead of the
# one-shot llama-cli reload-per-call.
if [[ -x "$llama_prefix/bin/llama-server" ]]; then
  cp -a "$llama_prefix/bin/llama-server" "$payload_dir/llama-server"
  chmod +x "$payload_dir/llama-server"
fi

# Bundle Swift/Foundation runtime shared libraries when the official Swift
# toolchain linked them dynamically, plus OpenBLAS for tarball installs. This
# lets the tarball/deb run on clean Ubuntu hosts without requiring users to
# install the Swift compiler toolchain first.
if [[ "${MERERUN_BUNDLE_SWIFT_LIBS:-1}" == "1" ]]; then
  mkdir -p "$payload_dir/lib"
  while IFS= read -r lib_path; do
    [[ -f "$lib_path" ]] || continue
    case "$(basename "$lib_path")" in
      libswift*|libFoundation*|lib_Foundation*|libdispatch*|libBlocksRuntime*|libopenblas*)
        resolved_lib_path="$(readlink -f "$lib_path")"
        if [[ -f "$resolved_lib_path" ]]; then
          cp -aL "$resolved_lib_path" "$payload_dir/lib/$(basename "$lib_path")"
        fi
        ;;
    esac
  done < <(ldd "$payload_dir/mere.run-bin" 2>/dev/null | awk '/=> \/|^\// { for (i=1; i<=NF; i++) if ($i ~ /^\//) print $i }' | sort -u)
fi

if [[ -d vendor/ds4 ]]; then
  mkdir -p "$payload_dir/vendor"
  cp -a vendor/ds4 "$payload_dir/vendor/ds4"
fi

(
  cd "$staging"
  tar -czf "$output_root/${payload_name}.tar.gz" "$payload_name"
)

echo "[package-linux] wrote $output_dir/${payload_name}.tar.gz"

if (( do_deb )); then
  if command -v dpkg-deb >/dev/null 2>&1; then
    deb_root="$staging/deb-root"
    install_root="$deb_root/usr/lib/mere-run"
    mkdir -p "$install_root" "$deb_root/usr/bin" "$deb_root/DEBIAN"
    cp -a "$payload_dir"/. "$install_root/"
    ln -s ../lib/mere-run/mere.run "$deb_root/usr/bin/mere.run"

    default_depends="ffmpeg, libcurl4, zlib1g, libopenblas0-pthread | libopenblas0, liblapacke"
    if [[ "$linux_accel" == "cuda" && -z "${MERERUN_PACKAGE_LINUX_DEPS:-}" ]]; then
      if ! cuda_toolkit_major="$(cuda_toolkit_major_from_binary "$payload_dir/mere.run-bin")"; then
        echo "[package-linux] error: refusing to emit a CUDA .deb with unverifiable runtime dependencies." >&2
        echo "[package-linux] error: use --skip-deb to keep the already-written portable tarball." >&2
        exit 70
      fi
      case "$cuda_toolkit_major" in
        13)
          default_depends="$default_depends, cuda-cccl-13-0, cuda-cudart-13-0, cuda-nvrtc-13-0, libcublas-13-0, libcufft-13-0, libcudnn9-cuda-13, libnccl2"
          ;;
        *)
          echo "[package-linux] error: CUDA .deb packaging supports toolkit major 13, but the built binary links libcudart.so.$cuda_toolkit_major." >&2
          echo "[package-linux] error: use --skip-deb for this toolkit; the already-written tarball remains valid for a matching CUDA host." >&2
          exit 70
          ;;
      esac
    fi
    depends="${MERERUN_PACKAGE_LINUX_DEPS:-$default_depends}"
    installed_size="$(du -sk "$deb_root/usr" | awk '{print $1}')"
    deb_package_name="mere-run${artifact_suffix_part}"
    cat >"$deb_root/DEBIAN/control" <<CONTROL
Package: $deb_package_name
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
    dpkg-deb --root-owner-group --build "$deb_root" "$output_root/${deb_package_name}_${deb_version}_${deb_arch}.deb"
    echo "[package-linux] wrote $output_dir/${deb_package_name}_${deb_version}_${deb_arch}.deb"
  else
    echo "[package-linux] warning: dpkg-deb not found; skipping .deb package." >&2
  fi
fi

(
  cd "$output_root"
  rm -f SHA256SUMS
  sha256sum ./* | sed 's#  \./#  #' >SHA256SUMS
)
echo "[package-linux] wrote $output_dir/SHA256SUMS"
