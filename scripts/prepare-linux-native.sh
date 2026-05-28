#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/prepare-linux-native.sh [--check] [--help]

Build or verify Linux-native runtime artifacts for the mere.run CLI.

The script builds a pinned llama.cpp shared-library install into:
  .build/native/linux-<arch>/llama

It also writes a pkg-config file at:
  .build/native/linux-<arch>/pkgconfig/llama.pc

On Linux it also resolves SwiftPM dependencies for CPU builds. CUDA builds validate
upstream mlx-swift's CMake lane and print a SwiftPM bridge environment that makes
mere.run consume those prebuilt CUDA artifacts.

Options:
  --check     Verify the current native layout without building.
  -h, --help  Show this help.

Environment:
  LLAMA_CPP_COMMIT            Override pinned llama.cpp commit.
  LLAMA_CPP_URL               Override llama.cpp git URL.
  MERERUN_LINUX_ACCEL         Native acceleration mode. Set to cuda to build
                              llama.cpp with GGML_CUDA=ON and run an optional
                              mlx-swift CMake CUDA smoke. Default: cpu.
  MERERUN_LLAMA_GPU_LAYERS    Override llama.cpp GPU offload layers on Linux.
                              Defaults to all layers when MERERUN_LINUX_ACCEL=cuda
                              and 0 otherwise.
  CUDNN_INCLUDE_PATH          Optional cuDNN include directory for mlx-swift
                              CMake CUDA smoke.
  CUDNN_LIBRARY_PATH          Optional cuDNN library directory for mlx-swift
                              CMake CUDA smoke.
                              When unset, Linux defaults are detected from
                              /usr/include/<multiarch> and
                              /usr/lib/<multiarch>.
  CUDA_LIBRARY_PATH           Optional CUDA toolkit library directory for the
                              SwiftPM MLX CUDA bridge. When unset, Linux SBSA
                              and lib64 defaults are detected.
  BLAS_LIBRARIES              Optional BLAS library path for mlx-swift CUDA smoke.
  BLAS_INCLUDE_DIRS           Optional BLAS include directory for mlx-swift CUDA smoke.
  LAPACK_LIBRARIES            Optional LAPACK library path for mlx-swift CUDA smoke.
  LAPACK_INCLUDE_DIRS         Optional LAPACK include directory for mlx-swift CUDA smoke.
  MERERUN_SKIP_MLX_SWIFT_PATCH=1
                              Skip the Linux mlx-swift SwiftPM package fix.
  MERERUN_SKIP_MLX_CUDA_SMOKE=1
                              Skip the mlx-swift CMake CUDA smoke when
                              MERERUN_LINUX_ACCEL=cuda.
  MERERUN_MLX_SWIFT_LINKAGE=cuda-prebuilt
                              Package.swift mode that consumes CMake-built
                              mlx-swift CUDA artifacts instead of SwiftPM mlx.
                              prepare-linux-native.sh prints the full export set.
  MERERUN_DS4_LINUX_BIN_DIR   Optional directory containing Linux ds4 binaries
                              to stage under vendor/ds4/linux-<arch>/.
USAGE
}

check_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      check_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[prepare-linux-native] error: unknown argument: $1" >&2
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
  echo "[prepare-linux-native] non-Linux host; skipping Linux-native asset preparation."
  exit 0
fi

case "$(uname -m)" in
  x86_64|amd64)
    arch="x86_64"
    deb_multiarch="x86_64-linux-gnu"
    ;;
  aarch64|arm64)
    arch="arm64"
    deb_multiarch="aarch64-linux-gnu"
    ;;
  *)
    echo "[prepare-linux-native] error: unsupported Linux architecture: $(uname -m)" >&2
    exit 65
    ;;
esac

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[prepare-linux-native] error: required tool not found in PATH: $tool" >&2
    exit 127
  fi
}

patch_mlx_cuda_jit_include_path() {
  local mlx_jit_module="$1"
  if [[ ! -f "$mlx_jit_module" ]] ||
     grep -Fq "MERERUN_CUDA_CCCL_INCLUDE" "$mlx_jit_module"; then
    return
  fi

  local mlx_jit_tmp
  mlx_jit_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-mlx-jit.XXXXXX")"
  awk '
    {
      print
      if ($0 ~ /args\.push_back\(fmt::format\("--include-path=\{\}\/include", home\)\);/) {
        print "        // MERERUN_CUDA_CCCL_INCLUDE: CUDA 13 SBSA installs cuda/std under include/cccl."
        print "        auto cuda_cccl_path = std::filesystem::path(home) / \"include\" / \"cccl\";"
        print "        if (std::filesystem::exists(cuda_cccl_path)) {"
        print "          args.push_back(fmt::format(\"--include-path={}\", cuda_cccl_path.string()));"
        print "        }"
      }
    }
  ' "$mlx_jit_module" >"$mlx_jit_tmp"
  mv "$mlx_jit_tmp" "$mlx_jit_module"
  echo "[prepare-linux-native] patched mlx-swift CUDA JIT include path in ${mlx_jit_module#$repo_root/}."
}

detect_cuda_dependency_defaults() {
  if [[ "$linux_accel" != "cuda" ]]; then
    return
  fi

  if [[ -z "${CUDNN_INCLUDE_PATH:-}" ]]; then
    for candidate in \
      "/usr/include/$deb_multiarch" \
      /usr/local/cuda/include \
      /usr/local/cuda/targets/sbsa-linux/include; do
      if [[ -f "$candidate/cudnn.h" ]]; then
        export CUDNN_INCLUDE_PATH="$candidate"
        break
      fi
    done
  fi

  if [[ -z "${CUDNN_LIBRARY_PATH:-}" ]]; then
    for candidate in \
      "/usr/lib/$deb_multiarch" \
      /usr/local/cuda/lib64 \
      /usr/local/cuda/targets/sbsa-linux/lib; do
      if [[ -f "$candidate/libcudnn.so" || -f "$candidate/libcudnn.so.9" ]]; then
        export CUDNN_LIBRARY_PATH="$candidate"
        break
      fi
    done
  fi
}

platform_dir="linux-$arch"
native_root="$repo_root/.build/native/$platform_dir"
llama_prefix="$native_root/llama"
pkgconfig_dir="$native_root/pkgconfig"
llama_pc="$pkgconfig_dir/llama.pc"
llama_src="$repo_root/.build/native/src/llama.cpp"
llama_build="$native_root/build/llama.cpp"
llama_commit="${LLAMA_CPP_COMMIT:-6d957078270f58d4ea14e8c205f5ef4e49be33f3}"
llama_url="${LLAMA_CPP_URL:-https://github.com/ggml-org/llama.cpp.git}"
mlx_swift_checkout="$repo_root/.build/checkouts/mlx-swift"
swift_numerics_checkout="$repo_root/.build/checkouts/swift-numerics"
linux_accel="${MERERUN_LINUX_ACCEL:-cpu}"
case "$linux_accel" in
  cpu|cuda)
    ;;
  *)
    echo "[prepare-linux-native] error: unsupported MERERUN_LINUX_ACCEL=$linux_accel (expected cpu or cuda)" >&2
    exit 64
    ;;
esac
if [[ "$check_only" != "1" ]]; then
  configure_linux_arm64_bf16_toolchain "$arch" "prepare-linux-native"
fi

stage_ds4() {
  local ds4_root="$repo_root/vendor/ds4"
  local ds4_platform_dir="$ds4_root/$platform_dir"

  if [[ -n "${MERERUN_DS4_LINUX_BIN_DIR:-}" ]]; then
    if [[ "$check_only" == "1" ]]; then
      echo "[prepare-linux-native] --check set; not staging MERERUN_DS4_LINUX_BIN_DIR."
    else
      if [[ ! -d "$MERERUN_DS4_LINUX_BIN_DIR" ]]; then
        echo "[prepare-linux-native] error: MERERUN_DS4_LINUX_BIN_DIR is not a directory: $MERERUN_DS4_LINUX_BIN_DIR" >&2
        exit 66
      fi
      mkdir -p "$ds4_platform_dir"
      cp -a "$MERERUN_DS4_LINUX_BIN_DIR"/. "$ds4_platform_dir"/
      echo "[prepare-linux-native] staged DS4 binaries into vendor/ds4/$platform_dir"
    fi
  fi

  if [[ -d "$ds4_platform_dir" ]]; then
    for executable_name in ds4 ds4-bench ds4-server; do
      executable_path="$ds4_platform_dir/$executable_name"
      if [[ -f "$executable_path" ]]; then
        chmod +x "$executable_path"
      fi
    done
    if [[ ! -x "$ds4_platform_dir/ds4-server" ]]; then
      echo "[prepare-linux-native] warning: vendor/ds4/$platform_dir/ds4-server is missing or not executable." >&2
    fi
  else
    echo "[prepare-linux-native] warning: vendor/ds4/$platform_dir is not present; DS4 runtime lookup will fall back to PATH." >&2
  fi
}

write_llama_pc() {
  mkdir -p "$pkgconfig_dir"
  cat >"$llama_pc" <<PC
prefix=$llama_prefix
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: llama
Description: llama.cpp runtime for mere.run
Version: $llama_commit
Libs: -L\${libdir} -l:libllama.so
Cflags: -I\${includedir}
PC
}

build_llama() {
  require_tool git
  require_tool cmake
  require_tool pkg-config

  mkdir -p "$(dirname "$llama_src")" "$llama_build" "$llama_prefix"

  if [[ ! -d "$llama_src/.git" ]]; then
    echo "[prepare-linux-native] cloning $llama_url"
    git clone --filter=blob:none "$llama_url" "$llama_src"
  fi

  echo "[prepare-linux-native] checking out llama.cpp $llama_commit"
  git -C "$llama_src" fetch --depth 1 origin "$llama_commit"
  git -C "$llama_src" checkout --detach "$llama_commit"

  local cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$llama_prefix"
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_CURL=OFF
  )

  if [[ "$linux_accel" == "cuda" ]]; then
    require_tool nvcc
    echo "[prepare-linux-native] enabling llama.cpp CUDA via GGML_CUDA=ON"
    cmake_args+=(
      -DGGML_CUDA=ON
    )
  fi

  echo "[prepare-linux-native] configuring llama.cpp"
  cmake -S "$llama_src" -B "$llama_build" "${cmake_args[@]}"

  echo "[prepare-linux-native] building llama.cpp"
  cmake --build "$llama_build" --config Release --parallel "$(nproc)"

  echo "[prepare-linux-native] installing llama.cpp into $llama_prefix"
  cmake --install "$llama_build"
  write_llama_pc
}

patch_mlx_swift_for_linux() {
  if [[ "${MERERUN_SKIP_MLX_SWIFT_PATCH:-0}" == "1" || "$linux_accel" == "cuda" ]]; then
    echo "[prepare-linux-native] skipping mlx-swift SwiftPM package fix; CUDA builds use the CMake prebuilt bridge."
    return
  fi

  require_tool swift
  echo "[prepare-linux-native] resolving Swift package dependencies"
  swift package resolve

  local package_file="$mlx_swift_checkout/Package.swift"
  if [[ ! -f "$package_file" ]]; then
    echo "[prepare-linux-native] error: expected mlx-swift checkout at $mlx_swift_checkout" >&2
    exit 68
  fi

  if grep -Fq '"mlx-c/mlx/c/fast.cpp",  // Exclude on Linux - calls metal_kernel unconditionally' "$package_file"; then
    sed -i '/"mlx-c\/mlx\/c\/fast.cpp",  \/\/ Exclude on Linux - calls metal_kernel unconditionally/d' "$package_file"
  fi
  if grep -Fq '"MLXFast.swift",' "$package_file"; then
    sed -i '/"MLXFast.swift",/d;/"MLXFastKernel.swift",/d' "$package_file"
  fi

  local numerics_header="$swift_numerics_checkout/Sources/_NumericsShims/include/_NumericsShims.h"
  local numerics_float16="$swift_numerics_checkout/Sources/RealModule/Float16+Real.swift"
  if [[ -f "$numerics_header" ]] && grep -Fq '#if !defined __wasm__ // No _Float16 on wasm' "$numerics_header"; then
    sed -i 's/#if !defined __wasm__ \/\/ No _Float16 on wasm/#if !defined __wasm__ \&\& defined(__FLT16_MANT_DIG__) \/\/ No _Float16 on wasm, or on targets whose C compiler lacks _Float16/' "$numerics_header"
  fi
  if [[ -f "$numerics_float16" ]] && grep -Fq '#if !arch(wasm32)' "$numerics_float16"; then
    sed -i 's/#if !arch(wasm32)/#if !arch(wasm32) \&\& !(os(Linux) \&\& arch(x86_64))/' "$numerics_float16"
  fi

  echo "[prepare-linux-native] mlx-swift SwiftPM package fix applied."
}

smoke_mlx_swift_cuda() {
  if [[ "$linux_accel" != "cuda" ]]; then
    return
  fi
  if [[ "${MERERUN_SKIP_MLX_CUDA_SMOKE:-0}" == "1" ]]; then
    echo "[prepare-linux-native] skipping mlx-swift CMake CUDA smoke by request."
    return
  fi

  require_tool git
  require_tool cmake
  require_tool ninja
  require_tool nvcc
  require_tool swift

  if [[ -z "${CUDA_HOME:-}" && -f /usr/include/cuda.h ]]; then
    export CUDA_HOME=/usr
  fi
  if [[ -z "${CUDA_PATH:-}" && -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="$CUDA_HOME"
  fi

  local mlx_cmake_src="$repo_root/.build/native/src/mlx-swift"
  local mlx_cmake_build="$native_root/build/mlx-swift-cuda-smoke"

  if [[ ! -d "$mlx_cmake_src/.git" ]]; then
    echo "[prepare-linux-native] cloning mlx-swift for CMake CUDA smoke"
    git clone --filter=blob:none https://github.com/ml-explore/mlx-swift.git "$mlx_cmake_src"
  fi

  echo "[prepare-linux-native] updating mlx-swift CMake checkout"
  git -C "$mlx_cmake_src" fetch --depth 1 origin main
  git -C "$mlx_cmake_src" checkout --detach FETCH_HEAD

  patch_mlx_cuda_jit_include_path "$mlx_cmake_src/Source/Cmlx/mlx/mlx/backend/cuda/jit_module.cpp"
  patch_mlx_cuda_jit_include_path "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/jit_module.cpp"

  local local_openblas_root="$native_root/deps/apt-root"
  local local_openblas_lib="$local_openblas_root/usr/lib/$deb_multiarch/openblas-pthread/libopenblas.so"
  local local_openblas_include="$local_openblas_root/usr/include/$deb_multiarch/openblas-pthread;$local_openblas_root/usr/include"

  if [[ ! -f "$local_openblas_root/usr/include/$deb_multiarch/openblas-pthread/cblas.h" &&
        ! -f /usr/include/$deb_multiarch/openblas-pthread/cblas.h &&
        ! -f /usr/include/cblas.h ]]; then
    if command -v apt-get >/dev/null 2>&1 && command -v dpkg-deb >/dev/null 2>&1; then
      echo "[prepare-linux-native] downloading local OpenBLAS/LAPACK headers for mlx-swift CUDA smoke"
      local apt_cache="$native_root/deps/apt-cache"
      mkdir -p "$apt_cache" "$local_openblas_root"
      (
        cd "$apt_cache"
        apt-get download libopenblas-dev libopenblas-pthread-dev libopenblas0-pthread liblapacke-dev liblapacke
        for deb in *.deb; do
          dpkg-deb -x "$deb" "$local_openblas_root"
        done
      )
    else
      echo "[prepare-linux-native] warning: OpenBLAS/LAPACK headers were not found; set BLAS_INCLUDE_DIRS/LAPACK_INCLUDE_DIRS or install libopenblas-dev liblapacke-dev." >&2
    fi
  fi
  if [[ ! -f "$local_openblas_root/usr/include/lapack.h" &&
        ! -f /usr/include/lapack.h ]]; then
    if command -v apt-get >/dev/null 2>&1 && command -v dpkg-deb >/dev/null 2>&1; then
      echo "[prepare-linux-native] downloading local LAPACK headers for mlx-swift CUDA smoke"
      local apt_cache="$native_root/deps/apt-cache"
      mkdir -p "$apt_cache" "$local_openblas_root"
      (
        cd "$apt_cache"
        apt-get download liblapacke-dev liblapacke
        for deb in *.deb; do
          dpkg-deb -x "$deb" "$local_openblas_root"
        done
      )
    fi
  fi

  local mlx_cmake_args=(
    -DMLX_BUILD_METAL=OFF
    -DMLX_BUILD_CUDA=ON
    -DMLX_C_BUILD_EXAMPLES=OFF
  )
  detect_cuda_dependency_defaults
  if [[ -n "${CUDNN_INCLUDE_PATH:-}" ]]; then
    mlx_cmake_args+=("-DCUDNN_INCLUDE_PATH=$CUDNN_INCLUDE_PATH")
  fi
  if [[ -n "${CUDNN_LIBRARY_PATH:-}" ]]; then
    mlx_cmake_args+=("-DCUDNN_LIBRARY_PATH=$CUDNN_LIBRARY_PATH")
  fi
  local local_openblas_root="$native_root/deps/apt-root"
  local local_openblas_lib="$local_openblas_root/usr/lib/$deb_multiarch/openblas-pthread/libopenblas.so"
  local local_openblas_include="$local_openblas_root/usr/include/$deb_multiarch/openblas-pthread;$local_openblas_root/usr/include"

  if [[ -n "${BLAS_LIBRARIES:-}" ]]; then
    mlx_cmake_args+=("-DBLAS_LIBRARIES=$BLAS_LIBRARIES")
  elif [[ -f "$local_openblas_lib" ]]; then
    mlx_cmake_args+=("-DBLAS_LIBRARIES=$local_openblas_lib")
  elif [[ -f /usr/lib/$deb_multiarch/openblas-pthread/libopenblas.so ]]; then
    mlx_cmake_args+=("-DBLAS_LIBRARIES=/usr/lib/$deb_multiarch/openblas-pthread/libopenblas.so")
  elif [[ -f /usr/lib/$deb_multiarch/blas/libblas.so.3.12.0 ]]; then
    mlx_cmake_args+=("-DBLAS_LIBRARIES=/usr/lib/$deb_multiarch/blas/libblas.so.3.12.0")
  fi
  if [[ -n "${BLAS_INCLUDE_DIRS:-}" ]]; then
    mlx_cmake_args+=("-DBLAS_INCLUDE_DIRS=$BLAS_INCLUDE_DIRS")
  elif [[ -f "$local_openblas_root/usr/include/$deb_multiarch/openblas-pthread/cblas.h" ]]; then
    mlx_cmake_args+=("-DBLAS_INCLUDE_DIRS=$local_openblas_include")
  elif [[ -d /usr/include/$deb_multiarch/openblas-pthread ]]; then
    mlx_cmake_args+=("-DBLAS_INCLUDE_DIRS=/usr/include/$deb_multiarch/openblas-pthread")
  fi
  if [[ -n "${LAPACK_LIBRARIES:-}" ]]; then
    mlx_cmake_args+=("-DLAPACK_LIBRARIES=$LAPACK_LIBRARIES")
  elif [[ -f "$local_openblas_lib" ]]; then
    mlx_cmake_args+=("-DLAPACK_LIBRARIES=$local_openblas_lib")
  elif [[ -f /usr/lib/$deb_multiarch/openblas-pthread/libopenblas.so ]]; then
    mlx_cmake_args+=("-DLAPACK_LIBRARIES=/usr/lib/$deb_multiarch/openblas-pthread/libopenblas.so")
  elif [[ -f /usr/lib/$deb_multiarch/lapack/liblapack.so.3.12.0 ]]; then
    mlx_cmake_args+=("-DLAPACK_LIBRARIES=/usr/lib/$deb_multiarch/lapack/liblapack.so.3.12.0")
  fi
  if [[ -n "${LAPACK_INCLUDE_DIRS:-}" ]]; then
    mlx_cmake_args+=("-DLAPACK_INCLUDE_DIRS=$LAPACK_INCLUDE_DIRS")
  elif [[ -f "$local_openblas_root/usr/include/lapack.h" ]]; then
    mlx_cmake_args+=("-DLAPACK_INCLUDE_DIRS=$local_openblas_include")
  elif [[ -d /usr/include/$deb_multiarch/openblas-pthread ]]; then
    mlx_cmake_args+=("-DLAPACK_INCLUDE_DIRS=/usr/include/$deb_multiarch/openblas-pthread")
  fi

  echo "[prepare-linux-native] configuring mlx-swift CUDA smoke via CMake"
  cmake -S "$mlx_cmake_src" -B "$mlx_cmake_build" -G Ninja "${mlx_cmake_args[@]}"

  echo "[prepare-linux-native] building mlx-swift CUDA smoke"
  cmake --build "$mlx_cmake_build" --parallel "$(nproc)"

  local example="$mlx_cmake_build/example1"
  if [[ -x "$example" ]]; then
    echo "[prepare-linux-native] running mlx-swift CUDA smoke example"
    "$example" --device gpu
  else
    echo "[prepare-linux-native] warning: mlx-swift CUDA build finished but example1 was not produced; CUDA is not wired into SwiftPM package consumption yet." >&2
  fi

  cat <<NOTICE
[prepare-linux-native] MLX CUDA CMake smoke completed.
[prepare-linux-native] The CMake-built MLX Swift artifacts are available for SwiftPM builds via:
  export MERERUN_MLX_SWIFT_LINKAGE="cuda-prebuilt"
  export MERERUN_MLX_SWIFT_BUILD_DIR="$mlx_cmake_build"
  export MERERUN_MLX_SWIFT_SOURCE_DIR="$mlx_cmake_src"
NOTICE
}

mlx_swift_cuda_link_flags() {
  local mlx_cmake_build="$native_root/build/mlx-swift-cuda-smoke"
  local local_openblas_root="$native_root/deps/apt-root"
  local cudnn_library_path="${CUDNN_LIBRARY_PATH:-}"
  local cuda_library_path="${CUDA_LIBRARY_PATH:-}"
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
    for candidate in \
      "/usr/lib/$deb_multiarch" \
      /usr/local/cuda/lib64 \
      /usr/local/cuda/targets/sbsa-linux/lib; do
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
    for candidate in \
      "${CUDA_HOME:-}/lib64" \
      "${CUDA_HOME:-}/targets/sbsa-linux/lib" \
      "${CUDA_PATH:-}/lib64" \
      "${CUDA_PATH:-}/targets/sbsa-linux/lib" \
      /usr/local/cuda/lib64 \
      /usr/local/cuda/targets/sbsa-linux/lib; do
      if [[ -f "$candidate/libcublasLt.so" || -f "$candidate/libnvrtc.so" || -f "$candidate/libcudart.so" ]]; then
        cuda_library_path="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$cuda_library_path" ]]; then
    flags+=("-L" "$cuda_library_path")
    flags+=("-Xlinker" "-rpath" "-Xlinker" "$cuda_library_path")
  fi
  if [[ -d /usr/lib/$deb_multiarch ]]; then
    flags+=("-L" "/usr/lib/$deb_multiarch")
  fi

  flags+=(
    "-lcublasLt"
    "-lnvrtc"
    "-lcuda"
    "-lcudart"
    "-lnccl"
    "-lrt"
  )

  printf '%q ' "${flags[@]}"
}

verify_mlx_swift_cuda_bridge() {
  if [[ "$linux_accel" != "cuda" || "${MERERUN_SKIP_MLX_CUDA_SMOKE:-0}" == "1" ]]; then
    return
  fi

  local mlx_cmake_src="$repo_root/.build/native/src/mlx-swift"
  local mlx_cmake_build="$native_root/build/mlx-swift-cuda-smoke"
  local missing=0
  local required_paths=(
    "$mlx_cmake_src/Source/Cmlx/include/module.modulemap"
    "$mlx_cmake_build/MLX.swiftmodule"
    "$mlx_cmake_build/MLXFast.swiftmodule"
    "$mlx_cmake_build/MLXNN.swiftmodule"
    "$mlx_cmake_build/MLXRandom.swiftmodule"
    "$mlx_cmake_build/MLXOptimizers.swiftmodule"
    "$mlx_cmake_build/libMLX.a"
    "$mlx_cmake_build/libMLXFast.a"
    "$mlx_cmake_build/libMLXNN.a"
    "$mlx_cmake_build/libMLXRandom.a"
    "$mlx_cmake_build/libMLXOptimizers.a"
    "$mlx_cmake_build/_deps/mlx-c-build/libmlxc.a"
    "$mlx_cmake_build/_deps/mlx-build/libmlx.a"
    "$mlx_cmake_build/_deps/mlx-build/mlx/io/libgguflib.a"
    "$mlx_cmake_build/lib/libNumerics.a"
    "$mlx_cmake_build/lib/libComplexModule.a"
    "$mlx_cmake_build/lib/libRealModule.a"
  )

  for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "[prepare-linux-native] error: missing MLX Swift CUDA bridge artifact: $path" >&2
      missing=1
    fi
  done
  if [[ "$missing" == "1" ]]; then
    exit 69
  fi
}

verify_mlx_swift_patch() {
  if [[ "${MERERUN_SKIP_MLX_SWIFT_PATCH:-0}" == "1" || "$linux_accel" == "cuda" ]]; then
    return
  fi

  local package_file="$mlx_swift_checkout/Package.swift"
  if [[ ! -f "$package_file" ]]; then
    return
  fi
  if grep -Fq '"mlx-c/mlx/c/fast.cpp",  // Exclude on Linux - calls metal_kernel unconditionally' "$package_file" ||
     grep -Fq '"MLXFast.swift",' "$package_file" ||
     grep -Fq '"MLXFastKernel.swift",' "$package_file"; then
    echo "[prepare-linux-native] error: mlx-swift checkout still has the Linux MLXFast exclusions." >&2
    echo "[prepare-linux-native] rerun scripts/prepare-linux-native.sh before building." >&2
    exit 68
  fi
}

verify_llama() {
  require_tool pkg-config
  if [[ ! -f "$repo_root/Sources/llama/module.modulemap" ]]; then
    echo "[prepare-linux-native] error: Sources/llama/module.modulemap is missing." >&2
    exit 67
  fi

  export PKG_CONFIG_PATH="$pkgconfig_dir:${PKG_CONFIG_PATH:-}"
  if ! pkg-config --exists llama; then
    echo "[prepare-linux-native] error: pkg-config could not find llama." >&2
    echo "[prepare-linux-native] run scripts/prepare-linux-native.sh or export PKG_CONFIG_PATH=$pkgconfig_dir:\$PKG_CONFIG_PATH" >&2
    exit 67
  fi
}

stage_ds4

if [[ "$check_only" != "1" ]]; then
  build_llama
  smoke_mlx_swift_cuda
  patch_mlx_swift_for_linux
fi

verify_llama
verify_mlx_swift_patch
verify_mlx_swift_cuda_bridge

cat <<EOF
[prepare-linux-native] Linux-native layout ready for $platform_dir (accel: $linux_accel).

Export these before building/running the Linux CLI:
  export PKG_CONFIG_PATH="$pkgconfig_dir:\${PKG_CONFIG_PATH:-}"
  export LIBRARY_PATH="$llama_prefix/lib:\${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$llama_prefix/lib:\${LD_LIBRARY_PATH:-}"
EOF

if [[ "$linux_accel" == "cuda" ]]; then
  mlx_link_flags="$(mlx_swift_cuda_link_flags)"
  cat <<EOF
  export MERERUN_LINUX_ACCEL="cuda"
  export MERERUN_MLX_SWIFT_LINKAGE="cuda-prebuilt"
  export MERERUN_MLX_SWIFT_BUILD_DIR="$native_root/build/mlx-swift-cuda-smoke"
  export MERERUN_MLX_SWIFT_SOURCE_DIR="$repo_root/.build/native/src/mlx-swift"
  export MERERUN_MLX_SWIFT_LINK_FLAGS="$mlx_link_flags"
  # Optional: cap llama.cpp GPU offload layers instead of the default all-layers setting.
  # export MERERUN_LLAMA_GPU_LAYERS="999"
EOF
fi
