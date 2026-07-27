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
  MERERUN_NATIVE_BUILD_JOBS   Override CMake build parallelism. Defaults to
                              nproc. Useful in containers where nproc reports
                              the physical host instead of the rented worker.
  MERERUN_CUDA_ARCHITECTURES  Optional CMake CUDA architecture list for llama.cpp
                              and mlx-swift CUDA builds, for example
                              "86-real;90-virtual".
  MLX_SWIFT_CUDA_COMMIT       Override the mlx-swift revision used by the CMake
                              CUDA bridge. Defaults to the exact revision
                              resolved by the active Linux Swift toolchain.
  MLX_SWIFT_CUDA_URL          Override the mlx-swift repository used by the
                              CMake CUDA bridge. Defaults to the pinned fork.
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
  MERERUN_SKIP_MLX_CUDA_EXAMPLE=1
                              Build the mlx-swift CUDA bridge but skip running
                              the GPU example. Use this for CPU-only release
                              builders with CUDA development packages.
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

native_build_jobs() {
  local jobs="${MERERUN_NATIVE_BUILD_JOBS:-${MERERUN_BUILD_JOBS:-}}"
  if [[ -z "$jobs" ]]; then
    jobs="$(nproc)"
  fi
  if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "[prepare-linux-native] error: MERERUN_NATIVE_BUILD_JOBS must be a positive integer." >&2
    exit 64
  fi
  printf '%s\n' "$jobs"
}

require_cmake_at_least() {
  local minimum_major="$1"
  local minimum_minor="$2"
  local minimum_patch="$3"
  require_tool cmake

  local version
  version="$(cmake --version | awk '/^cmake version / { print $3; exit }')"
  local major minor patch
  IFS=. read -r major minor patch <<<"$version"
  patch="${patch%%[^0-9]*}"
  major="${major:-0}"
  minor="${minor:-0}"
  patch="${patch:-0}"

  if ! [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ && "$patch" =~ ^[0-9]+$ ]]; then
    echo "[prepare-linux-native] error: could not parse cmake version: $version" >&2
    exit 64
  fi
  if (( major > minimum_major )); then
    return
  fi
  if (( major == minimum_major && minor > minimum_minor )); then
    return
  fi
  if (( major == minimum_major && minor == minimum_minor && patch >= minimum_patch )); then
    return
  fi

  echo "[prepare-linux-native] error: MERERUN_LINUX_ACCEL=cuda requires cmake >= ${minimum_major}.${minimum_minor}.${minimum_patch}; found $version." >&2
  exit 64
}

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

cuda_cccl_include_candidates() {
  local cuda_root
  while IFS= read -r cuda_root; do
    printf '%s\n' "$cuda_root/include/cccl"
    local cuda_target
    for cuda_target in "${cuda_target_names[@]}"; do
      printf '%s\n' "$cuda_root/targets/$cuda_target/include/cccl"
    done
  done < <(cuda_toolkit_root_candidates)
  printf '%s\n' /usr/include/cccl
}

detect_cuda_cccl_include_path() {
  local candidate
  while IFS= read -r candidate; do
    if [[ -d "$candidate/cuda/std" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done < <(cuda_cccl_include_candidates)
}

detect_cuda_toolkit_defaults() {
  if [[ -z "${CUDA_HOME:-}" && -z "${CUDA_PATH:-}" ]]; then
    local cuda_root
    while IFS= read -r cuda_root; do
      if [[ -x "$cuda_root/bin/nvcc" || -f "$cuda_root/include/cuda.h" ]]; then
        export CUDA_HOME="$cuda_root"
        break
      fi
      local cuda_target
      for cuda_target in "${cuda_target_names[@]}"; do
        if [[ -f "$cuda_root/targets/$cuda_target/include/cuda.h" ]]; then
          export CUDA_HOME="$cuda_root"
          break 2
        fi
      done
    done < <(cuda_toolkit_root_candidates)
  fi
  if [[ -z "${CUDA_PATH:-}" && -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="$CUDA_HOME"
  fi
}

patch_mlx_cuda_jit_include_path() {
  local mlx_jit_module="$1"
  if [[ ! -f "$mlx_jit_module" ]]; then
    return
  fi
  if grep -Fq "MERERUN_MLX_CUDA_JIT_INCLUDE_PATH" "$mlx_jit_module"; then
    return
  fi

  local mlx_jit_tmp
  mlx_jit_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-mlx-jit.XXXXXX")"
  if ! awk '
    /return args;/ && !inserted {
      print "      // MERERUN_MLX_CUDA_JIT_INCLUDE: packaged MLX CUDA kernels need"
      print "      // their matching CUTLASS/CuTe headers at NVRTC compile time."
      print "      if (auto mererun_mlx_cuda_jit = std::getenv(\"MERERUN_MLX_CUDA_JIT_INCLUDE_PATH\")) {"
      print "        auto mererun_mlx_cuda_jit_path = std::filesystem::path(mererun_mlx_cuda_jit);"
      print "        if (std::filesystem::exists(mererun_mlx_cuda_jit_path / \"cute\") &&"
      print "            std::filesystem::exists(mererun_mlx_cuda_jit_path / \"cutlass\")) {"
      print "          args.push_back(fmt::format(\"--include-path={}\", mererun_mlx_cuda_jit_path.string()));"
      print "        }"
      print "      }"
      print "      // MERERUN_CUDA_CCCL_INCLUDE: CUDA 13 installs cuda/std under include/cccl."
      print "      // Keep this explicit because NVRTC does not inherit the package launcher CPATH."
      print "      auto add_mererun_cuda_cccl_include = [&](const std::filesystem::path& cuda_cccl_path) {"
      print "        if (!cuda_cccl_path.empty() &&"
      print "            std::filesystem::exists(cuda_cccl_path / \"cuda\" / \"std\")) {"
      print "          args.push_back(fmt::format(\"--include-path={}\", cuda_cccl_path.string()));"
      print "        }"
      print "      };"
      print "      if (auto mererun_cuda_cccl = std::getenv(\"MERERUN_CUDA_CCCL_INCLUDE_PATH\")) {"
      print "        add_mererun_cuda_cccl_include(std::filesystem::path(mererun_cuda_cccl));"
      print "      }"
      print "      auto mererun_cuda_home = []() -> std::filesystem::path {"
      print "        if (auto home = std::getenv(\"CUDA_HOME\")) { return home; }"
      print "        if (auto path = std::getenv(\"CUDA_PATH\")) { return path; }"
      print "        return std::filesystem::path(\"/usr/local/cuda\");"
      print "      }();"
      print "      add_mererun_cuda_cccl_include(mererun_cuda_home / \"include\" / \"cccl\");"
      print "      add_mererun_cuda_cccl_include(mererun_cuda_home / \"targets\" / \"sbsa-linux\" / \"include\" / \"cccl\");"
      print "      add_mererun_cuda_cccl_include(mererun_cuda_home / \"targets\" / \"aarch64-linux\" / \"include\" / \"cccl\");"
      print "      add_mererun_cuda_cccl_include(mererun_cuda_home / \"targets\" / \"x86_64-linux\" / \"include\" / \"cccl\");"
      print "      add_mererun_cuda_cccl_include(std::filesystem::path(\"/usr/include/cccl\"));"
      print "      add_mererun_cuda_cccl_include(std::filesystem::path(\"/usr/local/cuda/include/cccl\"));"
      print "      add_mererun_cuda_cccl_include(std::filesystem::path(\"/usr/local/cuda/targets/sbsa-linux/include/cccl\"));"
      print "      add_mererun_cuda_cccl_include(std::filesystem::path(\"/usr/local/cuda/targets/aarch64-linux/include/cccl\"));"
      print "      add_mererun_cuda_cccl_include(std::filesystem::path(\"/usr/local/cuda/targets/x86_64-linux/include/cccl\"));"
      inserted=1
    }
    { print }
    END { if (!inserted) exit 42 }
  ' "$mlx_jit_module" >"$mlx_jit_tmp"; then
    rm -f "$mlx_jit_tmp"
    echo "[prepare-linux-native] error: could not patch MLX CUDA JIT include path in ${mlx_jit_module#$repo_root/}." >&2
    exit 69
  fi
  mv "$mlx_jit_tmp" "$mlx_jit_module"
  echo "[prepare-linux-native] patched mlx-swift CUDA JIT include path in ${mlx_jit_module#$repo_root/}."
}

patch_mlx_cuda_bf16_sigmoid() {
  local mlx_unary_ops="$1"
  if [[ ! -f "$mlx_unary_ops" ]]; then
    return
  fi
  if grep -Fq "MERERUN_CUDA_BF16_SIGMOID_ABS" "$mlx_unary_ops"; then
    return
  fi

  local mlx_unary_tmp
  mlx_unary_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-mlx-unary.XXXXXX")"
  if ! awk '
    /T y = 1 \/ \(1 \+ cuda::std::exp\(cuda::std::abs\(x\)\)\);/ && !patched {
      print "    T y;"
      print "    if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {"
      print "      // MERERUN_CUDA_BF16_SIGMOID_ABS: CUDA 12.x cannot resolve"
      print "      // cuda::std::abs(__nv_bfloat16) in NVRTC JIT kernels."
      print "      float abs_x = cuda::std::abs(static_cast<float>(x));"
      print "      y = static_cast<T>(1.0f / (1.0f + cuda::std::exp(abs_x)));"
      print "    } else {"
      print "      y = 1 / (1 + cuda::std::exp(cuda::std::abs(x)));"
      print "    }"
      patched=1
      next
    }
    { print }
    END { if (!patched) exit 42 }
  ' "$mlx_unary_ops" >"$mlx_unary_tmp"; then
    rm -f "$mlx_unary_tmp"
    echo "[prepare-linux-native] error: could not patch MLX CUDA bf16 sigmoid in ${mlx_unary_ops#$repo_root/}." >&2
    exit 69
  fi
  mv "$mlx_unary_tmp" "$mlx_unary_ops"
  echo "[prepare-linux-native] patched mlx-swift CUDA bf16 sigmoid in ${mlx_unary_ops#$repo_root/}."
}

patch_mlx_cuda_bf16_power() {
  local mlx_binary_ops="$1"
  if [[ ! -f "$mlx_binary_ops" ]]; then
    return
  fi
  if grep -Fq "MERERUN_CUDA_BF16_POWER" "$mlx_binary_ops"; then
    return
  fi

  local mlx_binary_tmp
  mlx_binary_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-mlx-binary.XXXXXX")"
  if ! awk '
    $0 == "    } else if constexpr (is_complex_v<T>) {" && !patched {
      line1 = $0
      if ((getline line2) <= 0 || (getline line3) <= 0 || (getline line4) <= 0) {
        print line1
        if (line2 != "") print line2
        if (line3 != "") print line3
        if (line4 != "") print line4
        next
      }
      if (line2 == "      return cuda::std::pow(base, exp);" &&
          line3 == "    } else {" &&
          line4 == "      return cuda::std::pow(base, exp);") {
        print line1
        print line2
        print "    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {"
        print "      // MERERUN_CUDA_BF16_POWER: CUDA 12.x cannot resolve"
        print "      // cuda::std::pow(__nv_bfloat16, __nv_bfloat16) in NVRTC JIT kernels."
        print "      float base_f = static_cast<float>(base);"
        print "      float exp_f = static_cast<float>(exp);"
        print "      return static_cast<T>(cuda::std::pow(base_f, exp_f));"
        print "    } else {"
        print line4
        patched=1
        next
      }
      print line1
      print line2
      print line3
      print line4
      next
    }
    { print }
    END { if (!patched) exit 42 }
  ' "$mlx_binary_ops" >"$mlx_binary_tmp"; then
    rm -f "$mlx_binary_tmp"
    echo "[prepare-linux-native] error: could not patch MLX CUDA bf16 power in ${mlx_binary_ops#$repo_root/}." >&2
    exit 69
  fi
  mv "$mlx_binary_tmp" "$mlx_binary_ops"
  echo "[prepare-linux-native] patched mlx-swift CUDA bf16 power in ${mlx_binary_ops#$repo_root/}."
}

patch_mlx_cpu_jit_f16c_probe() {
  local mlx_jit_compiler="$1"
  if [[ ! -f "$mlx_jit_compiler" ]]; then
    return
  fi
  if grep -Fq "MERERUN_MLX_X86_AVX2_IMPLIES_F16C" "$mlx_jit_compiler"; then
    return
  fi

  local mlx_jit_tmp
  mlx_jit_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-mlx-cpu-jit.XXXXXX")"
  if ! awk '
    /return __builtin_cpu_supports\("avx2"\) && __builtin_cpu_supports\("fma"\) &&/ && !patched {
      line1 = $0
      if ((getline line2) <= 0) {
        print line1
        next
      }
      if (line2 == "      __builtin_cpu_supports(\"f16c\");") {
        print "  // MERERUN_MLX_X86_AVX2_IMPLIES_F16C: the Clang shipped with the"
        print "  // Swift 6.0 Jammy image rejects f16c as a builtin feature string."
        print "  // Every production x86 CPU implementing AVX2 also implements F16C."
        print "  return __builtin_cpu_supports(\"avx2\") && __builtin_cpu_supports(\"fma\");"
        patched=1
        next
      }
      print line1
      print line2
      next
    }
    { print }
    END { if (!patched) exit 42 }
  ' "$mlx_jit_compiler" >"$mlx_jit_tmp"; then
    rm -f "$mlx_jit_tmp"
    echo "[prepare-linux-native] error: could not patch MLX x86 F16C probe in ${mlx_jit_compiler#$repo_root/}." >&2
    exit 69
  fi
  mv "$mlx_jit_tmp" "$mlx_jit_compiler"
  echo "[prepare-linux-native] patched MLX x86 F16C probe in ${mlx_jit_compiler#$repo_root/}."
}

detect_cuda_dependency_defaults() {
  if [[ "$linux_accel" != "cuda" ]]; then
    return
  fi

  if [[ -z "${CUDNN_INCLUDE_PATH:-}" ]]; then
    local cudnn_include_candidates=("/usr/include/$deb_multiarch")
    local cuda_root
    while IFS= read -r cuda_root; do
      cudnn_include_candidates+=("$cuda_root/include")
      local cuda_target
      for cuda_target in "${cuda_target_names[@]}"; do
        cudnn_include_candidates+=("$cuda_root/targets/$cuda_target/include")
      done
    done < <(cuda_toolkit_root_candidates)
    for candidate in "${cudnn_include_candidates[@]}"; do
      if [[ -f "$candidate/cudnn.h" ]]; then
        export CUDNN_INCLUDE_PATH="$candidate"
        break
      fi
    done
  fi

  if [[ -z "${CUDNN_LIBRARY_PATH:-}" ]]; then
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
        export CUDNN_LIBRARY_PATH="$candidate"
        break
      fi
    done
  fi

  if [[ -z "${MERERUN_CUDA_CCCL_INCLUDE_PATH:-}" ]]; then
    local cuda_cccl_include_path
    cuda_cccl_include_path="$(detect_cuda_cccl_include_path || true)"
    if [[ -n "$cuda_cccl_include_path" ]]; then
      export MERERUN_CUDA_CCCL_INCLUDE_PATH="$cuda_cccl_include_path"
    fi
  fi
}

platform_dir="linux-$arch"
native_root="$repo_root/.build/native/$platform_dir"
llama_prefix="$native_root/llama"
pkgconfig_dir="$native_root/pkgconfig"
llama_pc="$pkgconfig_dir/llama.pc"
llama_src="$repo_root/.build/native/src/llama.cpp"
llama_build="$native_root/build/llama.cpp"
llama_commit="${LLAMA_CPP_COMMIT:-4988f6e866057afd130c1515ecef0c9bab9a15f8}"
llama_url="${LLAMA_CPP_URL:-https://github.com/ggml-org/llama.cpp.git}"
mlx_swift_checkout="$repo_root/.build/checkouts/mlx-swift"
swift_numerics_checkout="$repo_root/.build/checkouts/swift-numerics"

resolved_mlx_swift_revision() {
  awk '
    /"identity"[[:space:]]*:[[:space:]]*"mlx-swift"/ { found = 1 }
    found && /"revision"[[:space:]]*:/ {
      line = $0
      sub(/^.*"revision"[[:space:]]*:[[:space:]]*"/, "", line)
      sub(/".*$/, "", line)
      print line
      exit
    }
  ' "$repo_root/Package.resolved"
}

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
  if [[ "$linux_accel" == "cuda" ]]; then
    require_cmake_at_least 3 25 0
    # llama.cpp is configured before the mlx-swift CUDA smoke. Resolve the
    # toolkit now so CMake can find installations such as
    # /usr/local/cuda-12.4/targets/<arch>/include instead of waiting until the
    # later MLX phase to populate CUDA_HOME/CUDA_PATH.
    detect_cuda_toolkit_defaults
  fi
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
    -DLLAMA_BUILD_COMMON=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_APP=OFF
    -DLLAMA_CURL=OFF
  )

  if [[ "$linux_accel" == "cuda" ]]; then
    require_tool nvcc
    echo "[prepare-linux-native] enabling llama.cpp CUDA via GGML_CUDA=ON"
    local cuda_compiler
    cuda_compiler="$(command -v nvcc)"
    cmake_args+=(
      -DGGML_CUDA=ON
      "-DCMAKE_CUDA_COMPILER=$cuda_compiler"
    )
    if [[ -n "${CUDA_HOME:-}" ]]; then
      cmake_args+=(
        "-DCUDAToolkit_ROOT=$CUDA_HOME"
      )
    fi
    if [[ -n "${MERERUN_CUDA_ARCHITECTURES:-}" ]]; then
      echo "[prepare-linux-native] using CUDA architectures: $MERERUN_CUDA_ARCHITECTURES"
      cmake_args+=(
        "-DCMAKE_CUDA_ARCHITECTURES=$MERERUN_CUDA_ARCHITECTURES"
      )
    fi
  fi

  echo "[prepare-linux-native] configuring llama.cpp"
  cmake -S "$llama_src" -B "$llama_build" "${cmake_args[@]}"

  echo "[prepare-linux-native] building llama.cpp"
  local build_jobs
  build_jobs="$(native_build_jobs)"
  echo "[prepare-linux-native] using $build_jobs native build jobs"
  cmake --build "$llama_build" --config Release --parallel "$build_jobs"

  echo "[prepare-linux-native] installing llama.cpp into $llama_prefix"
  cmake --install "$llama_build"
  write_llama_pc
}

patch_mlx_swift_for_linux() {
  if [[ "${MERERUN_SKIP_MLX_SWIFT_PATCH:-0}" == "1" ]]; then
    echo "[prepare-linux-native] skipping SwiftPM Linux package fixes by request."
    return
  fi

  require_tool swift
  echo "[prepare-linux-native] resolving Swift package dependencies"
  swift package resolve

  patch_mlx_cpu_jit_f16c_probe \
    "$mlx_swift_checkout/Source/Cmlx/mlx/mlx/backend/cpu/jit_compiler.cpp"

  if [[ "$linux_accel" == "cuda" ]]; then
    patch_mlx_cuda_jit_include_path "$mlx_swift_checkout/Source/Cmlx/mlx/mlx/backend/cuda/jit_module.cpp"
    patch_mlx_cuda_bf16_sigmoid "$mlx_swift_checkout/Source/Cmlx/mlx/mlx/backend/cuda/device/unary_ops.cuh"
    patch_mlx_cuda_bf16_power "$mlx_swift_checkout/Source/Cmlx/mlx/mlx/backend/cuda/device/binary_ops.cuh"
    echo "[prepare-linux-native] skipping mlx-swift SwiftPM package fix; CUDA builds use the CMake prebuilt bridge."
  else
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
  fi

  local numerics_header="$swift_numerics_checkout/Sources/_NumericsShims/include/_NumericsShims.h"
  local numerics_float16="$swift_numerics_checkout/Sources/RealModule/Float16+Real.swift"
  if [[ -f "$numerics_header" ]] && grep -Fq '#if !defined __wasm__ // No _Float16 on wasm' "$numerics_header"; then
    sed -i 's/#if !defined __wasm__ \/\/ No _Float16 on wasm/#if !defined __wasm__ \&\& defined(__FLT16_MANT_DIG__) \/\/ No _Float16 on wasm, or on targets whose C compiler lacks _Float16/' "$numerics_header"
  fi
  if [[ -f "$numerics_float16" ]] && grep -Fq '#if !arch(wasm32)' "$numerics_float16"; then
    sed -i 's/#if !arch(wasm32)/#if !arch(wasm32) \&\& !(os(Linux) \&\& arch(x86_64))/' "$numerics_float16"
  fi

  echo "[prepare-linux-native] SwiftPM Linux package fixes applied."
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
  require_tool swiftc

  detect_cuda_toolkit_defaults

  local mlx_cmake_src="$repo_root/.build/native/src/mlx-swift"
  local mlx_cmake_build="$native_root/build/mlx-swift-cuda-smoke"
  local mlx_swift_cuda_commit="${MLX_SWIFT_CUDA_COMMIT:-}"
  local mlx_swift_cuda_url="${MLX_SWIFT_CUDA_URL:-https://github.com/sawfwair/mlx-swift.git}"
  if [[ -z "$mlx_swift_cuda_commit" && -d "$mlx_swift_checkout/.git" ]]; then
    mlx_swift_cuda_commit="$(git -C "$mlx_swift_checkout" rev-parse HEAD)"
  fi
  if [[ -z "$mlx_swift_cuda_commit" ]]; then
    mlx_swift_cuda_commit="$(resolved_mlx_swift_revision)"
  fi
  if [[ -z "$mlx_swift_cuda_commit" ]]; then
    echo "[prepare-linux-native] error: could not resolve the Linux mlx-swift revision." >&2
    exit 68
  fi

  if [[ ! -d "$mlx_cmake_src/.git" ]]; then
    echo "[prepare-linux-native] cloning mlx-swift for CMake CUDA smoke"
    git clone --filter=blob:none "$mlx_swift_cuda_url" "$mlx_cmake_src"
  fi

  echo "[prepare-linux-native] checking out mlx-swift $mlx_swift_cuda_commit for the CMake CUDA bridge"
  git -C "$mlx_cmake_src" remote set-url origin "$mlx_swift_cuda_url"
  git -C "$mlx_cmake_src" fetch --depth 1 origin "$mlx_swift_cuda_commit"
  git -C "$mlx_cmake_src" checkout --detach "$mlx_swift_cuda_commit"
  git -C "$mlx_cmake_src" submodule sync --recursive
  git -C "$mlx_cmake_src" submodule update --init --depth 1 \
    Source/Cmlx/mlx Source/Cmlx/mlx-c

  patch_mlx_cuda_jit_include_path "$mlx_cmake_src/Source/Cmlx/mlx/mlx/backend/cuda/jit_module.cpp"
  patch_mlx_cuda_jit_include_path "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/jit_module.cpp"
  patch_mlx_cuda_jit_include_path "$mlx_cmake_build/_deps/mlx-c-src/mlx/backend/cuda/jit_module.cpp"
  patch_mlx_cpu_jit_f16c_probe "$mlx_cmake_src/Source/Cmlx/mlx/mlx/backend/cpu/jit_compiler.cpp"
  patch_mlx_cpu_jit_f16c_probe "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cpu/jit_compiler.cpp"
  patch_mlx_cpu_jit_f16c_probe "$mlx_cmake_build/_deps/mlx-c-src/mlx/backend/cpu/jit_compiler.cpp"
  patch_mlx_cuda_bf16_sigmoid "$mlx_cmake_src/Source/Cmlx/mlx/mlx/backend/cuda/device/unary_ops.cuh"
  patch_mlx_cuda_bf16_sigmoid "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/device/unary_ops.cuh"
  patch_mlx_cuda_bf16_sigmoid "$mlx_cmake_build/_deps/mlx-c-src/mlx/backend/cuda/device/unary_ops.cuh"
  patch_mlx_cuda_bf16_power "$mlx_cmake_src/Source/Cmlx/mlx/mlx/backend/cuda/device/binary_ops.cuh"
  patch_mlx_cuda_bf16_power "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/device/binary_ops.cuh"
  patch_mlx_cuda_bf16_power "$mlx_cmake_build/_deps/mlx-c-src/mlx/backend/cuda/device/binary_ops.cuh"

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
    "-DCMAKE_CUDA_COMPILER=$(command -v nvcc)"
    "-DCMAKE_Swift_COMPILER=$(command -v swiftc)"
    -DMLX_BUILD_METAL=OFF
    -DMLX_BUILD_CUDA=ON
    -DMLX_C_BUILD_EXAMPLES=OFF
  )
  if [[ -n "${CUDA_HOME:-}" ]]; then
    mlx_cmake_args+=("-DCUDAToolkit_ROOT=$CUDA_HOME")
  fi
  if [[ -n "${MERERUN_CUDA_ARCHITECTURES:-}" ]]; then
    echo "[prepare-linux-native] using CUDA architectures: $MERERUN_CUDA_ARCHITECTURES"
    mlx_cmake_args+=(
      "-DCMAKE_CUDA_ARCHITECTURES=$MERERUN_CUDA_ARCHITECTURES"
      "-DMLX_CUDA_ARCHITECTURES=$MERERUN_CUDA_ARCHITECTURES"
    )
  fi
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
  patch_mlx_cuda_jit_include_path "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/jit_module.cpp"
  patch_mlx_cuda_jit_include_path "$mlx_cmake_build/_deps/mlx-c-src/mlx/backend/cuda/jit_module.cpp"
  patch_mlx_cuda_bf16_sigmoid "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/device/unary_ops.cuh"
  patch_mlx_cuda_bf16_sigmoid "$mlx_cmake_build/_deps/mlx-c-src/mlx/backend/cuda/device/unary_ops.cuh"
  patch_mlx_cuda_bf16_power "$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/device/binary_ops.cuh"
  patch_mlx_cuda_bf16_power "$mlx_cmake_build/_deps/mlx-c-src/mlx/backend/cuda/device/binary_ops.cuh"

  echo "[prepare-linux-native] building mlx-swift CUDA smoke"
  local build_jobs
  build_jobs="$(native_build_jobs)"
  echo "[prepare-linux-native] using $build_jobs native build jobs"
  cmake --build "$mlx_cmake_build" --parallel "$build_jobs"

  local example="$mlx_cmake_build/example1"
  if [[ -x "$example" ]]; then
    if [[ "${MERERUN_SKIP_MLX_CUDA_EXAMPLE:-0}" == "1" ]]; then
      echo "[prepare-linux-native] skipping mlx-swift CUDA GPU example by request."
    else
      echo "[prepare-linux-native] running mlx-swift CUDA smoke example"
      "$example" --device gpu
    fi
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
  patch_mlx_swift_for_linux
  # Resolve SwiftPM first: older Linux Swift toolchains may select an older
  # compatible mlx-swift release than a lockfile written by a newer macOS
  # toolchain. The CMake CUDA bridge must use that exact resolved checkout to
  # keep the Swift and native ABIs aligned.
  smoke_mlx_swift_cuda
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
