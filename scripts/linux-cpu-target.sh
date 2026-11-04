#!/usr/bin/env bash

# CI uses fixed instruction sets for native libraries and SwiftPM. Ordinary
# Linux builds retain their existing compiler and llama.cpp defaults.
configure_linux_cpu_target() {
  local arch="$1"
  local accel="$2"
  linux_cpu_target="${MERERUN_LINUX_CPU_TARGET:-native}"
  linux_cpu_cmake_flags=()
  linux_cpu_swift_flags=()
  case "$linux_cpu_target" in
    native) return 0 ;;
    x86-64-v3)
      if [[ "$arch" != x86_64 || "$accel" != cpu ]]; then
        echo '[linux-cpu-target] x86-64-v3 requires an x86_64 CPU build.' >&2
        return 64
      fi
      ;;
    *)
      echo "[linux-cpu-target] unsupported MERERUN_LINUX_CPU_TARGET=$linux_cpu_target" >&2
      return 64
      ;;
  esac

  linux_cpu_cmake_flags=(
    -DGGML_NATIVE=OFF
    -DGGML_CPU_ALL_VARIANTS=OFF
    '-DCMAKE_C_FLAGS=-march=x86-64-v3 -mtune=generic'
    '-DCMAKE_CXX_FLAGS=-march=x86-64-v3 -mtune=generic'
    -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON
    -DGGML_BMI2=ON -DGGML_F16C=ON -DGGML_FMA=ON
    -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF
    -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF
    -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF
  )
  linux_cpu_swift_flags=(
    # Swift 6.1's libstdc++ importer cycles through std -> _Builtin_intrinsics
    # with SSE3 enabled. Keep SwiftPM (including C/C++) at generic x86-64;
    # llama's separately built libraries retain the fixed v3 target above.
    # Swift code generation inherits Clang's target info through -Xcc.
    -Xcc -march=x86-64 -Xcc -mtune=generic
    -Xcxx -march=x86-64 -Xcxx -mtune=generic
  )
}

verify_linux_cpu_target() {
  [[ "$linux_cpu_target" != native ]] || return 0
  local probe_dir
  probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-cpu-target.XXXXXX")"
  # Compile detection at the generic baseline before executing any cached v3
  # binaries. GCC 11 exposes these individual features, but not the v3 alias.
  cat > "$probe_dir/probe.c" <<'C'
int main(void) {
  __builtin_cpu_init();
  return !(__builtin_cpu_supports("sse3") && __builtin_cpu_supports("ssse3") &&
    __builtin_cpu_supports("cmpxchg16b") && __builtin_cpu_supports("lahf_lm") &&
    __builtin_cpu_supports("popcnt") && __builtin_cpu_supports("sse4.1") &&
    __builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("avx") &&
    __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi") &&
    __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("f16c") &&
    __builtin_cpu_supports("fma") && __builtin_cpu_supports("lzcnt") &&
    __builtin_cpu_supports("movbe") && __builtin_cpu_supports("xsave"));
}
C
  local status=0
  if cc -march=x86-64 -mtune=generic "$probe_dir/probe.c" -o "$probe_dir/probe" \
      && "$probe_dir/probe"; then
    echo "[linux-cpu-target] verified $linux_cpu_target support."
  else
    echo "[linux-cpu-target] this compiler and runner must support $linux_cpu_target." >&2
    status=65
  fi
  rm -rf "$probe_dir"
  return "$status"
}
