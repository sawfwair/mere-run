#!/usr/bin/env bash

linux_arm64_bf16_compiler_supports() {
  local compiler="$1"
  local tmpdir
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-bf16-compiler.XXXXXX")" || return 1
  local source="$tmpdir/bf16.cpp"
  cat >"$source" <<'CPP'
#include <arm_bf16.h>
#include <string>
bfloat16_t mererun_bf16_probe(bfloat16_t value) { return value; }
int main() { return std::string("bf16").empty() ? 1 : 0; }
CPP
  "$compiler" -x c++ "$source" -o "$tmpdir/bf16" >/dev/null 2>&1
  local status=$?
  rm -rf "$tmpdir"
  return "$status"
}

configure_linux_arm64_bf16_toolchain() {
  local arch="$1"
  local log_prefix="${2:-linux-arm64-bf16}"
  if [[ "$arch" != "arm64" ]]; then
    return 0
  fi

  if [[ -n "${CXX:-}" ]]; then
    if linux_arm64_bf16_compiler_supports "$CXX"; then
      return 0
    fi
    echo "[$log_prefix] error: CXX=$CXX cannot compile arm_bf16.h on Linux arm64." >&2
    echo "[$log_prefix] use a Swift toolchain C++ driver with __bf16 support, for example CXX=/usr/bin/clang++-17." >&2
    return 65
  fi

  local default_cxx
  default_cxx="$(command -v clang++ || true)"
  if [[ -n "$default_cxx" ]] && linux_arm64_bf16_compiler_supports "$default_cxx"; then
    return 0
  fi

  local candidate
  for candidate in clang++-18 clang++-17 /usr/bin/clang++-18 /usr/bin/clang++-17; do
    if ! command -v "$candidate" >/dev/null 2>&1 && [[ ! -x "$candidate" ]]; then
      continue
    fi
    if linux_arm64_bf16_compiler_supports "$candidate"; then
      export CXX="$candidate"
      if [[ -z "${CC:-}" ]]; then
        export CC="${candidate/++/}"
      fi
      echo "[$log_prefix] using CXX=$CXX for Linux arm64 MLX bf16 support." >&2
      return 0
    fi
  done

  local toolchain_dir="${MERERUN_BF16_TOOLCHAIN_DIR:-$PWD/.build/toolchains/linux-arm64-bf16}"
  for candidate in /usr/bin/clang-18 /usr/bin/clang-17 clang-18 clang-17; do
    if ! command -v "$candidate" >/dev/null 2>&1 && [[ ! -x "$candidate" ]]; then
      continue
    fi
    mkdir -p "$toolchain_dir"
    local compiler_path
    compiler_path="$(command -v "$candidate" || printf '%s' "$candidate")"
    ln -sf "$compiler_path" "$toolchain_dir/clang++"
    ln -sf "$compiler_path" "$toolchain_dir/clang"
    if linux_arm64_bf16_compiler_supports "$toolchain_dir/clang++"; then
      export CXX="$toolchain_dir/clang++"
      if [[ -z "${CC:-}" ]]; then
        export CC="$toolchain_dir/clang"
      fi
      echo "[$log_prefix] using CXX=$CXX for Linux arm64 MLX bf16 support." >&2
      return 0
    fi
  done

  echo "[$log_prefix] error: Linux arm64 MLX builds need a Clang that can compile arm_bf16.h." >&2
  echo "[$log_prefix] install or select the Swift toolchain Clang, then rerun with CXX=/path/to/clang++." >&2
  return 65
}
