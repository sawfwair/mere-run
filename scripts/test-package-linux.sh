#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-package-test.XXXXXX")"
cleanup() {
  rm -rf "$fixture_root"
  if [[ -n "${output_dir:-}" ]]; then
    rm -rf "$output_dir"
  fi
}
trap cleanup EXIT

fake_bin="$fixture_root/bin"
fake_build="$fixture_root/build"
fake_libs="$fixture_root/libs"
mkdir -p "$fake_bin" "$fake_build" "$fake_libs/real" "$fake_libs/alternatives"
mkdir -p "$fake_build/MereRun_MereRunCLI.resources/Guides"
mlx_cuda_jit_fixture="$fixture_root/cutlass/include"
mkdir -p "$mlx_cuda_jit_fixture/cute/numeric" "$mlx_cuda_jit_fixture/cutlass"
printf '// CuTe package fixture\n' >"$mlx_cuda_jit_fixture/cute/numeric/numeric_types.hpp"
printf '// CUTLASS package fixture\n' >"$mlx_cuda_jit_fixture/cutlass/cutlass.h"
printf 'CUTLASS BSD-3-Clause package fixture\n' >"$(dirname "$mlx_cuda_jit_fixture")/LICENSE.txt"
export MERERUN_MLX_CUDA_JIT_INCLUDE_DIR="$mlx_cuda_jit_fixture"

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
    echo "[test-package-linux] unsupported Linux architecture: $(uname -m)" >&2
    exit 65
    ;;
esac

output_dir="$fixture_root/output"
mkdir -p "$output_dir"
printf 'stale checksum manifest\n' >"$output_dir/SHA256SUMS"

cat >"$fake_build/mere.run" <<'CLI'
#!/usr/bin/env bash
if [[ "${1:-}" == "__print-env" ]]; then
  printf 'MERERUN_CUDA_CCCL_INCLUDE_PATH=%s\n' "${MERERUN_CUDA_CCCL_INCLUDE_PATH:-}"
  printf 'MERERUN_MLX_CUDA_JIT_INCLUDE_PATH=%s\n' "${MERERUN_MLX_CUDA_JIT_INCLUDE_PATH:-}"
  printf 'CPATH=%s\n' "${CPATH:-}"
  exit 0
fi
echo "mere.run package fixture"
CLI
chmod +x "$fake_build/mere.run"
printf '# Text Chat\n' >"$fake_build/MereRun_MereRunCLI.resources/Guides/text-chat.md"

printf 'openblas runtime fixture\n' >"$fake_libs/real/libopenblas.so.0.3.26"
ln -s "$fake_libs/real/libopenblas.so.0.3.26" "$fake_libs/alternatives/libopenblas.so.0-${platform_arch}-linux-gnu"
ln -s "$fake_libs/alternatives/libopenblas.so.0-${platform_arch}-linux-gnu" "$fake_libs/libopenblas.so.0"

cat >"$fake_bin/swift" <<SWIFT
#!/usr/bin/env bash
if [[ "\$*" == *"--show-bin-path"* ]]; then
  printf '%s\n' '$fake_build'
  exit 0
fi
echo "unexpected swift invocation: \$*" >&2
exit 64
SWIFT
chmod +x "$fake_bin/swift"

cat >"$fake_bin/ldd" <<LDD
#!/usr/bin/env bash
cat <<'EOF'
	libopenblas.so.0 => $fake_libs/libopenblas.so.0 (0x0000000000000000)
EOF
if [[ -n "\${FAKE_CUDA_MAJOR:-}" ]]; then
  printf '\tlibcudart.so.%s => /usr/local/cuda/lib64/libcudart.so.%s (0x0000000000000000)\n' \
    "\$FAKE_CUDA_MAJOR" "\$FAKE_CUDA_MAJOR"
fi
LDD
chmod +x "$fake_bin/ldd"

PATH="$fake_bin:$PATH" \
  MERERUN_BUNDLE_SWIFT_LIBS=1 \
  MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE=1 \
  bash scripts/package-linux.sh \
    --version symlink-fixture \
    --configuration release \
    --skip-build \
    --skip-native \
    --skip-deb \
    --output-dir "$output_dir" >/dev/null

tarball="$output_dir/mere-run-symlink-fixture-linux-${platform_arch}.tar.gz"
[[ -f "$tarball" ]]
[[ -f "$output_dir/SHA256SUMS" ]]
if grep -q '/' "$output_dir/SHA256SUMS"; then
  echo "[test-package-linux] SHA256SUMS should contain artifact basenames, not output-dir paths:" >&2
  cat "$output_dir/SHA256SUMS" >&2
  exit 1
fi
(cd "$output_dir" && sha256sum -c SHA256SUMS >/dev/null)

tar -xzf "$tarball" -C "$fixture_root"
payload_dir="$fixture_root/mere-run-symlink-fixture-linux-${platform_arch}"
staged_lib="$payload_dir/lib/libopenblas.so.0"
staged_guide="$payload_dir/MereRun_MereRunCLI.resources/Guides/text-chat.md"

if [[ ! -f "$staged_guide" ]]; then
  echo "[test-package-linux] expected SwiftPM CLI resources to be bundled at $staged_guide" >&2
  find "$payload_dir" -maxdepth 3 \( -type f -o -type d \) -print >&2
  exit 1
fi

if [[ ! -f "$staged_lib" || -L "$staged_lib" ]]; then
  echo "[test-package-linux] expected libopenblas.so.0 to be bundled as a real file, got:" >&2
  ls -l "$staged_lib" >&2 || true
  exit 1
fi

if ! grep -q 'openblas runtime fixture' "$staged_lib"; then
  echo "[test-package-linux] bundled libopenblas.so.0 did not contain the resolved library contents" >&2
  exit 1
fi

cuda_home_fixture="$fixture_root/cuda-13.0"
cuda_cccl_fixture="$cuda_home_fixture/targets/sbsa-linux/include/cccl"
mkdir -p "$cuda_cccl_fixture/cuda/std"
wrapper_env_output="$(CUDA_HOME="$cuda_home_fixture" "$payload_dir/mere.run" __print-env)"
if ! grep -q "^MERERUN_CUDA_CCCL_INCLUDE_PATH=$cuda_cccl_fixture$" <<<"$wrapper_env_output"; then
  echo "[test-package-linux] launcher did not export the target-specific CUDA CCCL include root:" >&2
  printf '%s\n' "$wrapper_env_output" >&2
  exit 1
fi
if ! grep -q "^CPATH=$cuda_cccl_fixture" <<<"$wrapper_env_output"; then
  echo "[test-package-linux] launcher did not add the CUDA CCCL include root to CPATH:" >&2
  printf '%s\n' "$wrapper_env_output" >&2
  exit 1
fi

echo "[test-package-linux] package runtime library symlink test passed."

cuda_output_dir="$fixture_root/output-cuda"
PATH="$fake_bin:$PATH" \
  FAKE_CUDA_MAJOR=13 \
  MERERUN_LINUX_ACCEL=cuda \
  MERERUN_MLX_SWIFT_LINK_FLAGS="-lcuda" \
  bash scripts/package-linux.sh \
    --version 0.0.0+cuda-deps-fixture \
    --artifact-suffix cuda \
    --configuration release \
    --skip-build \
    --skip-native \
    --output-dir "$cuda_output_dir" >/dev/null

cuda_tarball="$cuda_output_dir/mere-run-0.0.0+cuda-deps-fixture-linux-${platform_arch}-cuda.tar.gz"
cuda_deb="$cuda_output_dir/mere-run-cuda_0.0.0+cuda-deps-fixture_${deb_arch}.deb"
[[ -f "$cuda_tarball" ]]
[[ -f "$cuda_deb" ]]
cuda_tarball_listing="$fixture_root/cuda-tarball-listing.txt"
tar -tzf "$cuda_tarball" >"$cuda_tarball_listing"
grep -q "/.mererun-linux-cuda$" "$cuda_tarball_listing"
grep -q "/include/cute/numeric/numeric_types.hpp$" "$cuda_tarball_listing"
grep -q "/include/cutlass/cutlass.h$" "$cuda_tarball_listing"
grep -q "/include/CUTLASS-LICENSE.txt$" "$cuda_tarball_listing"
cuda_payload_root="$fixture_root/cuda-payload"
mkdir -p "$cuda_payload_root"
tar -xzf "$cuda_tarball" -C "$cuda_payload_root"
cuda_payload_dir="$cuda_payload_root/mere-run-0.0.0+cuda-deps-fixture-linux-${platform_arch}-cuda"
cuda_wrapper_env_output="$(CUDA_HOME="$cuda_home_fixture" "$cuda_payload_dir/mere.run" __print-env)"
if ! grep -q "^MERERUN_MLX_CUDA_JIT_INCLUDE_PATH=$cuda_payload_dir/include$" <<<"$cuda_wrapper_env_output"; then
  echo "[test-package-linux] launcher did not export the bundled MLX CUDA JIT header root:" >&2
  printf '%s\n' "$cuda_wrapper_env_output" >&2
  exit 1
fi
if ! grep -q "$cuda_payload_dir/include" <<<"$cuda_wrapper_env_output" ||
   ! grep -q "$cuda_cccl_fixture" <<<"$cuda_wrapper_env_output"; then
  echo "[test-package-linux] launcher CPATH did not include both MLX JIT and CUDA CCCL headers:" >&2
  printf '%s\n' "$cuda_wrapper_env_output" >&2
  exit 1
fi
if ! dpkg-deb --field "$cuda_deb" Depends | grep -q 'libcufft-13-0'; then
  echo "[test-package-linux] expected CUDA .deb dependencies to include libcufft-13-0:" >&2
  dpkg-deb --field "$cuda_deb" Depends >&2
  exit 1
fi
if ! dpkg-deb --field "$cuda_deb" Depends | grep -q 'cuda-cudart-dev-13-0'; then
  echo "[test-package-linux] expected CUDA .deb dependencies to include runtime JIT headers:" >&2
  dpkg-deb --field "$cuda_deb" Depends >&2
  exit 1
fi
if [[ "$(dpkg-deb --field "$cuda_deb" Package)" != "mere-run-cuda" ]]; then
  echo "[test-package-linux] expected CUDA .deb package name to be mere-run-cuda:" >&2
  dpkg-deb --field "$cuda_deb" Package >&2
  exit 1
fi

echo "[test-package-linux] CUDA deb dependency test passed."

cuda12_output_dir="$fixture_root/output-cuda12"
PATH="$fake_bin:$PATH" \
  FAKE_CUDA_MAJOR=12 \
  MERERUN_LINUX_ACCEL=cuda \
  MERERUN_MLX_SWIFT_LINK_FLAGS="-lcuda" \
  bash scripts/package-linux.sh \
    --version 0.0.0+cuda12-fixture \
    --artifact-suffix cuda12 \
    --configuration release \
    --skip-build \
    --skip-native \
    --output-dir "$cuda12_output_dir" >/dev/null
cuda12_tarball="$cuda12_output_dir/mere-run-0.0.0+cuda12-fixture-linux-${platform_arch}-cuda12.tar.gz"
cuda12_deb="$cuda12_output_dir/mere-run-cuda12_0.0.0+cuda12-fixture_${deb_arch}.deb"
[[ -f "$cuda12_tarball" ]]
[[ -f "$cuda12_deb" ]]
cuda12_depends="$(dpkg-deb --field "$cuda12_deb" Depends)"
if ! grep -q 'cuda-cudart-12-8 | libcudart12' <<<"$cuda12_depends"; then
  echo "[test-package-linux] expected CUDA 12 .deb dependencies to support NVIDIA and Lambda Stack runtimes:" >&2
  printf '%s\n' "$cuda12_depends" >&2
  exit 1
fi
if ! grep -q 'cuda-cudart-dev-12-8' <<<"$cuda12_depends"; then
  echo "[test-package-linux] expected CUDA 12 .deb dependencies to include runtime JIT headers:" >&2
  printf '%s\n' "$cuda12_depends" >&2
  exit 1
fi
if ! grep -q 'libcudnn9-cuda-12 | python3-torch-cuda' <<<"$cuda12_depends"; then
  echo "[test-package-linux] expected CUDA 12 .deb dependencies to support NVIDIA and Lambda Stack cuDNN packages:" >&2
  printf '%s\n' "$cuda12_depends" >&2
  exit 1
fi
if [[ "$(dpkg-deb --field "$cuda12_deb" Package)" != "mere-run-cuda12" ]]; then
  echo "[test-package-linux] expected CUDA 12 .deb package name to be mere-run-cuda12:" >&2
  dpkg-deb --field "$cuda12_deb" Package >&2
  exit 1
fi

cuda12_override_output_dir="$fixture_root/output-cuda12-override"
PATH="$fake_bin:$PATH" \
  FAKE_CUDA_MAJOR=12 \
  MERERUN_LINUX_ACCEL=cuda \
  MERERUN_MLX_SWIFT_LINK_FLAGS="-lcuda" \
  MERERUN_PACKAGE_LINUX_DEPS="ffmpeg, cuda-cudart-12-4" \
  bash scripts/package-linux.sh \
    --version 0.0.0+cuda12-override-fixture \
    --artifact-suffix cuda \
    --configuration release \
    --skip-build \
    --skip-native \
    --output-dir "$cuda12_override_output_dir" >/dev/null
cuda12_override_deb="$cuda12_override_output_dir/mere-run-cuda_0.0.0+cuda12-override-fixture_${deb_arch}.deb"
[[ -f "$cuda12_override_deb" ]]
if [[ "$(dpkg-deb --field "$cuda12_override_deb" Depends)" != "ffmpeg, cuda-cudart-12-4" ]]; then
  echo "[test-package-linux] explicit CUDA 12 dependency override was not preserved exactly:" >&2
  dpkg-deb --field "$cuda12_override_deb" Depends >&2
  exit 1
fi

unknown_cuda_output_dir="$fixture_root/output-cuda-unknown"
set +e
unknown_cuda_error="$(
  PATH="$fake_bin:$PATH" \
    MERERUN_LINUX_ACCEL=cuda \
    MERERUN_MLX_SWIFT_LINK_FLAGS="-lcuda" \
    bash scripts/package-linux.sh \
      --version 0.0.0+cuda-unknown-fixture \
      --artifact-suffix cuda \
      --configuration release \
      --skip-build \
      --skip-native \
      --output-dir "$unknown_cuda_output_dir" 2>&1
)"
unknown_cuda_status=$?
set -e
if [[ "$unknown_cuda_status" -ne 70 ]] || ! grep -q 'could not determine the built CUDA toolkit major' <<<"$unknown_cuda_error"; then
  echo "[test-package-linux] expected a clean unknown-CUDA-major .deb failure:" >&2
  printf '%s\n' "$unknown_cuda_error" >&2
  exit 1
fi

unknown_tar_output_dir="$fixture_root/output-cuda-unknown-tar"
PATH="$fake_bin:$PATH" \
  MERERUN_LINUX_ACCEL=cuda \
  MERERUN_MLX_SWIFT_LINK_FLAGS="-lcuda" \
  bash scripts/package-linux.sh \
    --version 0.0.0+cuda-unknown-tar-fixture \
    --artifact-suffix cuda \
    --configuration release \
    --skip-build \
    --skip-native \
    --skip-deb \
    --output-dir "$unknown_tar_output_dir" >/dev/null
[[ -f "$unknown_tar_output_dir/mere-run-0.0.0+cuda-unknown-tar-fixture-linux-${platform_arch}-cuda.tar.gz" ]]
(cd "$unknown_tar_output_dir" && sha256sum -c SHA256SUMS >/dev/null)

echo "[test-package-linux] CUDA toolkit-major mappings and explicit override passed."
