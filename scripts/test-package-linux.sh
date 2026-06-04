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

case "$(uname -m)" in
  x86_64|amd64)
    platform_arch="x86_64"
    ;;
  aarch64|arm64)
    platform_arch="arm64"
    ;;
  *)
    echo "[test-package-linux] unsupported Linux architecture: $(uname -m)" >&2
    exit 65
    ;;
esac

output_dir="$fixture_root/output"

cat >"$fake_build/mere.run" <<'CLI'
#!/usr/bin/env bash
if [[ "${1:-}" == "__print-env" ]]; then
  printf 'MERERUN_CUDA_CCCL_INCLUDE_PATH=%s\n' "${MERERUN_CUDA_CCCL_INCLUDE_PATH:-}"
  printf 'CPATH=%s\n' "${CPATH:-}"
  exit 0
fi
echo "mere.run package fixture"
CLI
chmod +x "$fake_build/mere.run"

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
