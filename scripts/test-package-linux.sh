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
output_dir=".build/package-linux-symlink-test-$$"
rm -rf "$output_dir"
mkdir -p "$fake_bin" "$fake_build" "$fake_libs/real" "$fake_libs/alternatives" "$output_dir"

cat >"$fake_build/mere.run" <<'CLI'
#!/usr/bin/env bash
echo "mere.run package fixture"
CLI
chmod +x "$fake_build/mere.run"

printf 'openblas runtime fixture\n' >"$fake_libs/real/libopenblas.so.0.3.26"
ln -s "$fake_libs/real/libopenblas.so.0.3.26" "$fake_libs/alternatives/libopenblas.so.0-x86_64-linux-gnu"
ln -s "$fake_libs/alternatives/libopenblas.so.0-x86_64-linux-gnu" "$fake_libs/libopenblas.so.0"

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

PATH="$fake_bin:$PATH" MERERUN_BUNDLE_SWIFT_LIBS=1 \
  bash scripts/package-linux.sh \
    --version symlink-fixture \
    --configuration release \
    --skip-build \
    --skip-native \
    --skip-deb \
    --output-dir "$output_dir" >/dev/null

tarball="$output_dir/mere-run-symlink-fixture-linux-x86_64.tar.gz"
[[ -f "$tarball" ]]

tar -xzf "$tarball" -C "$fixture_root"
staged_lib="$fixture_root/mere-run-symlink-fixture-linux-x86_64/lib/libopenblas.so.0"

if [[ ! -f "$staged_lib" || -L "$staged_lib" ]]; then
  echo "[test-package-linux] expected libopenblas.so.0 to be bundled as a real file, got:" >&2
  ls -l "$staged_lib" >&2 || true
  exit 1
fi

if ! grep -q 'openblas runtime fixture' "$staged_lib"; then
  echo "[test-package-linux] bundled libopenblas.so.0 did not contain the resolved library contents" >&2
  exit 1
fi

echo "[test-package-linux] package runtime library symlink test passed."
