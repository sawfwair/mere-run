#!/usr/bin/env bash
set -euo pipefail

configuration="${1:-debug}"
case "$configuration" in
  debug|release) ;;
  *)
    echo "Usage: $0 [debug|release]" >&2
    exit 64
    ;;
esac

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# Version/build derive from git when not provided, so the bundle has a single source of
# truth in CI. Fall back to a pinned value when git metadata is unavailable.
default_version="0.17.0"
if git_version="$(git describe --tags --abbrev=0 2>/dev/null)"; then
  default_version="${git_version#v}"
fi
default_build="$(git rev-list --count HEAD 2>/dev/null || echo 1)"
app_version="${MERERUN_APP_VERSION:-$default_version}"
app_build="${MERERUN_APP_BUILD:-$default_build}"

app_icon="${repo_root}/assets/MereRunApp/AppIcon.icns"
entitlements="${repo_root}/scripts/MereRun.entitlements"
# Developer ID Application identity; defaults to ad-hoc ("-") for local/dev builds.
# Set MERERUN_CODESIGN_IDENTITY="Developer ID Application: ..." for distributable builds.
identity="${MERERUN_CODESIGN_IDENTITY:--}"

swift_app_args=(build --product mere.run.app)
swift_cli_args=(build --product mere.run)
swift_bin_path_args=(build --show-bin-path)
if [[ "$configuration" == "release" ]]; then
  swift_app_args+=(--configuration release)
  swift_cli_args+=(--configuration release)
  swift_bin_path_args+=(--configuration release)
fi

swift "${swift_app_args[@]}"
swift "${swift_cli_args[@]}"

build_dir="$(swift "${swift_bin_path_args[@]}")"
executable="${build_dir}/mere.run.app"
cli_executable="${build_dir}/mere.run"

if [[ ! -x "$executable" ]]; then
  echo "Built executable not found: ${executable}" >&2
  exit 66
fi

if [[ ! -x "$cli_executable" ]]; then
  echo "Built CLI not found: ${cli_executable}" >&2
  exit 66
fi

bundle="${build_dir}/MereRun.app"
contents="${bundle}/Contents"
macos="${contents}/MacOS"
resources="${contents}/Resources"
# Executable code (CLI, frameworks, vendored helpers) must NOT live under
# Contents/Resources or notarization rejects the bundle. Place it flat in Helpers/ (not a
# same-named subfolder, which codesign would mistake for a malformed bundle) so the CLI and
# its co-located framework/bundle stay together for dyld @executable_path resolution.
helpers="${contents}/Helpers"
cli_payload="${helpers}"

rm -rf "$bundle"
mkdir -p "$macos" "$resources" "$cli_payload"
cp "$executable" "${macos}/mere.run.app"
cp "$cli_executable" "${cli_payload}/mere.run"

if [[ -d "${repo_root}/skills/use-mere-run" ]]; then
  mkdir -p "${resources}/skills"
  cp -R "${repo_root}/skills/use-mere-run" "${resources}/skills/use-mere-run"
fi

# Frameworks/bundles co-located beside the CLI so its @executable_path rpath resolves.
for asset in \
  "${build_dir}/llama.framework" \
  "${build_dir}/mlx-swift_Cmlx.bundle" \
  "${build_dir}/Resources"
do
  if [[ -e "$asset" ]]; then
    cp -R "$asset" "$cli_payload/"
  fi
done

# Bundle the vendored DeepSeek V4 Flash inference binaries + Metal shaders.
# The premier agent tier spawns vendor/ds4/ds4-server as a subprocess.
if [[ -d "${repo_root}/vendor/ds4" ]]; then
  mkdir -p "${cli_payload}/vendor"
  cp -R "${repo_root}/vendor/ds4" "${cli_payload}/vendor/ds4"
fi

plutil -create xml1 "${contents}/Info.plist"
plutil -insert CFBundleExecutable -string "mere.run.app" "${contents}/Info.plist"
plutil -insert CFBundleIdentifier -string "run.mere.MereRunApp" "${contents}/Info.plist"
plutil -insert CFBundleName -string "MereRun" "${contents}/Info.plist"
plutil -insert CFBundleDisplayName -string "MereRun" "${contents}/Info.plist"
plutil -insert CFBundlePackageType -string "APPL" "${contents}/Info.plist"
plutil -insert CFBundleVersion -string "$app_build" "${contents}/Info.plist"
plutil -insert CFBundleShortVersionString -string "$app_version" "${contents}/Info.plist"
plutil -insert LSMinimumSystemVersion -string "15.0" "${contents}/Info.plist"
plutil -insert NSPrincipalClass -string "NSApplication" "${contents}/Info.plist"
plutil -insert NSHighResolutionCapable -bool true "${contents}/Info.plist"
# TCC usage strings. The CLI captures the camera/mic as a child of this bundle, so TCC
# attributes access to the app and the strings must live here.
plutil -insert NSCameraUsageDescription -string \
  "MereRun uses the camera for live object tracking (vision track-live)." "${contents}/Info.plist"
plutil -insert NSMicrophoneUsageDescription -string \
  "MereRun uses the microphone for realtime music input." "${contents}/Info.plist"

if [[ -f "$app_icon" ]]; then
  cp "$app_icon" "${resources}/AppIcon.icns"
  plutil -insert CFBundleIconFile -string "AppIcon" "${contents}/Info.plist"
fi

# Codesign the whole bundle in a single deep pass so nested code (llama.framework, the mlx
# bundle, the CLI, and the vendored ds4-server) is signed consistently. With a Developer ID
# identity this produces a hardened-runtime, notarizable bundle; otherwise it ad-hoc signs
# for local development. Manual inside-out signing of the mixed vendored tree is brittle, so
# --deep is used deliberately here.
if [[ "$identity" == "-" ]]; then
  codesign --force --deep --sign - "$bundle"
else
  codesign --force --deep --timestamp --options runtime \
    --entitlements "$entitlements" --sign "$identity" "$bundle"
  codesign --verify --deep --strict --verbose=2 "$bundle"
fi

echo "$bundle"
