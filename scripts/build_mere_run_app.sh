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
default_version="0.18.0"
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

# Regenerate + stamp the MLX Metal kernel library from the current checkout.
# `swift build` never compiles the .metal sources, and shipping a stale
# leftover metallib silently corrupts inference (gibberish, nondeterministic
# generation past ~1024 tokens of context). See scripts/build_mlx_metallib.sh.
"${repo_root}/scripts/build_mlx_metallib.sh" --configuration "$configuration"

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
  "${build_dir}/magentart.framework" \
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

# Refuse to ship a bundle whose metallib doesn't match the checkout it was
# supposedly built from.
"${repo_root}/scripts/build_mlx_metallib.sh" --verify-only "${cli_payload}/mlx-swift_Cmlx.bundle"

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
plutil -insert CFBundleInfoDictionaryVersion -string "6.0" "${contents}/Info.plist"
plutil -insert LSApplicationCategoryType -string "public.app-category.developer-tools" "${contents}/Info.plist"
plutil -insert NSHumanReadableCopyright -string "© mere.run" "${contents}/Info.plist"
# TCC usage string. The CLI opens the camera as a child of this bundle, so TCC attributes
# access to the app and the string must live here. (No microphone string: nothing records the
# mic yet — add NSMicrophoneUsageDescription when `music realtime` mic capture ships.)
plutil -insert NSCameraUsageDescription -string \
  "MereRun uses the camera for live object tracking (vision track-live)." "${contents}/Info.plist"

if [[ -f "$app_icon" ]]; then
  cp "$app_icon" "${resources}/AppIcon.icns"
  plutil -insert CFBundleIconFile -string "AppIcon" "${contents}/Info.plist"
fi

# Codesign inside-out so each Mach-O carries the entitlements it actually needs on its OWN
# signature. We deliberately do not lean on `codesign --deep --force` for entitlements: a forced
# deep pass applies the *same* entitlements to every nested binary (unreliable across toolchains
# and wrong here — the app and CLI need different sets). The app executable links only system
# frameworks and just opens the camera, so it gets the minimal app set; the bundled CLI and the
# vendored ds4 inference binaries JIT, load co-located unsigned frameworks, and capture the
# camera, so they get the broader CLI set. With a Developer ID identity this yields a
# hardened-runtime, notarizable bundle; with the default "-" it ad-hoc signs for local dev.
cli_entitlements="${repo_root}/scripts/MereRunCLI.entitlements"

sign() {
  # sign <entitlements-or-empty> <path...>
  local ents="$1"; shift
  local args=(--force --options runtime --sign "$identity")
  [[ "$identity" != "-" ]] && args+=(--timestamp)
  [[ -n "$ents" ]] && args+=(--entitlements "$ents")
  codesign "${args[@]}" "$@"
}

# 1. Co-located frameworks/bundles — no entitlements.
while IFS= read -r -d '' asset; do
  sign "" "$asset"
done < <(find "$helpers" \( -name '*.framework' -o -name '*.bundle' \) -prune -print0)

# 2. The bundled CLI and every vendored Mach-O under Helpers — CLI entitlements. (The .metal
#    shader sources sit beside the ds4 executables; the file-type test skips them.)
sign "$cli_entitlements" "${helpers}/mere.run"
while IFS= read -r -d '' candidate; do
  if file -b "$candidate" | grep -q 'Mach-O'; then
    sign "$cli_entitlements" "$candidate"
  fi
done < <(find "${helpers}/vendor" -type f -print0 2>/dev/null)

# 3. Seal the bundle. The app executable arrives ad-hoc-signed by the Swift toolchain, so strip
#    that first — then a `--deep` pass WITHOUT `--force` signs only the now-unsigned app exe
#    (with the app entitlements) and seals CodeResources, while skipping — never re-stamping —
#    the already-signed nested code, so each nested binary keeps its own entitlement set. --deep
#    (vs a plain seal) is needed so codesign walks the loose vendored ds4 tree correctly instead
#    of mistaking its sibling .metal shaders for unsigned nested code.
codesign --remove-signature "${macos}/mere.run.app" 2>/dev/null || true
seal_args=(--deep --options runtime --entitlements "$entitlements" --sign "$identity")
[[ "$identity" != "-" ]] && seal_args+=(--timestamp)
codesign "${seal_args[@]}" "$bundle"
codesign --verify --verbose=2 "$bundle"

echo "$bundle"
