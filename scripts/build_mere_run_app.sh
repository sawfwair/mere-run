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
default_version="0.49.0"
if git_version="$(git describe --tags --exact-match 2>/dev/null)"; then
  default_version="${git_version#v}"
fi
default_build="$(git rev-list --count HEAD 2>/dev/null || echo 1)"
app_version="${MERERUN_APP_VERSION:-$default_version}"
app_build="${MERERUN_APP_BUILD:-$default_build}"
sparkle_feed_url="${MERERUN_SPARKLE_FEED_URL:-https://mere.run/releases/appcast.xml}"
sparkle_public_ed_key="6sFs+7UqYcE7rThPAovzMDsZtKyf/h4/d8rUmPSH2rw="

app_icon="${repo_root}/apps/macos/Assets/MereRunApp/AppIcon.icns"
release_entitlements="${repo_root}/scripts/MereRun.entitlements"
debug_entitlements="${repo_root}/scripts/MereRunDebug.entitlements"
# Developer ID Application identity; defaults to ad-hoc ("-") for local/dev builds.
# Set MERERUN_CODESIGN_IDENTITY="Developer ID Application: ..." for distributable builds.
identity="${MERERUN_CODESIGN_IDENTITY:--}"
entitlements="$release_entitlements"
if [[ "$identity" == "-" ]]; then
  entitlements="$debug_entitlements"
fi

swift_app_args=(build --product mere.run.app)
swift_cli_args=(build --product mere.run)
swift_bin_path_args=(build --show-bin-path)
swift_scratch_path="${MERERUN_SWIFT_SCRATCH_PATH:-}"
if [[ -n "$swift_scratch_path" ]]; then
  if [[ "$swift_scratch_path" != /* ]]; then
    echo "MERERUN_SWIFT_SCRATCH_PATH must be an absolute path: ${swift_scratch_path}" >&2
    exit 64
  fi
  swift_app_args+=(--scratch-path "$swift_scratch_path")
  swift_cli_args+=(--scratch-path "$swift_scratch_path")
  swift_bin_path_args+=(--scratch-path "$swift_scratch_path")
fi

private_build_roots=("$repo_root")
source_path_maps=("${repo_root}=/src/mere-run")
if [[ -n "$swift_scratch_path" && "$swift_scratch_path" != "${repo_root}/.build" ]]; then
  private_build_roots+=("$swift_scratch_path")
  source_path_maps+=("${swift_scratch_path}=/src/mere-run/.build")
fi
swift_path_map_args=()
if [[ "${MERERUN_SWIFT_DISABLE_INDEX_STORE:-0}" == "1" ]]; then
  swift_app_args+=(--disable-index-store)
  swift_cli_args+=(--disable-index-store)
  swift_bin_path_args+=(--disable-index-store)
fi
if [[ "$configuration" == "release" ]]; then
  # Only distributable builds need checkout-path sanitization. Keeping debug
  # builds on the same command line as scripts/check.sh lets CI reuse the warm
  # SwiftPM build directory instead of recompiling MereRunCore for the bundle.
  for source_path_map in "${source_path_maps[@]}"; do
    swift_path_map_args+=(
      -Xcc "-ffile-prefix-map=${source_path_map}"
      -Xcxx "-ffile-prefix-map=${source_path_map}"
      -Xswiftc -file-prefix-map
      -Xswiftc "$source_path_map"
    )
  done
  swift_app_args+=("${swift_path_map_args[@]}")
  swift_cli_args+=("${swift_path_map_args[@]}")
  swift_bin_path_args+=("${swift_path_map_args[@]}")
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

verify_private_build_path_absent() {
  local candidate="$1"
  local private_root
  local embedded_path
  for private_root in "${private_build_roots[@]}"; do
    while IFS= read -r embedded_path; do
      # SwiftPM intentionally embeds sibling resource-bundle fallbacks in Bundle.module.
      # Installed bundles resolve their main-bundle copy first; these are not source paths
      # and cannot appear in MLX/C++ diagnostics. Everything else remains release-blocking.
      if [[ "$embedded_path" == "${build_dir}/"*.bundle ]]; then
        continue
      fi
      echo "Private source path leaked into release executable: ${candidate}" >&2
      echo "Leaked path: ${embedded_path}" >&2
      exit 66
    done < <(strings -a "$candidate" | grep -F "${private_root}/" || true)
  done
}

if [[ ! -x "$executable" ]]; then
  echo "Built executable not found: ${executable}" >&2
  exit 66
fi

if [[ ! -x "$cli_executable" ]]; then
  echo "Built CLI not found: ${cli_executable}" >&2
  exit 66
fi

# Debug executables intentionally retain third-party Swift source paths in DWARF and
# assertion metadata. Only release executables are shipped, so enforce the private-path
# boundary on the optimized payloads without rejecting CI's ad-hoc debug bundle.
if [[ "$configuration" == "release" ]]; then
  verify_private_build_path_absent "$executable"
  verify_private_build_path_absent "$cli_executable"
fi

bundle="${build_dir}/MereRun.app"
contents="${bundle}/Contents"
macos="${contents}/MacOS"
resources="${contents}/Resources"
frameworks="${contents}/Frameworks"
# Executable code (CLI, frameworks, vendored helpers) must NOT live under
# Contents/Resources or notarization rejects the bundle. Place it flat in Helpers/ (not a
# same-named subfolder, which codesign would mistake for a malformed bundle) so the CLI and
# its co-located framework/bundle stay together for dyld @executable_path resolution.
helpers="${contents}/Helpers"
cli_payload="${helpers}"

rm -rf "$bundle"
mkdir -p "$macos" "$resources" "$frameworks" "$cli_payload"
cp "$executable" "${macos}/mere.run.app"
cp "$cli_executable" "${cli_payload}/mere.run"

# Sparkle is a versioned framework with symlinked helpers and XPC services.
# `ditto` preserves that layout; flattening or dereferencing it breaks both
# dyld lookup and its nested code signatures.
sparkle_framework="${repo_root}/.build/artifacts/sparkle/Sparkle/Sparkle.xcframework/macos-arm64_x86_64/Sparkle.framework"
if [[ ! -d "$sparkle_framework" ]]; then
  echo "Sparkle framework not found: ${sparkle_framework}" >&2
  exit 66
fi
sparkle_expected_version="2.9.5"
sparkle_framework_version="$(plutil -extract CFBundleShortVersionString raw \
  "${sparkle_framework}/Versions/B/Resources/Info.plist")"
if [[ "$sparkle_framework_version" != "$sparkle_expected_version" ]]; then
  echo "Expected Sparkle ${sparkle_expected_version}, found ${sparkle_framework_version}" >&2
  exit 66
fi
ditto "$sparkle_framework" "${frameworks}/Sparkle.framework"

if [[ -d "${repo_root}/skills/use-mere-run" ]]; then
  mkdir -p "${resources}/skills"
  cp -R "${repo_root}/skills/use-mere-run" "${resources}/skills/use-mere-run"
fi

# Frameworks co-located beside the CLI so its @executable_path rpath resolves.
# MLX's resource-only .bundle is not embedded here because stricter codesign
# treats unsigned nested .bundle directories under Helpers as invalid code.
# The stamped flat Resources/default.metallib layout is enough for runtime
# lookup and is verified below.
for asset in \
  "${build_dir}/llama.framework" \
  "${build_dir}/magentart.framework" \
  "${build_dir}/Resources" \
  "${build_dir}"/*.dylib
do
  if [[ -e "$asset" ]]; then
    cp -R "$asset" "$cli_payload/"
  fi
done

# Ship the complete third-party notices with the app so licenses for loose
# runtime libraries remain available after the source checkout is gone.
third_party_licenses="${resources}/ThirdPartyLicenses"
mkdir -p "$third_party_licenses"
cp "${repo_root}/THIRD_PARTY_NOTICES.md" "${third_party_licenses}/THIRD_PARTY_NOTICES.md"

# Bundle the vendored DeepSeek V4 Flash inference binaries + Metal shaders.
# The premier agent tier spawns vendor/ds4/ds4-server as a subprocess.
if [[ -d "${repo_root}/vendor/ds4" ]]; then
  mkdir -p "${cli_payload}/vendor"
  cp -R "${repo_root}/vendor/ds4" "${cli_payload}/vendor/ds4"
fi

# Refuse to ship a metallib whose stamp doesn't match the checkout it was
# supposedly built from.
"${repo_root}/scripts/build_mlx_metallib.sh" --verify-only "${cli_payload}/Resources"

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
plutil -insert CFBundleURLTypes -json \
  '[{"CFBundleTypeRole":"Viewer","CFBundleURLName":"run.mere.links","CFBundleURLSchemes":["mererun"]}]' \
  "${contents}/Info.plist"
plutil -insert SUFeedURL -string "$sparkle_feed_url" "${contents}/Info.plist"
plutil -insert SUPublicEDKey -string "$sparkle_public_ed_key" "${contents}/Info.plist"
plutil -insert SUEnableAutomaticChecks -bool true "${contents}/Info.plist"
plutil -insert SUAutomaticallyUpdate -bool false "${contents}/Info.plist"
plutil -insert SUScheduledCheckInterval -integer 86400 "${contents}/Info.plist"
plutil -insert SUVerifyUpdateBeforeExtraction -bool true "${contents}/Info.plist"
plutil -insert SURequireSignedFeed -bool true "${contents}/Info.plist"
# TCC usage strings. The CLI opens the camera as a child of this bundle and Voice Studio records
# microphone references in the app process, so both descriptions must live in the bundle plist.
plutil -insert NSCameraUsageDescription -string \
  "MereRun uses the camera for live object tracking (vision track-live)." "${contents}/Info.plist"
plutil -insert NSMicrophoneUsageDescription -string \
  "MereRun uses the microphone to record local voice references and transcription input." "${contents}/Info.plist"

if [[ -f "$app_icon" ]]; then
  cp "$app_icon" "${resources}/AppIcon.icns"
  plutil -insert CFBundleIconFile -string "AppIcon" "${contents}/Info.plist"
fi

# Codesign inside-out so each Mach-O carries the entitlements it actually needs on its OWN
# signature. We deliberately do not lean on `codesign --deep --force` for entitlements: a forced
# deep pass applies the *same* entitlements to every nested binary (unreliable across toolchains
# and wrong here — the app and CLI need different sets). The app executable links only system
# frameworks and opens the camera or microphone, so it gets the minimal app set; the bundled CLI and the
# vendored ds4 inference binaries JIT, load co-located unsigned frameworks, and capture the
# camera or microphone, so they get the broader CLI set. With a Developer ID identity this yields a
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

# Sparkle's nested services require distinct signing treatment. In particular,
# Downloader.xpc carries entitlements that must survive re-signing. Keep the
# explicit inside-out order from Sparkle's distribution guidance.
sparkle_bundle="${frameworks}/Sparkle.framework"
sparkle_version_root="${sparkle_bundle}/Versions/B"
sign "" "${sparkle_version_root}/XPCServices/Installer.xpc"
sparkle_downloader_args=(--force --options runtime --sign "$identity" --preserve-metadata=entitlements)
[[ "$identity" != "-" ]] && sparkle_downloader_args+=(--timestamp)
codesign "${sparkle_downloader_args[@]}" "${sparkle_version_root}/XPCServices/Downloader.xpc"
sign "" "${sparkle_version_root}/Autoupdate"
sign "" "${sparkle_version_root}/Updater.app"
sign "" "$sparkle_bundle"

# 1. Co-located executable frameworks/bundles — no entitlements. SwiftPM also
#    places resource-only .bundle directories here; those have no executable
#    to sign and are sealed as bundle resources in step 3.
while IFS= read -r -d '' asset; do
  if [[ "$asset" == *.bundle ]]; then
    executable_name="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' \
      "${asset}/Contents/Info.plist" 2>/dev/null || true)"
    [[ -n "$executable_name" && -e "${asset}/Contents/MacOS/${executable_name}" ]] || continue
  fi
  sign "" "$asset"
done < <(find "$helpers" \( -name '*.framework' -o -name '*.bundle' \) -prune -print0)

# Loose SwiftPM runtime libraries also need their own signatures before the
# enclosing app is sealed. ONNX Runtime is one such co-located dylib.
while IFS= read -r -d '' asset; do
  if file -b "$asset" | grep -q 'Mach-O'; then
    sign "" "$asset"
  fi
done < <(find "$helpers" -maxdepth 1 -type f -name '*.dylib' -print0)

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
