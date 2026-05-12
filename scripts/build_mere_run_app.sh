#!/usr/bin/env bash
set -euo pipefail

configuration="${1:-debug}"
app_version="${MERERUN_APP_VERSION:-0.4.12}"
app_build="${MERERUN_APP_BUILD:-12}"
case "$configuration" in
  debug|release) ;;
  *)
    echo "Usage: $0 [debug|release]" >&2
    exit 64
    ;;
esac

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
app_icon="${repo_root}/assets/MereRunApp/AppIcon.icns"

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
cli_payload="${resources}/mere.run"

rm -rf "$bundle"
mkdir -p "$macos" "$resources" "$cli_payload"
cp "$executable" "${macos}/mere.run.app"
cp "$cli_executable" "${cli_payload}/mere.run"

for asset in \
  "${build_dir}/llama.framework" \
  "${build_dir}/mlx-swift_Cmlx.bundle" \
  "${build_dir}/Resources"
do
  if [[ -e "$asset" ]]; then
    cp -R "$asset" "$cli_payload/"
  fi
done

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

if [[ -f "$app_icon" ]]; then
  cp "$app_icon" "${resources}/AppIcon.icns"
  plutil -insert CFBundleIconFile -string "AppIcon" "${contents}/Info.plist"
fi

codesign --force --sign - "$bundle" >/dev/null
echo "$bundle"
