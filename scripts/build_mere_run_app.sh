#!/usr/bin/env bash
set -euo pipefail

configuration="${1:-debug}"
app_version="${MERERUN_APP_VERSION:-0.4.9}"
app_build="${MERERUN_APP_BUILD:-9}"
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

swift_args=(build --product mere.run.app)
if [[ "$configuration" == "release" ]]; then
  swift_args+=(--configuration release)
fi

swift "${swift_args[@]}"

triple="$(swift -print-target-info | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["target"]["triple"])')"
build_dir=".build/${triple}/${configuration}"
executable="${build_dir}/mere.run.app"

if [[ ! -x "$executable" ]]; then
  build_dir=".build/${configuration}"
  executable="${build_dir}/mere.run.app"
fi

if [[ ! -x "$executable" ]]; then
  echo "Built executable not found: ${executable}" >&2
  exit 66
fi

bundle="${build_dir}/MereRun.app"
contents="${bundle}/Contents"
macos="${contents}/MacOS"
resources="${contents}/Resources"

rm -rf "$bundle"
mkdir -p "$macos" "$resources"
cp "$executable" "${macos}/mere.run.app"

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
echo "$repo_root/$bundle"
