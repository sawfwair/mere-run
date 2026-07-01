#!/usr/bin/env bash
set -euo pipefail

# Builds a signed MereRun.app, packages it into a DMG, notarizes, and staples both.
#
# Requires (for a distributable build):
#   MERERUN_CODESIGN_IDENTITY   "Developer ID Application: NAME (TEAMID)"
#   MERERUN_NOTARY_PROFILE      notarytool keychain profile name
#                               (created via: xcrun notarytool store-credentials)
#     OR the trio:
#   MERERUN_NOTARY_APPLE_ID, MERERUN_NOTARY_PASSWORD, MERERUN_NOTARY_TEAM_ID
#
# Without a codesign identity the script still produces an (ad-hoc, un-notarized) DMG so
# the packaging path can be exercised locally and in CI without secrets.

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

configuration="${1:-release}"
identity="${MERERUN_CODESIGN_IDENTITY:--}"

build_log="$(mktemp -t mere-run-app-build.XXXXXX)"
trap 'rm -f "$build_log"' EXIT
if ! MERERUN_CODESIGN_IDENTITY="$identity" "${repo_root}/scripts/build_mere_run_app.sh" "$configuration" 2>&1 | tee "$build_log"; then
  echo "build_mere_run_app.sh failed; see build output above." >&2
  exit 66
fi
bundle="$(awk 'NF { line = $0 } END { print line }' "$build_log")"
if [[ ! -d "$bundle" ]]; then
  echo "build_mere_run_app.sh did not produce a bundle: ${bundle}" >&2
  exit 66
fi

build_dir="$(dirname "$bundle")"
app_version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "${bundle}/Contents/Info.plist" 2>/dev/null || echo "0.0.0")"
dmg_path="${build_dir}/MereRun-${app_version}.dmg"
staging="${build_dir}/dmg-staging"

notarize() {
  local target="$1"
  if [[ -n "${MERERUN_NOTARY_PROFILE:-}" ]]; then
    xcrun notarytool submit "$target" --keychain-profile "$MERERUN_NOTARY_PROFILE" --wait
  elif [[ -n "${MERERUN_NOTARY_APPLE_ID:-}" && -n "${MERERUN_NOTARY_PASSWORD:-}" && -n "${MERERUN_NOTARY_TEAM_ID:-}" ]]; then
    xcrun notarytool submit "$target" \
      --apple-id "$MERERUN_NOTARY_APPLE_ID" \
      --password "$MERERUN_NOTARY_PASSWORD" \
      --team-id "$MERERUN_NOTARY_TEAM_ID" \
      --wait
  else
    echo "No notarytool credentials set; skipping notarization of ${target}." >&2
    return 1
  fi
}

# Assemble a DMG with an /Applications drop target.
rm -rf "$staging" "$dmg_path"
mkdir -p "$staging"
cp -R "$bundle" "$staging/"
ln -s /Applications "${staging}/Applications"

hdiutil create \
  -volname "MereRun" \
  -srcfolder "$staging" \
  -ov -format UDZO \
  "$dmg_path"
rm -rf "$staging"

if [[ "$identity" != "-" ]]; then
  codesign --force --timestamp --sign "$identity" "$dmg_path"
fi

if [[ "$identity" != "-" ]]; then
  if notarize "$bundle"; then
    xcrun stapler staple "$bundle"
  fi
  if notarize "$dmg_path"; then
    xcrun stapler staple "$dmg_path"
  fi
  echo "Gatekeeper assessment:"
  spctl --assess --type open --context context:primary-signature --verbose=4 "$dmg_path" || true
fi

echo "$dmg_path"
