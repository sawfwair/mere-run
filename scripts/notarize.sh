#!/usr/bin/env bash
set -euo pipefail

# Notarize and staple a mere.run DMG.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DMG_PATH="${DMG_PATH:-$ROOT_DIR/dist/mere-run.dmg}"
PROFILE_NAME="${PROFILE_NAME:-mere-notary}"
TEAM_ID="${TEAM_ID:-S5JDPCT8RC}"

# API key auth (same credentials as Zero pipeline).
# Omit all three to fall back to keychain profile.
KEY_PATH="${KEY_PATH:-$HOME/.config/appstore/AuthKey_VGHT4C4L72.p8}"
KEY_ID="${KEY_ID:-VGHT4C4L72}"
ISSUER_ID="${ISSUER_ID:-69a6de8e-a5b4-47e3-e053-5b8c7c11a4d1}"

if [[ ! -f "$DMG_PATH" ]]; then
  echo "[notarize] dmg not found at: $DMG_PATH" >&2
  echo "[notarize] run scripts/make_dmg.sh first." >&2
  exit 1
fi

# Build auth args
NOTARY_AUTH_ARGS=()
if [[ -n "$KEY_PATH" && -n "$KEY_ID" && -n "$ISSUER_ID" ]]; then
  if [[ ! -f "$KEY_PATH" ]]; then
    echo "[notarize] API key not found at: $KEY_PATH" >&2
    echo "[notarize] falling back to keychain profile: $PROFILE_NAME" >&2
    NOTARY_AUTH_ARGS+=(--keychain-profile "$PROFILE_NAME")
  else
    NOTARY_AUTH_ARGS+=(--key "$KEY_PATH" --key-id "$KEY_ID" --issuer "$ISSUER_ID")
  fi
else
  NOTARY_AUTH_ARGS+=(--keychain-profile "$PROFILE_NAME")
fi

# Zip for upload
ZIP_PATH="$DMG_PATH.zip"
echo "[notarize] zipping: $DMG_PATH -> $ZIP_PATH"
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$DMG_PATH" "$ZIP_PATH"

echo "[notarize] submitting to Apple Notary Service..."
SUBMIT_JSON="$(xcrun notarytool submit "$ZIP_PATH" "${NOTARY_AUTH_ARGS[@]}" --wait --output-format json)" || {
  echo "[notarize] submit failed." >&2
  echo "[notarize] to create a keychain profile:" >&2
  echo "  xcrun notarytool store-credentials \"$PROFILE_NAME\" --team-id \"$TEAM_ID\" --key /path/to/AuthKey.p8 --key-id KEY_ID --issuer ISSUER_ID" >&2
  rm -f "$ZIP_PATH"
  exit 1
}

SUBMIT_ID="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("id",""))' <<<"$SUBMIT_JSON")"
SUBMIT_STATUS="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("status",""))' <<<"$SUBMIT_JSON")"
echo "[notarize] result: id=$SUBMIT_ID status=$SUBMIT_STATUS"

if [[ -z "$SUBMIT_ID" || "$SUBMIT_STATUS" != "Accepted" ]]; then
  echo "[notarize] notarization not accepted; fetching log..." >&2
  if [[ -n "$SUBMIT_ID" ]]; then
    xcrun notarytool log "$SUBMIT_ID" "${NOTARY_AUTH_ARGS[@]}" "$ROOT_DIR/dist/notary-log.json" || true
    echo "[notarize] log saved: $ROOT_DIR/dist/notary-log.json" >&2
  fi
  rm -f "$ZIP_PATH"
  exit 1
fi

rm -f "$ZIP_PATH"

echo "[notarize] stapling..."
xcrun stapler staple "$DMG_PATH"

echo "[notarize] verifying..."
spctl --assess --type open --verbose=4 "$DMG_PATH" || true

echo "[notarize] done: $DMG_PATH (notarized + stapled)"
