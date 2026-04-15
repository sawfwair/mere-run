#!/usr/bin/env bash
set -euo pipefail

# Upload a mere.run DMG to a Cloudflare R2-compatible bucket.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DMG_PATH="${DMG_PATH:-$ROOT_DIR/dist/mere-run.dmg}"
R2_KEY="${R2_KEY:-mere-run-releases/mere-run.dmg}"
R2_ENV_FILE="${R2_ENV_FILE:-}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"

# Optionally source credentials from an env file provided by the caller.
if [[ -n "$R2_ENV_FILE" && -f "$R2_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$R2_ENV_FILE"
fi

CLOUDFLARE_ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-${R2_ACCOUNT_ID:-}}"
R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-}"
R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-}"
R2_BUCKET="${R2_BUCKET:-public}"
R2_ENDPOINT="https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"

if [[ ! -f "$DMG_PATH" ]]; then
  echo "[upload_release] DMG not found at: $DMG_PATH" >&2
  echo "[upload_release] run scripts/release.sh first." >&2
  exit 1
fi

if [[ -z "$CLOUDFLARE_ACCOUNT_ID" || -z "$R2_ACCESS_KEY_ID" || -z "$R2_SECRET_ACCESS_KEY" ]]; then
  echo "[upload_release] R2 credentials not set." >&2
  echo "[upload_release] set CLOUDFLARE_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY" >&2
  echo "[upload_release] or point R2_ENV_FILE at a shell env file that exports them" >&2
  exit 1
fi

DMG_SIZE="$(du -h "$DMG_PATH" | cut -f1 | xargs)"
echo "[upload_release] uploading $DMG_PATH ($DMG_SIZE) -> s3://$R2_BUCKET/$R2_KEY"

AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
aws s3 cp "$DMG_PATH" "s3://${R2_BUCKET}/${R2_KEY}" \
  --endpoint-url "$R2_ENDPOINT" \
  --content-type "application/x-apple-diskimage" \
  --cache-control "public, max-age=3600"

if [[ -n "$PUBLIC_BASE_URL" ]]; then
  echo "[upload_release] done: ${PUBLIC_BASE_URL%/}/$R2_KEY"
else
  echo "[upload_release] done: s3://$R2_BUCKET/$R2_KEY"
fi
