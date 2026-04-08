#!/usr/bin/env bash
set -euo pipefail

# Upload mere.run DMG to Cloudflare R2.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DMG_PATH="${DMG_PATH:-$ROOT_DIR/dist/mere-run.dmg}"
R2_KEY="${R2_KEY:-mere-run-releases/mere-run.dmg}"

# Load credentials from Zero's .env (shared R2 bucket)
ZERO_ENV="${ZERO_ENV:-$HOME/projects/zero/.env}"
if [[ -f "$ZERO_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$ZERO_ENV"
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
  echo "[upload_release] or create a .env at: $ZERO_ENV" >&2
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

echo "[upload_release] done: https://public.stereovoid.com/$R2_KEY"
