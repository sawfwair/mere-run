#!/bin/zsh
set -euo pipefail

# Double-click this file from a maintainer macOS signing host. Running through
# Terminal.app gives codesign normal access to the user's unlocked login keychain.

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_PATH="$LOG_DIR/release-macos-app-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_PATH") 2>&1

finish() {
  local exit_code=$?
  echo "[release-macos-app.command] done $(date) status=$exit_code"
  echo "[release-macos-app.command] log: $LOG_PATH"
}
trap finish EXIT

source_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
    echo "[release-macos-app.command] env file: $env_file"
  fi
}

echo "[release-macos-app.command] started $(date)"
echo "[release-macos-app.command] host $(hostname)"
echo "[release-macos-app.command] root $ROOT_DIR"

source_env_file "$ROOT_DIR/.notarize.local.env"
source_env_file "$HOME/.config/mere-run/notary.env"

export MERERUN_CODESIGN_IDENTITY="${MERERUN_CODESIGN_IDENTITY:-${APPLE_SIGNING_IDENTITY:-${SIGN_IDENTITY:-}}}"
export MERERUN_NOTARY_KEY_ID="${MERERUN_NOTARY_KEY_ID:-${APPLE_API_KEY:-${KEY_ID:-}}}"
export MERERUN_NOTARY_ISSUER_ID="${MERERUN_NOTARY_ISSUER_ID:-${APPLE_API_ISSUER:-${ISSUER_ID:-}}}"
export MERERUN_NOTARY_KEY_PATH="${MERERUN_NOTARY_KEY_PATH:-${APPLE_API_KEY_PATH:-${KEY_PATH:-}}}"
export MERERUN_NOTARY_TEAM_ID="${MERERUN_NOTARY_TEAM_ID:-${APPLE_TEAM_ID:-${TEAM_ID:-}}}"

if [[ -z "$MERERUN_CODESIGN_IDENTITY" ]]; then
  echo "[release-macos-app.command] missing MERERUN_CODESIGN_IDENTITY." >&2
  echo "[release-macos-app.command] set it in .notarize.local.env or ~/.config/mere-run/notary.env." >&2
  exit 1
fi

echo "[release-macos-app.command] signing identity: $MERERUN_CODESIGN_IDENTITY"
if [[ -n "${MERERUN_NOTARY_PROFILE:-}" ]]; then
  echo "[release-macos-app.command] notary auth: keychain profile"
elif [[ -n "$MERERUN_NOTARY_KEY_ID" && -n "$MERERUN_NOTARY_KEY_PATH" ]]; then
  echo "[release-macos-app.command] notary auth: App Store Connect API key"
  echo "[release-macos-app.command] api key path: configured"
elif [[ -n "${MERERUN_NOTARY_APPLE_ID:-}" ]]; then
  echo "[release-macos-app.command] notary auth: Apple ID"
else
  echo "[release-macos-app.command] notary auth: unset"
fi

cd "$ROOT_DIR"

bash scripts/package-macos.sh release
