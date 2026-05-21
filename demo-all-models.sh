#!/bin/bash
# Smoke-test every installed mere.run managed model on this machine.
#
# This is intentionally minimal, not a quality showcase. It runs the shortest
# practical command per model/category and writes logs/artifacts under:
#   ~/Desktop/mererun-model-smoke/<timestamp>/
set -uo pipefail

MERE_RUN="${MERE_RUN:-mere.run}"
OUT="${OUT:-$HOME/Desktop/mererun-model-smoke/$(date +%Y%m%d-%H%M%S)}"
ONLY_MODEL=""
LIST_ONLY=0
METADATA_ONLY=0
INCLUDE_API=1
INCLUDE_UNSUPPORTED=0

IMAGE_SIZE="${IMAGE_SIZE:-512}"
IMAGE_STEPS="${IMAGE_STEPS:-1}"
TEXT_MAX_TOKENS="${TEXT_MAX_TOKENS:-16}"
MUSIC_DURATION="${MUSIC_DURATION:-2}"
MUSIC_STEPS="${MUSIC_STEPS:-1}"
API_PORT_BASE="${API_PORT_BASE:-18080}"

usage() {
  cat <<'USAGE'
Usage: ./demo-all-models.sh [options]

Options:
  --list                 List installed models that would be tested.
  --only <model-id>      Test one installed model.
  --metadata-only        Only run `mere.run model info --components`.
  --no-api               Do not boot localhost API smoke tests for API-only models.
  --include-unsupported  Also test installed models unsupported on this machine.
  --out <dir>            Output directory.
  -h, --help             Show this help.

Environment knobs:
  MERE_RUN=/path/to/mere.run
  IMAGE_SIZE=512
  IMAGE_STEPS=1
  TEXT_MAX_TOKENS=16
  MUSIC_DURATION=2
  MUSIC_STEPS=1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --only)
      ONLY_MODEL="${2:-}"
      if [[ -z "$ONLY_MODEL" ]]; then
        echo "--only requires a model id" >&2
        exit 2
      fi
      shift 2
      ;;
    --metadata-only) METADATA_ONLY=1; shift ;;
    --no-api) INCLUDE_API=0; shift ;;
    --include-unsupported) INCLUDE_UNSUPPORTED=1; shift ;;
    --out)
      OUT="${2:-}"
      if [[ -z "$OUT" ]]; then
        echo "--out requires a directory" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT/logs" "$OUT/artifacts"
SUMMARY="$OUT/summary.tsv"
: > "$SUMMARY"

PASS=0
FAIL=0
SKIP=0

ts() { date "+%H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
slug() { printf "%s" "$1" | tr -c 'A-Za-z0-9._-' '_'; }

record() {
  status="$1"
  model="$2"
  category="$3"
  detail="$4"
  printf "%s\t%s\t%s\t%s\n" "$status" "$model" "$category" "$detail" >> "$SUMMARY"
  case "$status" in
    PASS) PASS=$((PASS + 1)) ;;
    FAIL) FAIL=$((FAIL + 1)) ;;
    SKIP) SKIP=$((SKIP + 1)) ;;
  esac
  log "$status  $model  $detail"
}

run_logged() {
  model="$1"
  category="$2"
  name="$3"
  shift 3
  logfile="$OUT/logs/$(slug "$model")--$(slug "$name").log"

  log "RUN   $model  $name"
  if "$@" >"$logfile" 2>&1; then
    record PASS "$model" "$category" "$name"
    return 0
  fi

  record FAIL "$model" "$category" "$name failed; see $logfile"
  return 1
}

model_root() {
  "$MERE_RUN" model info "$1" 2>/dev/null | awk -F': ' '/^Model Root:/ {print $2; exit}'
}

first_gguf() {
  root="$1"
  find -L "$root" -type f -name '*.gguf' -print 2>/dev/null | head -n 1
}

create_fixture_image() {
  image="$OUT/artifacts/fixture.png"
  [[ -f "$image" ]] && { printf "%s\n" "$image"; return 0; }

  swift - "$image" <<'SWIFT' >/dev/null 2>&1
import AppKit

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let size = NSSize(width: 512, height: 512)
let image = NSImage(size: size)
image.lockFocus()
NSColor.white.setFill()
NSRect(origin: .zero, size: size).fill()
NSColor.systemRed.setFill()
NSBezierPath(rect: NSRect(x: 72, y: 170, width: 368, height: 190)).fill()
NSColor.black.setStroke()
let border = NSBezierPath(rect: NSRect(x: 72, y: 170, width: 368, height: 190))
border.lineWidth = 8
border.stroke()
let paragraph = NSMutableParagraphStyle()
paragraph.alignment = .center
let attrs: [NSAttributedString.Key: Any] = [
    .font: NSFont.boldSystemFont(ofSize: 58),
    .foregroundColor: NSColor.black,
    .paragraphStyle: paragraph
]
"MERE RUN\n123".draw(in: NSRect(x: 80, y: 210, width: 352, height: 120), withAttributes: attrs)
image.unlockFocus()
guard
    let tiff = image.tiffRepresentation,
    let rep = NSBitmapImageRep(data: tiff),
    let png = rep.representation(using: .png, properties: [:])
else { exit(1) }
try png.write(to: url)
SWIFT

  if [[ ! -f "$image" ]]; then
    echo "Unable to create fixture image at $image" >&2
    return 1
  fi
  printf "%s\n" "$image"
}

ensure_audio_fixture() {
  audio="$OUT/artifacts/fixture.wav"
  [[ -f "$audio" ]] && { printf "%s\n" "$audio"; return 0; }

  "$MERE_RUN" speech synthesize \
    "Mere run smoke test. The quick check is working." \
    --model speech-tts-qwen3-nano \
    --voice "A calm clear voice" \
    --temperature 0.2 \
    --quiet \
    -o "$audio" >/dev/null 2>&1

  if [[ ! -f "$audio" ]]; then
    echo "Unable to create fixture audio at $audio" >&2
    return 1
  fi
  printf "%s\n" "$audio"
}

api_smoke() {
  model="$1"
  category="$2"
  engine="$3"
  model_arg="$4"
  port="$5"
  logfile="$OUT/logs/$(slug "$model")--api-serve.log"
  request="$OUT/artifacts/$(slug "$model")-api-request.json"
  response="$OUT/artifacts/$(slug "$model")-api-response.json"

  cat > "$request" <<'JSON'
{
  "model": "local",
  "messages": [
    { "role": "user", "content": "Reply with exactly OK." }
  ],
  "max_tokens": 8,
  "temperature": 0.1
}
JSON

  log "RUN   $model  api serve smoke"
  "$MERE_RUN" api serve --engine "$engine" -m "$model_arg" --port "$port" >"$logfile" 2>&1 &
  pid=$!

  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if [[ "$ready" -eq 1 ]] && curl -fsS \
      -H "Content-Type: application/json" \
      --data @"$request" \
      "http://127.0.0.1:$port/v1/chat/completions" >"$response" 2>>"$logfile"; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
    record PASS "$model" "$category" "api serve smoke"
    return 0
  fi

  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
  record FAIL "$model" "$category" "api serve smoke failed; see $logfile"
  return 1
}

smoke_model() {
  model="$1"
  category="$2"
  root="$(model_root "$model")"

  if [[ -z "$root" ]]; then
    record FAIL "$model" "$category" "could not resolve model root"
    return
  fi

  run_logged "$model" "$category" "model info" \
    "$MERE_RUN" model info "$model" --components || return

  if [[ "$METADATA_ONLY" -eq 1 ]]; then
    return
  fi

  case "$category" in
    image)
      run_logged "$model" "$category" "image generate" \
        "$MERE_RUN" image generate \
          --model "$model" \
          --prompt "a simple studio product photo of a red cube on a white table" \
          --width "$IMAGE_SIZE" \
          --height "$IMAGE_SIZE" \
          --steps "$IMAGE_STEPS" \
          --seed 42 \
          --quiet \
          --output "$OUT/artifacts/$(slug "$model").png"
      ;;
    text-chat)
      case "$model" in
        text-chat-mebot)
          if [[ "$INCLUDE_API" -eq 1 ]]; then
            api_smoke "$model" "$category" "text-chat-klein" "$root" "$API_PORT_BASE"
          else
            record SKIP "$model" "$category" "API-only smoke disabled by --no-api"
          fi
          ;;
        text-chat-q35|text-chat-q35-nano)
          run_logged "$model" "$category" "text chat" \
            "$MERE_RUN" text chat \
              --model "$model" \
              --prompt "Reply with exactly OK." \
              --max-tokens "$TEXT_MAX_TOKENS" \
              --temperature 0.1 \
              --quiet
          ;;
        *)
          run_logged "$model" "$category" "text chat" \
            "$MERE_RUN" text chat \
              --model "$model" \
              --prompt "Reply with exactly OK." \
              --max-tokens "$TEXT_MAX_TOKENS" \
              --temperature 0.1 \
              --quiet
          ;;
      esac
      ;;
    text-code)
      gguf="$(first_gguf "$root")"
      if [[ -z "$gguf" ]]; then
        record FAIL "$model" "$category" "no GGUF file found under $root"
      else
        run_logged "$model" "$category" "text code" \
          "$MERE_RUN" text code \
            --model "$gguf" \
            --prompt "Write one Swift print statement that prints OK." \
            --max-tokens "$TEXT_MAX_TOKENS" \
            --temperature 0.1 \
            --quiet
      fi
      ;;
    speech-tts)
      run_logged "$model" "$category" "speech synthesize" \
        "$MERE_RUN" speech synthesize \
          "Mere run smoke test." \
          --model "$model" \
          --voice "A calm clear voice" \
          --temperature 0.2 \
          --quiet \
          --output "$OUT/artifacts/$(slug "$model").wav"
      ;;
    speech-asr)
      audio="$(ensure_audio_fixture)" || {
        record FAIL "$model" "$category" "could not create audio fixture"
        return
      }
      backend="parakeet"
      [[ "$model" == "speech-asr-qwen3" ]] && backend="qwen"
      run_logged "$model" "$category" "speech transcribe" \
        "$MERE_RUN" speech transcribe "$audio" \
          --backend "$backend" \
          --model "$model" \
          --max-tokens 96 \
          --quiet
      ;;
    text-embed)
      run_logged "$model" "$category" "text embed" \
        "$MERE_RUN" text embed \
          --model "$model" \
          --output "$OUT/artifacts/$(slug "$model").json" \
          --pretty \
          "red cube" "white table"
      ;;
    text-anonymize)
      run_logged "$model" "$category" "text anonymize" \
        "$MERE_RUN" text anonymize \
          --model "$model" \
          --json \
          --pretty \
          --output "$OUT/artifacts/$(slug "$model").json" \
          "Alice Smith can be reached at alice@example.com."
      ;;
    vision-ocr)
      image="$(create_fixture_image)" || {
        record FAIL "$model" "$category" "could not create image fixture"
        return
      }
      run_logged "$model" "$category" "vision ocr" \
        "$MERE_RUN" vision ocr "$image" \
          --model "$model" \
          --backend lighton \
          --output-dir "$OUT/artifacts/ocr-$(slug "$model")" \
          --quiet
      ;;
    vision-segment)
      image="$(create_fixture_image)" || {
        record FAIL "$model" "$category" "could not create image fixture"
        return
      }
      run_logged "$model" "$category" "vision segment" \
        "$MERE_RUN" vision segment "$image" \
          --model "$model" \
          --box "72,152,440,342,red square" \
          --output "$OUT/artifacts/$(slug "$model")-segmented.png" \
          --json-output "$OUT/artifacts/$(slug "$model")-segmented.json" \
          --mask-output-dir "$OUT/artifacts/$(slug "$model")-masks"
      ;;
    vision-ground)
      image="$(create_fixture_image)" || {
        record FAIL "$model" "$category" "could not create image fixture"
        return
      }
      run_logged "$model" "$category" "vision ground" \
        "$MERE_RUN" vision ground "$image" \
          --model "$model" \
          --query "red square" \
          --output "$OUT/artifacts/$(slug "$model")-grounded.png" \
          --json-output "$OUT/artifacts/$(slug "$model")-grounded.json" \
          --mask-output-dir "$OUT/artifacts/$(slug "$model")-masks"
      ;;
    music)
      run_logged "$model" "$category" "music generate" \
        "$MERE_RUN" music generate \
          "short warm synth tone, simple pulse" \
          --model "$model" \
          --duration "$MUSIC_DURATION" \
          --steps "$MUSIC_STEPS" \
          --seed 42 \
          --quiet \
          --output "$OUT/artifacts/$(slug "$model").wav"
      ;;
    video)
      run_logged "$model" "$category" "video generate" \
        "$MERE_RUN" video generate \
          "a red cube rotating on a white table" \
          --model "$model" \
          --width 256 \
          --height 256 \
          --num-frames 9 \
          --fps 8 \
          --seed 42 \
          --quiet \
          --output "$OUT/artifacts/$(slug "$model").mp4"
      ;;
    *)
      record SKIP "$model" "$category" "no smoke command registered for category"
      ;;
  esac
}

installed_file="$OUT/installed-models.tsv"
if [[ "$INCLUDE_UNSUPPORTED" -eq 1 ]]; then
  "$MERE_RUN" model list | awk 'NR > 2 && $3 == "installed" { print $1 "\t" $2 }' > "$installed_file"
else
  supported_file="$OUT/supported-models.txt"
  "$MERE_RUN" model capabilities \
    | awk '/^- / && $3 == "[supported]" { print $2 }' \
    > "$supported_file"
  "$MERE_RUN" model list \
    | awk 'NR == FNR { supported[$1] = 1; next } NR > 2 && $3 == "installed" && supported[$1] { print $1 "\t" $2 }' \
      "$supported_file" - \
    > "$installed_file"
fi

if [[ "$LIST_ONLY" -eq 1 ]]; then
  cat "$installed_file"
  exit 0
fi

if [[ ! -s "$installed_file" ]]; then
  echo "No installed models found via: $MERE_RUN model list" >&2
  exit 1
fi

if [[ -n "$ONLY_MODEL" ]] && ! awk -v model="$ONLY_MODEL" '$1 == model { found = 1 } END { exit(found ? 0 : 1) }' "$installed_file"; then
  echo "Model is not installed or not supported on this machine: $ONLY_MODEL" >&2
  exit 1
fi

log "Output: $OUT"
log "mere.run: $($MERE_RUN --version 2>/dev/null || echo unknown)"

while IFS="$(printf '\t')" read -r model category; do
  [[ -z "$model" ]] && continue
  if [[ -n "$ONLY_MODEL" && "$model" != "$ONLY_MODEL" ]]; then
    continue
  fi
  smoke_model "$model" "$category"
done < "$installed_file"

echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Smoke complete: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
log "Summary: $SUMMARY"
log "Logs:    $OUT/logs"
log "Files:   $OUT/artifacts"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
