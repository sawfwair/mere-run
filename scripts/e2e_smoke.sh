#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_ROOT="${MERERUN_MODELS_DIR:-$HOME/Library/Application Support/MereRun/models}"
OUT_DIR="${MERERUN_E2E_OUT_DIR:-$REPO_ROOT/.build/e2e-smoke}"
SUITE="core"

if [[ "${1:-}" == "--installed" ]]; then
  SUITE="installed"
  shift
elif [[ "${1:-}" == "--core" ]]; then
  shift
fi

cd "$REPO_ROOT"

mkdir -p "$OUT_DIR"
find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type f -delete

swift build >/dev/null
MERERUN_BIN="$(swift build --show-bin-path)/mere.run"

declare -a PASSED=()
declare -a FAILED=()
declare -a SKIPPED=()

log() {
  printf '[e2e] %s\n' "$*"
}

record_pass() {
  PASSED+=("$1")
  log "PASS $1"
}

record_fail() {
  FAILED+=("$1")
  log "FAIL $1"
}

record_skip() {
  SKIPPED+=("$1")
  log "SKIP $1"
}

run_with_timeout() {
  local timeout="$1"
  shift
  /usr/bin/python3 - "$timeout" "$@" <<'PY'
import subprocess
import sys

timeout = int(sys.argv[1])
cmd = sys.argv[2:]

try:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
except subprocess.TimeoutExpired as exc:
    if exc.stdout:
        sys.stdout.write(exc.stdout)
    if exc.stderr:
        sys.stderr.write(exc.stderr)
    sys.stderr.write(f"[timeout after {timeout}s]\n")
    sys.exit(124)

if proc.stdout:
    sys.stdout.write(proc.stdout)
if proc.stderr:
    sys.stderr.write(proc.stderr)
sys.exit(proc.returncode)
PY
}

run_step() {
  local name="$1"
  local timeout="$2"
  shift 2
  local stdout_file="$OUT_DIR/$name.stdout"
  local stderr_file="$OUT_DIR/$name.stderr"

  log "RUN $name"
  if run_with_timeout "$timeout" "$@" >"$stdout_file" 2>"$stderr_file"; then
    record_pass "$name"
    return 0
  fi

  record_fail "$name"
  return 1
}

assert_file_nonempty() {
  local label="$1"
  local path="$2"
  if [[ -s "$path" ]]; then
    record_pass "$label"
  else
    record_fail "$label"
  fi
}

assert_contains() {
  local label="$1"
  local file="$2"
  local pattern="$3"
  if grep -Eiq "$pattern" "$file"; then
    record_pass "$label"
  else
    record_fail "$label"
  fi
}

require_model() {
  local name="$1"
  if [[ -d "$MODELS_ROOT/$name" ]]; then
    return 0
  fi
  record_skip "$name not installed at $MODELS_ROOT/$name"
  return 1
}

make_ocr_fixture() {
  local output_path="$1"
  /usr/bin/swift -e 'import AppKit; let size = NSSize(width: 512, height: 160); let image = NSImage(size: size); image.lockFocus(); NSColor.white.setFill(); NSBezierPath(rect: NSRect(origin: .zero, size: size)).fill(); let text = NSString(string: "MERE RUN OCR TEST 123"); let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.boldSystemFont(ofSize: 54), .foregroundColor: NSColor.black]; text.draw(at: NSPoint(x: 24, y: 52), withAttributes: attrs); image.unlockFocus(); let data = image.tiffRepresentation!; let rep = NSBitmapImageRep(data: data)!; try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]));' "$output_path"
}

make_hidream_reference_fixture() {
  local output_path="$1"
  local variant="${2:-product}"
  /usr/bin/swift - "$output_path" "$variant" <<'SWIFT'
import AppKit

let output = CommandLine.arguments[1]
let variant = CommandLine.arguments[2]
let size = NSSize(width: 512, height: 640)
let image = NSImage(size: size)
image.lockFocus()
NSColor(calibratedWhite: 0.96, alpha: 1).setFill()
NSBezierPath(rect: NSRect(origin: .zero, size: size)).fill()

if variant == "person" {
    NSColor(calibratedRed: 0.93, green: 0.78, blue: 0.58, alpha: 1).setFill()
    NSBezierPath(ovalIn: NSRect(x: 176, y: 360, width: 160, height: 160)).fill()
    NSColor(calibratedRed: 0.25, green: 0.35, blue: 0.85, alpha: 1).setFill()
    NSBezierPath(roundedRect: NSRect(x: 148, y: 140, width: 216, height: 230), xRadius: 42, yRadius: 42).fill()
    NSColor.black.setFill()
    NSBezierPath(ovalIn: NSRect(x: 216, y: 430, width: 18, height: 18)).fill()
    NSBezierPath(ovalIn: NSRect(x: 278, y: 430, width: 18, height: 18)).fill()
} else {
    NSColor(calibratedRed: 0.08, green: 0.50, blue: 0.35, alpha: 1).setFill()
    NSBezierPath(roundedRect: NSRect(x: 116, y: 140, width: 280, height: 300), xRadius: 46, yRadius: 46).fill()
    NSColor(calibratedRed: 0.95, green: 0.80, blue: 0.20, alpha: 1).setFill()
    NSBezierPath(roundedRect: NSRect(x: 156, y: 260, width: 200, height: 90), xRadius: 20, yRadius: 20).fill()
    NSColor.white.setFill()
    NSBezierPath(ovalIn: NSRect(x: 216, y: 470, width: 80, height: 80)).fill()
}

image.unlockFocus()
let data = image.tiffRepresentation!
let rep = NSBitmapImageRep(data: data)!
try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output))
SWIFT
}

core_suite() {
  local speech_tts_wav="$OUT_DIR/speech-tts-qwen3-nano.wav"

  if require_model "speech-tts-qwen3-nano"; then
    run_step "speech_synthesize_qwen3_nano" 240 \
      "$MERERUN_BIN" speech synthesize "mere.run smoke test one two three" --output "$speech_tts_wav" --model speech-tts-qwen3-nano --quiet
    assert_file_nonempty "speech_synthesize_qwen3_nano_artifact" "$speech_tts_wav"
  fi

  if require_model "speech-asr-parakeet" && [[ -f "$speech_tts_wav" ]]; then
    run_step "speech_transcribe_parakeet" 240 \
      "$MERERUN_BIN" speech transcribe "$speech_tts_wav" --backend parakeet --model speech-asr-parakeet --no-timestamps --quiet
    assert_contains "speech_transcribe_parakeet_output" "$OUT_DIR/speech_transcribe_parakeet.stdout" "mere.run smoke test"
  fi

  if require_model "speech-asr-qwen3" && [[ -f "$speech_tts_wav" ]]; then
    run_step "speech_transcribe_qwen3" 300 \
      "$MERERUN_BIN" speech transcribe "$speech_tts_wav" --backend qwen --model speech-asr-qwen3 --no-timestamps --quiet
    assert_contains "speech_transcribe_qwen3_output" "$OUT_DIR/speech_transcribe_qwen3.stdout" "mere.run smoke test"
  fi

  if require_model "text-chat-q35-nano"; then
    run_step "text_chat_q35_nano" 300 \
      "$MERERUN_BIN" text chat --prompt "Reply with exactly READY." --model text-chat-q35-nano --max-tokens 8 --temperature 0 --top-p 1 --quiet
    assert_contains "text_chat_q35_nano_output" "$OUT_DIR/text_chat_q35_nano.stdout" "^READY$"
  fi

  if require_model "image-zimage-max"; then
    run_step "image_generate_zimage_max" 300 \
      "$MERERUN_BIN" image generate --prompt "a red cube on a white background" --model "$MODELS_ROOT/image-zimage-max" --width 256 --height 256 --steps 1 --output "$OUT_DIR/image-zimage-max.png" --quiet
    assert_file_nonempty "image_generate_zimage_max_artifact" "$OUT_DIR/image-zimage-max.png"
  fi

  if require_model "music-acestep"; then
    run_step "music_generate" 300 \
      "$MERERUN_BIN" music generate "soft ambient synth pulse" --checkpoints-root "$MODELS_ROOT/music-acestep" --duration 2 --steps 2 --output "$OUT_DIR/music-acestep.wav" --quiet
    assert_file_nonempty "music_generate_artifact" "$OUT_DIR/music-acestep.wav"
  fi
}

installed_suite() {
  if require_model "text-chat-q35"; then
    run_step "text_chat_q35" 600 \
      "$MERERUN_BIN" text chat --prompt "Reply with exactly READY and nothing else." --model text-chat-q35 --max-tokens 64 --temperature 0 --top-p 1 --quiet
    if [[ -s "$OUT_DIR/text_chat_q35.stdout" ]]; then
      record_pass "text_chat_q35_output"
    else
      record_fail "text_chat_q35_output"
    fi
  fi

  if require_model "vision-ocr-lighton"; then
    make_ocr_fixture "$OUT_DIR/ocr-fixture.png"
    run_step "vision_ocr_lighton" 60 \
      "$MERERUN_BIN" vision ocr "$OUT_DIR/ocr-fixture.png" --backend lighton --model "$MODELS_ROOT/vision-ocr-lighton" --max-tokens 128 --quiet
    assert_contains "vision_ocr_lighton_output" "$OUT_DIR/vision_ocr_lighton.stdout" "mere|run|ocr|123"
  fi

  if require_model "image-klein-nano"; then
    run_step "image_generate_klein_nano" 180 \
      "$MERERUN_BIN" image generate --prompt "a red cube on a white background" --model "$MODELS_ROOT/image-klein-nano" --width 256 --height 256 --steps 1 --output "$OUT_DIR/image-klein-nano.png" --quiet
    assert_file_nonempty "image_generate_klein_nano_artifact" "$OUT_DIR/image-klein-nano.png"
  fi

  if require_model "image-klein-max"; then
    run_step "image_generate_klein_max" 180 \
      "$MERERUN_BIN" image generate --prompt "a red cube on a white background" --model "$MODELS_ROOT/image-klein-max" --width 256 --height 256 --steps 1 --output "$OUT_DIR/image-klein-max.png" --quiet
    assert_file_nonempty "image_generate_klein_max_artifact" "$OUT_DIR/image-klein-max.png"
  fi

  if require_model "video-ltx-av"; then
    run_step "video_generate_ltx_av" 900 \
      "$MERERUN_BIN" video generate "a red cube rotating slowly" --variant unified-av --model-root "$MODELS_ROOT/video-ltx-av" --width 256 --height 256 --num-frames 9 --fps 8 --output "$OUT_DIR/video-ltx-av.mp4" --quiet
    assert_file_nonempty "video_generate_ltx_av_artifact" "$OUT_DIR/video-ltx-av.mp4"
  fi

  if [[ "${MERERUN_E2E_HIDREAM:-}" == "1" ]] && require_model "image-hidream-o1-dev"; then
    make_hidream_reference_fixture "$OUT_DIR/hidream-ref-product.png" product
    make_hidream_reference_fixture "$OUT_DIR/hidream-ref-person.png" person

    run_step "image_generate_hidream_dev_text" 1200 \
      "$MERERUN_BIN" image generate --prompt "a small brass camera on a clean desk" --model "$MODELS_ROOT/image-hidream-o1-dev" --width 512 --height 512 --steps 1 --output "$OUT_DIR/image-hidream-o1-dev-text.png" --quiet
    assert_file_nonempty "image_generate_hidream_dev_text_artifact" "$OUT_DIR/image-hidream-o1-dev-text.png"

    run_step "image_generate_hidream_dev_single_ref" 1200 \
      "$MERERUN_BIN" image generate --prompt "turn this into a clean studio product photo" --model "$MODELS_ROOT/image-hidream-o1-dev" --ref-image "$OUT_DIR/hidream-ref-product.png" --width 512 --height 512 --steps 1 --output "$OUT_DIR/image-hidream-o1-dev-single-ref.png" --quiet
    assert_file_nonempty "image_generate_hidream_dev_single_ref_artifact" "$OUT_DIR/image-hidream-o1-dev-single-ref.png"

    run_step "image_generate_hidream_dev_multi_ref" 1200 \
      "$MERERUN_BIN" image generate --prompt "put the same subject into a cinematic city portrait" --model "$MODELS_ROOT/image-hidream-o1-dev" --ref-image "$OUT_DIR/hidream-ref-person.png" --ref-image "$OUT_DIR/hidream-ref-product.png" --width 512 --height 512 --steps 1 --output "$OUT_DIR/image-hidream-o1-dev-multi-ref.png" --quiet
    assert_file_nonempty "image_generate_hidream_dev_multi_ref_artifact" "$OUT_DIR/image-hidream-o1-dev-multi-ref.png"
  fi

  if [[ "${MERERUN_E2E_HIDREAM_FULL:-}" == "1" ]] && require_model "image-hidream-o1"; then
    run_step "image_generate_hidream_full_text" 1200 \
      "$MERERUN_BIN" image generate --prompt "a small brass camera on a clean desk" --model "$MODELS_ROOT/image-hidream-o1" --width 512 --height 512 --steps 3 --output "$OUT_DIR/image-hidream-o1-full-text.png" --quiet
    assert_file_nonempty "image_generate_hidream_full_text_artifact" "$OUT_DIR/image-hidream-o1-full-text.png"
  fi
}

log "Using models root: $MODELS_ROOT"
log "Writing artifacts to: $OUT_DIR"
log "Suite: $SUITE"

core_suite
if [[ "$SUITE" == "installed" ]]; then
  installed_suite
fi

printf '\n[e2e] Summary\n'
printf '[e2e] passed: %d\n' "${#PASSED[@]}"
printf '[e2e] failed: %d\n' "${#FAILED[@]}"
printf '[e2e] skipped: %d\n' "${#SKIPPED[@]}"

if ((${#PASSED[@]})); then
  printf '[e2e] pass list: %s\n' "${PASSED[*]}"
fi
if ((${#FAILED[@]})); then
  printf '[e2e] fail list: %s\n' "${FAILED[*]}"
fi
if ((${#SKIPPED[@]})); then
  printf '[e2e] skip list: %s\n' "${SKIPPED[*]}"
fi

if ((${#FAILED[@]})); then
  exit 1
fi
