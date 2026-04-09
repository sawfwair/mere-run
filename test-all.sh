#!/bin/bash
# mere.run — Comprehensive Test Script
# Exercises every command and option, pulling models on demand.
# Outputs saved to ~/Desktop/mererun-tests/
set -uo pipefail

OUT="$HOME/Desktop/mererun-tests"
LOG="$OUT/test.log"
PASS=0; FAIL=0; SKIP=0
SECTION=""

mkdir -p "$OUT"
: > "$LOG"

# ── Helpers ──────────────────────────────────────────────────────────────────

ts() { date "+%H:%M:%S"; }

log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

section() {
  SECTION="$1"
  echo ""
  log "════════════════════════════════════════════════════════════════"
  log "  $SECTION"
  log "════════════════════════════════════════════════════════════════"
}

# run_test NAME OUTPUT_FILE COMMAND...
# Runs command, checks exit code and that OUTPUT_FILE exists & is non-empty.
# If OUTPUT_FILE is "-" only the exit code is checked (for stdout-only commands).
run_test() {
  local name="$1"; shift
  local output="$1"; shift
  log "  TEST: $name"
  log "    CMD: $*"
  if eval "$@" >> "$LOG" 2>&1; then
    if [[ "$output" == "-" ]] || [[ -s "$output" ]]; then
      log "    ✓ PASS"
      ((PASS++))
      return 0
    else
      log "    ✗ FAIL (output missing or empty: $output)"
      ((FAIL++))
      return 1
    fi
  else
    log "    ✗ FAIL (exit code $?)"
    ((FAIL++))
    return 1
  fi
}

# run_test_stdout NAME COMMAND...
# Captures stdout, passes if non-empty.
run_test_stdout() {
  local name="$1"; shift
  log "  TEST: $name"
  log "    CMD: $*"
  local result
  if result=$(eval "$@" 2>>"$LOG"); then
    if [[ -n "$result" ]]; then
      log "    ✓ PASS"
      log "    OUTPUT (first 200 chars): ${result:0:200}"
      ((PASS++))
      return 0
    else
      log "    ✗ FAIL (empty stdout)"
      ((FAIL++))
      return 1
    fi
  else
    log "    ✗ FAIL (exit code $?)"
    ((FAIL++))
    return 1
  fi
}

# pull_model ID — returns 0 on success, 1 on failure (caller can skip section)
pull_model() {
  local id="$1"
  log "  Pulling model: $id ..."
  if mere.run model pull "$id" >> "$LOG" 2>&1; then
    log "  Model $id ready."
    return 0
  else
    log "  ⚠ Could not pull $id — skipping section."
    return 1
  fi
}

skip_section() {
  local count="$1"
  log "  Skipping $count tests in $SECTION"
  ((SKIP += count))
}

# ═══════════════════════════════════════════════════════════════════════════════
#  1. MODEL MANAGEMENT (no model pull needed)
# ═══════════════════════════════════════════════════════════════════════════════
section "Model Management"

run_test_stdout "model list" \
  "mere.run model list"

run_test_stdout "model info (music-acestep)" \
  "mere.run model info music-acestep"

run_test_stdout "model repair-manifests" \
  "mere.run model repair-manifests"

# ═══════════════════════════════════════════════════════════════════════════════
#  2. TEXT CHAT
# ═══════════════════════════════════════════════════════════════════════════════
section "Text Chat"

if pull_model text-chat-gemma4; then
  run_test_stdout "basic prompt" \
    "mere.run text chat -p 'What is 2+2? Answer in one word.'"

  run_test_stdout "system prompt" \
    "mere.run text chat -p 'Say hello' -s 'You are a pirate. Respond in pirate speak.'"

  run_test_stdout "temperature override" \
    "mere.run text chat -p 'Write a creative sentence about clouds' --temperature 1.2"

  run_test_stdout "max-tokens limit" \
    "mere.run text chat -p 'Tell me a long story' --max-tokens 32"

  run_test_stdout "stats flag" \
    "mere.run text chat -p 'Hello' --stats 2>&1"

  run_test_stdout "thinking mode" \
    "mere.run text chat -p 'What is 15 * 23?' --thinking"
else
  skip_section 6
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  3. TEXT CODE
# ═══════════════════════════════════════════════════════════════════════════════
section "Text Code"

if pull_model text-code-qwen3; then
  run_test_stdout "basic code generation" \
    "mere.run text code -p 'Write a Python function to check if a number is prime'"

  run_test_stdout "code with streaming" \
    "mere.run text code -p 'Write a bash one-liner to count files in a directory' --stream"

  run_test_stdout "code with stats" \
    "mere.run text code -p 'Write hello world in Rust' --stats 2>&1"
else
  skip_section 3
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  4. TEXT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
section "Text Embeddings"

if pull_model text-embed-qwen3-0.6b; then
  run_test_stdout "single string" \
    "mere.run text embed 'hello world'"

  run_test_stdout "multiple strings" \
    "mere.run text embed 'foo' 'bar' 'baz'"

  run_test "JSON output" "$OUT/embeddings.json" \
    "mere.run text embed 'test embedding' -o '$OUT/embeddings.json'"

  run_test "JSON pretty output" "$OUT/embeddings_pretty.json" \
    "mere.run text embed 'pretty test' -o '$OUT/embeddings_pretty.json' --pretty"
else
  skip_section 4
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  5. IMAGE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
section "Image Generation"

if pull_model image-klein-nano; then
  run_test "basic image" "$OUT/img_basic.png" \
    "mere.run image generate -p 'a red fox sitting in a meadow' -o '$OUT/img_basic.png'"

  run_test "custom dimensions" "$OUT/img_512x768.png" \
    "mere.run image generate -p 'a tall lighthouse at sunset' -W 512 -H 768 -o '$OUT/img_512x768.png'"

  run_test "seed reproducibility (A)" "$OUT/img_seed_a.png" \
    "mere.run image generate -p 'a blue cube on a white table' --seed 42 -o '$OUT/img_seed_a.png'"

  run_test "seed reproducibility (B)" "$OUT/img_seed_b.png" \
    "mere.run image generate -p 'a blue cube on a white table' --seed 42 -o '$OUT/img_seed_b.png'"

  # Compare seed outputs (should be identical)
  if cmp -s "$OUT/img_seed_a.png" "$OUT/img_seed_b.png" 2>/dev/null; then
    log "    ✓ Seed reproducibility confirmed (files identical)"
  else
    log "    ⚠ Seed reproducibility: files differ"
  fi

  run_test "steps override" "$OUT/img_8step.png" \
    "mere.run image generate -p 'a mountain landscape' -s 8 -o '$OUT/img_8step.png'"

  # Generate an image with text for OCR testing
  run_test "text-heavy image (for OCR)" "$OUT/img_text.png" \
    "mere.run image generate -p 'a photograph of a street sign that says HELLO WORLD in bold letters' --seed 99 -o '$OUT/img_text.png'"

  # img2img test
  run_test "img2img" "$OUT/img_img2img.png" \
    "mere.run image generate -p 'a painting of a fox in a meadow, oil painting style' -i '$OUT/img_basic.png' --strength 0.6 -o '$OUT/img_img2img.png'"
else
  skip_section 7
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  6. VISION — CAPTION
# ═══════════════════════════════════════════════════════════════════════════════
section "Vision — Caption"

if [[ -f "$OUT/img_basic.png" ]]; then
  mkdir -p "$OUT/captions"

  run_test_stdout "single image caption" \
    "mere.run vision caption '$OUT/img_basic.png'"

  run_test_stdout "custom prompt caption" \
    "mere.run vision caption '$OUT/img_basic.png' --prompt 'Describe the colors in this image'"

  run_test "output-dir captions" "$OUT/captions/" \
    "mere.run vision caption '$OUT/img_basic.png' '$OUT/img_512x768.png' --output-dir '$OUT/captions'" \
    || true
else
  log "  ⚠ No generated images available — skipping"
  skip_section 3
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  7. VISION — INSPECT
# ═══════════════════════════════════════════════════════════════════════════════
section "Vision — Inspect"

if [[ -f "$OUT/img_basic.png" ]]; then
  run_test_stdout "basic inspect" \
    "mere.run vision inspect '$OUT/img_basic.png' 'What is in this image?'"

  run_test_stdout "inspect with max-tokens" \
    "mere.run vision inspect '$OUT/img_basic.png' 'List every object you see' --max-tokens 512"
else
  log "  ⚠ No generated images available — skipping"
  skip_section 2
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  8. VISION — SEGMENT (SAM 3.1)
# ═══════════════════════════════════════════════════════════════════════════════
section "Vision — Segment"

if [[ -f "$OUT/img_basic.png" ]]; then
  if pull_model vision-segment-sam31; then
    mkdir -p "$OUT/masks"

    run_test "text prompt segment" "$OUT/seg_prompt.png" \
      "mere.run vision segment '$OUT/img_basic.png' --prompt 'fox' -o '$OUT/seg_prompt.png'"

    run_test "box prompt segment" "$OUT/seg_box.png" \
      "mere.run vision segment '$OUT/img_basic.png' --box 100,100,400,400,object -o '$OUT/seg_box.png'"

    run_test "point prompt segment" "$OUT/seg_point.png" \
      "mere.run vision segment '$OUT/img_basic.png' --point 256,256,positive -o '$OUT/seg_point.png'"

    run_test "segment with masks dir" "$OUT/seg_masks.png" \
      "mere.run vision segment '$OUT/img_basic.png' --prompt 'animal' -o '$OUT/seg_masks.png' --mask-output-dir '$OUT/masks'"

    run_test "segment JSON output" "$OUT/seg_output.json" \
      "mere.run vision segment '$OUT/img_basic.png' --prompt 'fox' --json-output '$OUT/seg_output.json'"

    run_test "segment with show-boxes" "$OUT/seg_boxes.png" \
      "mere.run vision segment '$OUT/img_basic.png' --prompt 'fox' -o '$OUT/seg_boxes.png' --show-boxes"

    run_test "segment multimask" "$OUT/seg_multi.png" \
      "mere.run vision segment '$OUT/img_basic.png' --prompt 'fox' -o '$OUT/seg_multi.png' --multimask"
  else
    skip_section 7
  fi
else
  log "  ⚠ No generated images available — skipping"
  skip_section 7
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  9. VISION — OCR
# ═══════════════════════════════════════════════════════════════════════════════
section "Vision — OCR"

if [[ -f "$OUT/img_text.png" ]]; then
  if pull_model vision-ocr-lighton; then
    mkdir -p "$OUT/ocr"

    run_test_stdout "basic OCR" \
      "mere.run vision ocr '$OUT/img_text.png'"

    run_test "OCR with output-dir" "$OUT/ocr/" \
      "mere.run vision ocr '$OUT/img_text.png' -o '$OUT/ocr'"
  else
    skip_section 2
  fi
else
  log "  ⚠ No text image available — skipping"
  skip_section 2
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  10. SPEECH SYNTHESIS (TTS)
# ═══════════════════════════════════════════════════════════════════════════════
section "Speech Synthesis (TTS)"

if pull_model speech-tts-qwen3-nano; then
  run_test "basic synthesis" "$OUT/tts_basic.wav" \
    "mere.run speech synthesize 'Hello, this is a test of the mere.run text to speech system.' -o '$OUT/tts_basic.wav'"

  run_test "voice description" "$OUT/tts_voice.wav" \
    "mere.run speech synthesize 'Welcome to the future of local AI inference.' --voice 'A warm, friendly female voice' -o '$OUT/tts_voice.wav'"

  run_test "TTS streaming" "$OUT/tts_stream.wav" \
    "mere.run speech synthesize 'Streaming speech synthesis test.' --stream -o '$OUT/tts_stream.wav'"

  run_test "TTS temperature" "$OUT/tts_temp.wav" \
    "mere.run speech synthesize 'Testing temperature control for speech.' --temperature 0.3 -o '$OUT/tts_temp.wav'"
else
  skip_section 4
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  11. SPEECH TRANSCRIPTION (ASR)
# ═══════════════════════════════════════════════════════════════════════════════
section "Speech Transcription (ASR)"

if [[ -f "$OUT/tts_basic.wav" ]]; then
  if pull_model speech-asr-parakeet; then
    run_test_stdout "basic transcribe" \
      "mere.run speech transcribe '$OUT/tts_basic.wav'"

    run_test_stdout "transcribe with parakeet backend" \
      "mere.run speech transcribe '$OUT/tts_basic.wav' --backend parakeet"

    run_test "transcribe to file" "$OUT/transcript.txt" \
      "mere.run speech transcribe '$OUT/tts_basic.wav' -o '$OUT/transcript.txt'"

    run_test_stdout "transcribe with timestamps" \
      "mere.run speech transcribe '$OUT/tts_basic.wav' --timestamps"

    run_test_stdout "transcribe streaming" \
      "mere.run speech transcribe '$OUT/tts_basic.wav' --stream"
  else
    skip_section 5
  fi

  # ASR with Qwen backend (for translate task)
  if pull_model speech-asr-qwen3; then
    run_test_stdout "transcribe with qwen backend" \
      "mere.run speech transcribe '$OUT/tts_basic.wav' --backend qwen"
  else
    skip_section 1
  fi
else
  log "  ⚠ No TTS output available — skipping"
  skip_section 6
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  12. SPEECH PROFILES
# ═══════════════════════════════════════════════════════════════════════════════
section "Speech Profiles"

run_test_stdout "profile list" \
  "mere.run speech profile list"

# Profile create/show/delete lifecycle (needs TTS model + reference audio)
if [[ -f "$OUT/tts_basic.wav" ]]; then
  if mere.run speech synthesize "test" --mode clone --ref-audio "$OUT/tts_basic.wav" --save-profile "test-profile" -o "$OUT/tts_clone.wav" >> "$LOG" 2>&1; then
    log "    ✓ PASS: profile create (via clone + save-profile)"
    ((PASS++))

    run_test_stdout "profile show" \
      "mere.run speech profile show test-profile"

    run_test_stdout "profile delete" \
      "mere.run speech profile delete test-profile"
  else
    log "    ⚠ Profile create via clone not available — skipping lifecycle"
    skip_section 2
  fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  13. MUSIC GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
section "Music Generation"

ACESTEP_ROOT="$HOME/Library/Application Support/MereRun/models/music-acestep"

run_test "basic caption" "$OUT/music_basic.wav" \
  "mere.run music generate 'upbeat electronic groove' --duration 10 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_basic.wav'"

run_test "with LM + metadata" "$OUT/music_lm.wav" \
  "mere.run music generate 'jazz piano trio with walking bass and brushed drums' --use-lm --bpm 120 --key 'C minor' --timesig 4 --duration 15 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_lm.wav'"

run_test "with lyrics" "$OUT/music_lyrics.wav" \
  "mere.run music generate 'indie folk ballad with acoustic guitar' --lyrics '[verse]
Walking down the old familiar road
Carrying the stories never told
[chorus]
And we sing for the ones who stayed' --use-lm --bpm 95 --key 'G major' --timesig 4 --duration 20 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_lyrics.wav'"

# Lyrics from file
cat > "$OUT/test_lyrics.txt" << 'LYRICS'
[verse]
Neon lights across the city sky
Digital reflections passing by

[chorus]
We are the signal in the noise
We are the static finding voice
LYRICS

run_test "lyrics from file" "$OUT/music_lyricsfile.wav" \
  "mere.run music generate 'synthwave track with pulsing bassline and arpeggiated synths' --lyrics-file '$OUT/test_lyrics.txt' --use-lm --bpm 128 --key 'A minor' --timesig 4 --duration 20 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_lyricsfile.wav'"

run_test "seed determinism" "$OUT/music_seed.wav" \
  "mere.run music generate 'ambient pad with soft textures' --seed 12345 --duration 10 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_seed.wav'"

run_test "steps override" "$OUT/music_steps.wav" \
  "mere.run music generate 'rock drum beat' -s 4 --duration 10 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_steps.wav'"

run_test "shift override" "$OUT/music_shift.wav" \
  "mere.run music generate 'classical string quartet' --shift 5.0 --duration 10 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_shift.wav'"

run_test "non-cover flag" "$OUT/music_noncover.wav" \
  "mere.run music generate 'lo-fi hip hop beat' --non-cover --duration 10 --checkpoints-root '$ACESTEP_ROOT' -o '$OUT/music_noncover.wav'"

# ═══════════════════════════════════════════════════════════════════════════════
#  14. VIDEO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
section "Video Generation"

if pull_model video-ltx-av; then
  run_test "basic video" "$OUT/video_basic.mp4" \
    "mere.run video generate 'a cinematic drone flythrough over snowy mountains' -o '$OUT/video_basic.mp4'"

  run_test "custom dimensions + frames" "$OUT/video_custom.mp4" \
    "mere.run video generate 'ocean waves crashing on rocks' --width 512 --height 384 --num-frames 25 -o '$OUT/video_custom.mp4'"

  run_test "video with seed" "$OUT/video_seed.mp4" \
    "mere.run video generate 'a candle flickering in the dark' --seed 42 -o '$OUT/video_seed.mp4'"

  run_test "video with fps override" "$OUT/video_fps.mp4" \
    "mere.run video generate 'timelapse of clouds moving' --fps 12 --num-frames 25 -o '$OUT/video_fps.mp4'"

  # img2vid using a generated image
  if [[ -f "$OUT/img_basic.png" ]]; then
    run_test "image-to-video" "$OUT/video_img2vid.mp4" \
      "mere.run video generate 'the scene comes alive with gentle wind' --image '$OUT/img_basic.png' --image-strength 0.8 -o '$OUT/video_img2vid.mp4'"
  else
    log "  ⚠ No source image for img2vid — skipping"
    skip_section 1
  fi
else
  skip_section 5
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  15. VISION — TRACK
# ═══════════════════════════════════════════════════════════════════════════════
section "Vision — Track"

if [[ -f "$OUT/video_basic.mp4" ]]; then
  mkdir -p "$OUT/track_masks"

  run_test "track with text prompt" "$OUT/track_annotated.mp4" \
    "mere.run vision track '$OUT/video_basic.mp4' --prompt 'mountain' -o '$OUT/track_annotated.mp4' --json-output '$OUT/track_result.json'"

  run_test "track with box prompt" "$OUT/track_box.mp4" \
    "mere.run vision track '$OUT/video_basic.mp4' --box 100,100,300,200,region -o '$OUT/track_box.mp4'"

  run_test "track with mask export" "$OUT/track_masks_out.mp4" \
    "mere.run vision track '$OUT/video_basic.mp4' --prompt 'sky' -o '$OUT/track_masks_out.mp4' --mask-output-dir '$OUT/track_masks'"
else
  log "  ⚠ No generated video available — skipping"
  skip_section 3
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  16. API SERVER
# ═══════════════════════════════════════════════════════════════════════════════
section "API Server"

# Need a text model for the API server
if mere.run model list 2>/dev/null | grep -q "text-chat-gemma4.*installed"; then
  API_PORT=18080
  log "  Starting API server on port $API_PORT ..."
  mere.run api serve --engine text-chat-gemma4 --port "$API_PORT" >> "$LOG" 2>&1 &
  API_PID=$!

  # Wait for server to be ready
  SERVER_READY=false
  for i in $(seq 1 30); do
    if curl -sf "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
      SERVER_READY=true
      break
    fi
    sleep 2
  done

  if $SERVER_READY; then
    run_test_stdout "GET /health" \
      "curl -sf 'http://localhost:$API_PORT/health'"

    run_test_stdout "GET /v1/models" \
      "curl -sf 'http://localhost:$API_PORT/v1/models'"

    run_test_stdout "POST /v1/chat/completions" \
      "curl -sf 'http://localhost:$API_PORT/v1/chat/completions' \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"default\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one word\"}], \"max_tokens\": 32}'"

    run_test_stdout "POST /v1/chat/completions (streaming)" \
      "curl -sf 'http://localhost:$API_PORT/v1/chat/completions' \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"default\", \"messages\": [{\"role\": \"user\", \"content\": \"Count to 3\"}], \"max_tokens\": 32, \"stream\": true}'"
  else
    log "  ⚠ API server failed to start — skipping"
    skip_section 4
  fi

  # Cleanup
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  log "  API server stopped."
else
  log "  ⚠ text-chat-gemma4 not installed — skipping API server tests"
  skip_section 4
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo ""
log "════════════════════════════════════════════════════════════════"
log "  RESULTS"
log "════════════════════════════════════════════════════════════════"
log "  ✓ Passed:  $PASS"
log "  ✗ Failed:  $FAIL"
log "  ⚠ Skipped: $SKIP"
log "  Total:     $((PASS + FAIL + SKIP))"
log ""
log "  Outputs:   $OUT/"
log "  Log:       $LOG"
log "════════════════════════════════════════════════════════════════"

# List output files
echo ""
log "Output files:"
find "$OUT" -type f ! -name "test.log" | sort | while read -r f; do
  log "  $(du -h "$f" | cut -f1)  $f"
done

exit $((FAIL > 0 ? 1 : 0))
