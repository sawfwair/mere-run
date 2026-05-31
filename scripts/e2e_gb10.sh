#!/usr/bin/env bash
set -uo pipefail

# Real-world end-to-end model sweep for Linux CUDA hosts (built for the NVIDIA
# GB10 / DGX Spark, but works on any CUDA box). Unlike scripts/e2e_smoke.sh and
# scripts/check-linux.sh, this exercises ACTUAL inference for every model
# category and records whether the run produced a usable artifact, how fast it
# decoded, how busy the GPU got, and whether any CUDA kernel was missing
# (e.g. "GatherQMM has no CUDA implementation").
#
# Why this exists: smoke tests load configs and validate layouts, so they pass
# even when a quantized/MoE MLX path has no CUDA kernel on the host. Only a real
# generate/decode call surfaces those crashes. Run this on the target GPU before
# shipping a Linux CUDA build.
#
# Usage:
#   scripts/e2e_gb10.sh [--bin PATH] [--pull] [--only CAT[,CAT...]]
#                       [--out DIR] [--max-tokens N]
#
#   --bin PATH      mere.run binary to test. Default: mere.run on PATH.
#   --pull          Auto-pull missing models before running them (downloads!).
#                   Default: skip models that are not installed.
#   --only CATS     Comma list of categories to run (text-chat,text-code,
#                   text-embed,text-anonymize,image,speech-tts,speech-asr,
#                   vision-caption,vision-ocr,vision-segment,vision-ground,
#                   music,video). Default: all.
#   --out DIR       Output/artifact dir. Default: ./e2e-gb10-out
#   --max-tokens N  Token cap for text decode runs. Default: 24
#
# Output: $OUT/results.tsv and $OUT/results.md (a pass/fail/throughput matrix),
# plus per-model logs and generated artifacts under $OUT.

BIN="mere.run"
DO_PULL=0
ONLY=""
OUT="./e2e-gb10-out"
MAX_TOKENS=24

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="${2:?}"; shift 2;;
    --pull) DO_PULL=1; shift;;
    --only) ONLY="${2:?}"; shift 2;;
    --out) OUT="${2:?}"; shift 2;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 64;;
  esac
done

mkdir -p "$OUT"
TSV="$OUT/results.tsv"
MD="$OUT/results.md"
ASSETS="$OUT/assets"
LOGS="$OUT/logs"
mkdir -p "$ASSETS" "$LOGS"
printf 'category\tmodel\tstatus\tseconds\tdecode_tps\tgpu_peak\tdetail\n' >"$TSV"

want() { [[ -z "$ONLY" ]] || [[ ",$ONLY," == *",$1,"* ]]; }

installed() {
  "$BIN" model list 2>/dev/null | awk -v m="$1" '$1==m && $3=="installed"{f=1} END{exit f?0:1}'
}

# Sample GPU utilization (%) once per second into a file for the run's duration.
# The background loop MUST redirect its own stdout/stderr, otherwise it inherits
# the command-substitution pipe and `sampler=$(start_gpu_sampler ...)` hangs
# forever waiting for EOF that the infinite loop never produces.
start_gpu_sampler() {
  local f="$1"
  : >"$f"
  ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null >>"$f"; sleep 1; done ) >/dev/null 2>&1 &
  echo $!
}
gpu_peak() { sort -n "$1" 2>/dev/null | tail -1; }

# Run one model. Args: category, model, kind(text|file), expect, timeout, cmd...
# kind=text  -> PASS if exit 0 and stdout non-empty (and contains $expect if set)
# kind=file  -> PASS if exit 0 and file $expect exists and is >1KB
run_case() {
  local cat="$1" model="$2" kind="$3" expect="$4" tmo="$5"; shift 5
  local log="$LOGS/${cat}__${model}.log"
  local gpuf="$LOGS/${cat}__${model}.gpu"

  # Commands that auto-download their own model (no --model id) bypass the
  # catalog/installed gate — label them "auto:..." so we still run them.
  if [[ "$model" != auto:* ]]; then
    if ! "$BIN" model list 2>/dev/null | awk -v m="$model" '$1==m{f=1} END{exit f?0:1}'; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cat" "$model" "SKIP" "0" "-" "-" "not in catalog" >>"$TSV"; return
    fi
    if ! installed "$model"; then
    if [[ "$DO_PULL" == "1" ]]; then
      echo "[pull] $model" | tee -a "$log"
      if ! "$BIN" model pull "$model" >>"$log" 2>&1; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cat" "$model" "PULL_FAIL" "0" "-" "-" "see log" >>"$TSV"; return
      fi
    else
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cat" "$model" "SKIP" "0" "-" "-" "not installed (use --pull)" >>"$TSV"; return
    fi
    fi
  fi

  echo "=== RUN $cat / $model ===" | tee -a "$log"
  local sampler; sampler="$(start_gpu_sampler "$gpuf")"
  local t0 t1 rc; t0="$(date +%s)"
  timeout "$tmo" "$@" >>"$log" 2>&1; rc=$?
  t1="$(date +%s)"
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null

  local secs=$((t1 - t0))
  local peak; peak="$(gpu_peak "$gpuf")"; peak="${peak:-0}%"
  local tps; tps="$(grep -oE 'decode_tps=[0-9.]+' "$log" | tail -1 | cut -d= -f2)"; tps="${tps:-NA}"
  local status detail=""

  # Classify CUDA / crash signatures first — these are the high-signal failures.
  if grep -qiE 'has no CUDA implementation|no CUDA implementation|CUDA error|Fatal error|core dumped|cudaError' "$log"; then
    status="CUDA_FAIL"
    detail="$(grep -iE 'has no CUDA implementation|CUDA error|Fatal error' "$log" | head -1 | cut -c1-90)"
  elif [[ $rc -eq 124 ]]; then
    status="TIMEOUT"; detail="exceeded ${tmo}s"
  elif [[ $rc -ne 0 ]]; then
    status="FAIL"; detail="exit $rc"
  else
    case "$kind" in
      text)
        if [[ -n "$expect" ]] && ! grep -qiE "$expect" "$log"; then
          status="WEAK"; detail="ran but missing expected token"
        else status="PASS"; fi;;
      file)
        if [[ -f "$expect" ]] && [[ "$(stat -c%s "$expect" 2>/dev/null || echo 0)" -gt 1024 ]]; then
          status="PASS"; detail="$(stat -c%s "$expect") bytes"
        else status="WEAK"; detail="no/empty artifact"; fi;;
    esac
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cat" "$model" "$status" "$secs" "$tps" "$peak" "$detail" >>"$TSV"
  echo "[$status] $cat/$model  ${secs}s  tps=$tps  gpu=$peak  $detail"
}

# Shared input fixtures (generated as we go where possible).
SPEECH_WAV="$ASSETS/tts.wav"
OCR_IMG="$ASSETS/ocr.png"

# ---- text ----
want text-chat && for m in text-chat-q35-nano text-chat-q36-nano text-chat-q35 \
                           text-chat-gemma4-turbo text-chat-gemma4-nano \
                           text-chat-gemma4 text-chat-gemma4-max; do
  run_case text-chat "$m" text "" 900 \
    "$BIN" text chat --model "$m" --prompt "Reply with exactly READY." --max-tokens "$MAX_TOKENS" --temperature 0 --quiet --stats
done
want text-code && for m in text-code-qwen3 text-agent-qwen35-9b; do
  run_case text-code "$m" text "" 900 \
    "$BIN" text code --model "$m" --prompt "Write a Python line that prints OK." --max-tokens "$MAX_TOKENS" --temperature 0 --quiet
done
want text-embed && run_case text-embed text-embed-qwen3-0.6b text "" 300 \
  "$BIN" text embed --model text-embed-qwen3-0.6b "hello world"
want text-anonymize && run_case text-anonymize text-anonymize-privacy-filter text "" 300 \
  "$BIN" text anonymize --model text-anonymize-privacy-filter "John Doe can be reached at john@example.com." --json

# ---- image (MLX-quantized image gens are prime CUDA-kernel risks) ----
want image && for m in image-zimage-nano image-zimage-max image-zimage-base \
                       image-klein-nano image-klein-max image-klein-base \
                       image-bonsai-binary image-bonsai-ternary \
                       image-hidream-o1 image-hidream-o1-dev; do
  out="$ASSETS/img_${m}.png"
  run_case image "$m" file "$out" 900 \
    "$BIN" image generate --model "$m" --prompt "a red cube on a white background" \
      --width 256 --height 256 --steps 1 --seed 42 --quiet --output "$out"
done

# ---- speech ----
want speech-tts && for m in speech-tts-qwen3-nano speech-tts-qwen3-customvoice; do
  wav="$ASSETS/tts_${m}.wav"; [[ "$m" == *nano* ]] && wav="$SPEECH_WAV"
  run_case speech-tts "$m" file "$wav" 400 \
    "$BIN" speech synthesize "mere run end to end test one two three" --model "$m" --quiet --output "$wav"
done
if want speech-asr; then
  [[ -f "$SPEECH_WAV" ]] || "$BIN" speech synthesize "mere run end to end test one two three" --model speech-tts-qwen3-nano --quiet --output "$SPEECH_WAV" >/dev/null 2>&1 || true
  for m in speech-asr-parakeet speech-asr-qwen3; do
    backend=parakeet; [[ "$m" == *qwen3* ]] && backend=qwen
    run_case speech-asr "$m" text "" 400 \
      "$BIN" speech transcribe "$SPEECH_WAV" --model "$m" --backend "$backend" --quiet
  done
fi

# ---- vision (needs an input image; reuse a generated one) ----
if want vision-ocr || want vision-caption || want vision-segment || want vision-ground; then
  [[ -f "$ASSETS/img_image-zimage-nano.png" ]] && cp "$ASSETS/img_image-zimage-nano.png" "$OCR_IMG" 2>/dev/null || true
fi
want vision-caption && [[ -f "$OCR_IMG" ]] && run_case vision-caption auto:qwen3-vl text "" 400 \
  "$BIN" vision caption "$OCR_IMG" --prompt "Describe this image." --max-tokens 48
want vision-ocr && [[ -f "$OCR_IMG" ]] && run_case vision-ocr vision-ocr-lighton text "" 300 \
  "$BIN" vision ocr "$OCR_IMG" --model vision-ocr-lighton --backend lighton --quiet
want vision-segment && [[ -f "$OCR_IMG" ]] && run_case vision-segment vision-segment-sam31 file "$ASSETS/seg.png" 400 \
  "$BIN" vision segment "$OCR_IMG" --model vision-segment-sam31 --box 32,32,224,224,obj --output "$ASSETS/seg.png"
want vision-ground && [[ -f "$OCR_IMG" ]] && run_case vision-ground vision-ground-falcon-perception file "$ASSETS/ground.png" 400 \
  "$BIN" vision ground "$OCR_IMG" --model vision-ground-falcon-perception --query "red cube" --output "$ASSETS/ground.png"

# ---- music / video ----
want music && run_case music music-acestep file "$ASSETS/music.wav" 600 \
  "$BIN" music generate "soft ambient synth pulse" --model music-acestep --duration 2 --steps 2 --seed 42 --quiet --output "$ASSETS/music.wav"
want video && run_case video video-ltx-av file "$ASSETS/video.mp4" 1200 \
  "$BIN" video generate "a red cube rotating slowly" --model video-ltx-av --width 256 --height 256 --num-frames 9 --fps 8 --seed 42 --quiet --output "$ASSETS/video.mp4"

# ---- render markdown matrix ----
{
  echo "# GB10 CUDA e2e sweep"
  echo
  echo "Binary: \`$BIN\` — $("$BIN" --version 2>/dev/null | head -1)"
  echo "Host: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1), $(uname -m), $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
  echo
  echo "| Category | Model | Status | Seconds | decode_tps | GPU peak | Detail |"
  echo "| --- | --- | --- | --- | --- | --- | --- |"
  tail -n +2 "$TSV" | while IFS=$'\t' read -r c m s sec tps gpu d; do
    echo "| $c | $m | $s | $sec | $tps | $gpu | $d |"
  done
  echo
  echo "Status legend: PASS = ran + produced artifact/text; WEAK = ran but output looked empty/unexpected; CUDA_FAIL = missing CUDA kernel or fatal CUDA error; FAIL = nonzero exit; TIMEOUT = exceeded budget; SKIP = not installed/in catalog; PULL_FAIL = download failed."
} >"$MD"

echo
echo "=== summary ==="
column -t -s$'\t' "$TSV"
echo
echo "wrote $MD and $TSV"
