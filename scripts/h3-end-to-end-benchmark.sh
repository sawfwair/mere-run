#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
binary="$repo_root/.build/arm64-apple-macosx/release/mere.run"
model_root="${MERERUN_H3_MODEL_ROOT:-/Users/nerd/Library/Application Support/MereRun/models/video-minimax-h3-fl2va-bf16-mlx}"
mode="${1:-probe}"
width=832
height=480
frames=124

case "$mode" in
  probe)
    steps=2
    acceleration=quality
    ;;
  reuse-probe)
    steps=4
    acceleration=maximum
    ;;
  probe-768)
    width=1344
    height=768
    steps=2
    acceleration=quality
    ;;
  reuse-probe-768)
    width=1344
    height=768
    steps=4
    acceleration=maximum
    ;;
  target-quality)
    steps=20
    acceleration=quality
    ;;
  target-maximum)
    steps=20
    acceleration=maximum
    ;;
  target-768-quality)
    width=1344
    height=768
    steps=20
    acceleration=quality
    ;;
  target-768-maximum)
    width=1344
    height=768
    steps=20
    acceleration=maximum
    ;;
  proxy-quality)
    width=416
    height=256
    frames=107
    steps=20
    acceleration=quality
    ;;
  proxy-maximum)
    width=416
    height=256
    frames=107
    steps=20
    acceleration=maximum
    ;;
  *)
    print -u2 "usage: scripts/h3-end-to-end-benchmark.sh [probe|reuse-probe|probe-768|reuse-probe-768|proxy-quality|proxy-maximum|target-quality|target-maximum|target-768-quality|target-768-maximum]"
    exit 64
    ;;
esac

width="${MERERUN_H3_BENCH_WIDTH:-$width}"
height="${MERERUN_H3_BENCH_HEIGHT:-$height}"
frames="${MERERUN_H3_BENCH_FRAMES:-$frames}"
steps="${MERERUN_H3_BENCH_STEPS:-$steps}"

if [[ ! -x "$binary" || "${MERERUN_H3_BENCH_REBUILD:-0}" == "1" ]]; then
  swift build -c release --product mere.run
fi
if pgrep -f '/mere\.run ' >/dev/null; then
  print -u2 "another mere.run workload is active; refusing a contaminated H3 benchmark"
  pgrep -fl '/mere\.run ' >&2
  exit 75
fi

output_dir="$repo_root/tmp/h3-benchmark"
mkdir -p "$output_dir"
label="${MERERUN_H3_BENCH_LABEL:+-${MERERUN_H3_BENCH_LABEL}}"
output="$output_dir/h3-${mode}${label}-${width}x${height}-f${frames}-s${steps}.mp4"
prompt="Epic cinematic night at a rain-soaked city bus stop: a caped superhero stands calmly beneath a bright yellow umbrella. A silent lightning vortex tears open above the street, a city bus lifts weightlessly into the air, transforms into a colossal luminous dragon, and circles the hero while rain freezes in midair; dramatic volumetric lighting, realistic materials, fluid continuous camera movement, coherent anatomy, blockbuster visual effects."

export MERERUN_H3_PROFILE_PHASES=1
export MERERUN_H3_PROFILE_STEPS=1
export CFFIXED_USER_HOME="${MERERUN_H3_BENCH_HOME:-${TMPDIR:-/tmp}/mere-run-h3-end-to-end-home}"
mkdir -p "$CFFIXED_USER_HOME"

exec "$binary" video generate "$prompt" \
  --model-root "$model_root" \
  --output "$output" \
  --width "$width" \
  --height "$height" \
  --num-frames "$frames" \
  --steps "$steps" \
  --seed 20260804 \
  --h3-weight-mode resident-bf16 \
  --h3-acceleration "$acceleration"
