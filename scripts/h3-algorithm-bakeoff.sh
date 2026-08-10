#!/bin/zsh

set -euo pipefail

if (( $# < 3 )); then
  print -u2 "usage: scripts/h3-algorithm-bakeoff.sh MODEL OUTPUT_DIR PROMPT [video-generate arguments...]"
  exit 64
fi

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
model="$1"
output_dir="${2:A}"
prompt="$3"
shift 3
extra_arguments=("$@")

width="${MERERUN_H3_BAKEOFF_WIDTH:-512}"
height="${MERERUN_H3_BAKEOFF_HEIGHT:-512}"
frames="${MERERUN_H3_BAKEOFF_FRAMES:-124}"
seed="${MERERUN_H3_BAKEOFF_SEED:-42}"
arms=",${MERERUN_H3_BAKEOFF_ARMS:-quality,render-75,render-625,layers-45,layers-40,velocity-reuse-2,token-reduction},"
executable="${MERERUN_H3_BAKEOFF_EXECUTABLE:-$repo_root/.build/release/mere.run}"

contaminant_pattern='/mere\.run |mlxfast|mlx-fast|MereRunPackageTests\.xctest|python.*(mlx|train|eval)'
if pgrep -if "$contaminant_pattern" >/dev/null; then
  print -u2 "another ML workload is active; refusing a contaminated H3 algorithm bake-off"
  pgrep -ifl "$contaminant_pattern" >&2
  exit 75
fi

if [[ ! -x "$executable" ]]; then
  (cd "$repo_root" && swift build -c release)
fi
mkdir -p "$output_dir"

arm_enabled() {
  [[ "$arms" == *",$1,"* ]]
}

valid_scaled_canvas() {
  local numerator="$1"
  local denominator="$2"
  (( width * numerator % denominator == 0 )) || return 1
  (( height * numerator % denominator == 0 )) || return 1
  local scaled_width=$((width * numerator / denominator))
  local scaled_height=$((height * numerator / denominator))
  (( scaled_width >= 32 && scaled_height >= 32 )) || return 1
  (( scaled_width % 32 == 0 && scaled_height % 32 == 0 )) || return 1
}

receipt="$output_dir/receipts.tsv"
print -r -- $'arm\tstatus\texact_kernel_mode\twall_seconds\tsha256\toutput' > "$receipt"
{
  print -r -- "commit=$(git -C "$repo_root" rev-parse HEAD)"
  if [[ -z "$(git -C "$repo_root" status --porcelain)" ]]; then
    print -r -- "worktree=clean"
  else
    print -r -- "worktree=dirty"
  fi
  print -r -- "model=$model"
  print -r -- "width=$width"
  print -r -- "height=$height"
  print -r -- "frames=$frames"
  print -r -- "seed=$seed"
  print -r -- "arms=${arms#,}"
  sw_vers
  uname -a
  /usr/sbin/system_profiler SPHardwareDataType
} > "$output_dir/environment.txt"

run_arm() {
  local arm="$1"
  shift
  local exact_kernel_mode="disabled"
  if [[ "${1:-}" == "--exact-kernel-mode" ]]; then
    exact_kernel_mode="$2"
    shift 2
  fi
  local output="$output_dir/$arm.mp4"
  local preflight="$output_dir/$arm.preflight.json"
  local stdout_log="$output_dir/$arm.stdout.log"
  local stderr_log="$output_dir/$arm.stderr.log"
  local start="$EPOCHREALTIME"
  local status

  env MERERUN_H3_EXACT_KERNELS="$exact_kernel_mode" \
    "$executable" video generate "$prompt" \
    --model "$model" \
    --width "$width" \
    --height "$height" \
    --num-frames "$frames" \
    --seed "$seed" \
    "${extra_arguments[@]}" \
    "$@" \
    --output "$output" \
    --preflight --json > "$preflight"

  set +e
  env MERERUN_H3_EXACT_KERNELS="$exact_kernel_mode" \
    "$executable" video generate "$prompt" \
    --model "$model" \
    --width "$width" \
    --height "$height" \
    --num-frames "$frames" \
    --seed "$seed" \
    "${extra_arguments[@]}" \
    "$@" \
    --output "$output" > "$stdout_log" 2> "$stderr_log"
  status=$?
  set -e

  local wall_seconds
  wall_seconds="$(awk -v start="$start" -v finish="$EPOCHREALTIME" 'BEGIN { printf "%.3f", finish - start }')"
  if (( status != 0 )); then
    print -r -- "$arm"$'\t'"failed:$status"$'\t'"$exact_kernel_mode"$'\t'"$wall_seconds"$'\t-'$'\t'"$output" >> "$receipt"
    return "$status"
  fi
  local checksum
  checksum="$(shasum -a 256 "$output" | awk '{print $1}')"
  print -r -- "$arm"$'\tpassed\t'"$exact_kernel_mode"$'\t'"$wall_seconds"$'\t'"$checksum"$'\t'"$output" >> "$receipt"
}

if arm_enabled quality; then
  run_arm quality --h3-acceleration quality
fi
if arm_enabled render-75; then
  if valid_scaled_canvas 3 4; then
    run_arm render-75 \
      --h3-acceleration quality \
      --h3-render-width "$((width * 3 / 4))" \
      --h3-render-height "$((height * 3 / 4))"
  else
    print -r -- $'render-75\tskipped:incompatible-32px-grid\tdisabled\t-\t-\t-' >> "$receipt"
  fi
fi
if arm_enabled render-625; then
  if valid_scaled_canvas 5 8; then
    run_arm render-625 \
      --h3-acceleration quality \
      --h3-render-width "$((width * 5 / 8))" \
      --h3-render-height "$((height * 5 / 8))"
  else
    print -r -- $'render-625\tskipped:incompatible-32px-grid\tdisabled\t-\t-\t-' >> "$receipt"
  fi
fi
if arm_enabled layers-45; then
  run_arm layers-45 --h3-acceleration layers-45
fi
if arm_enabled layers-40; then
  run_arm layers-40 --h3-acceleration layers-40
fi
if arm_enabled velocity-reuse-2; then
  run_arm velocity-reuse-2 --h3-acceleration velocity-reuse-2
fi
if arm_enabled token-reduction; then
  run_arm token-reduction --h3-acceleration token-reduction
fi
if arm_enabled exact-affine-q8; then
  run_arm exact-affine-q8 --exact-kernel-mode affine-q8 --h3-acceleration quality
fi

print -r -- "$receipt"
