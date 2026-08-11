#!/bin/zsh

set -euo pipefail
zmodload zsh/datetime

if (( $# < 3 )); then
  print -u2 "usage: scripts/h3-algorithm-bakeoff.sh MODEL OUTPUT_DIR PROMPT [video-generate arguments...]"
  exit 64
fi

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
invocation_dir="${PWD:A}"
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
scorer="$repo_root/scripts/h3-bakeoff-score.py"
reference_manifest="${MERERUN_H3_BAKEOFF_REFERENCE_MANIFEST:-}"
max_starting_swap_mib="${MERERUN_H3_BAKEOFF_MAX_STARTING_SWAP_MIB:-1024}"

if [[ ! "$max_starting_swap_mib" =~ '^[0-9]+([.][0-9]+)?$' ]]; then
  print -u2 "MERERUN_H3_BAKEOFF_MAX_STARTING_SWAP_MIB must be a nonnegative number"
  exit 64
fi

mkdir -p "$output_dir"

prompt_receipt="$output_dir/prompt.txt"
argument_receipt="$output_dir/arguments.tsv"
reference_receipt="$output_dir/references.tsv"
print -rn -- "$prompt" > "$prompt_receipt"
print -r -- $'order\tzsh_quoted_value' > "$argument_receipt"
for ((argument_index = 1; argument_index <= ${#extra_arguments}; argument_index++)); do
  printf '%d\t%q\n' "$argument_index" "${extra_arguments[$argument_index]}" >> "$argument_receipt"
done

print -r -- $'order\tkind\tbytes\tsha256\tresolved_path\targument' > "$reference_receipt"
reference_index=0
for ((argument_index = 1; argument_index <= ${#extra_arguments}; argument_index++)); do
  argument="${extra_arguments[$argument_index]}"
  reference=""
  if [[ "$argument" == "--reference" ]]; then
    if (( argument_index == ${#extra_arguments} )); then
      print -u2 "--reference is missing its kind:path value"
      exit 64
    fi
    argument_index=$((argument_index + 1))
    reference="${extra_arguments[$argument_index]}"
  elif [[ "$argument" == --reference=* ]]; then
    reference="${argument#--reference=}"
  elif [[ "$argument" == "--image" || "$argument" == "--end-image" ]]; then
    if (( argument_index == ${#extra_arguments} )); then
      print -u2 "$argument is missing its image path"
      exit 64
    fi
    argument_index=$((argument_index + 1))
    reference="image:${extra_arguments[$argument_index]}"
  elif [[ "$argument" == --image=* || "$argument" == --end-image=* ]]; then
    reference="image:${argument#*=}"
  else
    continue
  fi
  if [[ "$reference" != *:* ]]; then
    print -u2 "H3 bake-off reference must use kind:path: $reference"
    exit 64
  fi
  reference_kind="${reference%%:*}"
  reference_path="${reference#*:}"
  if [[ "$reference_kind" != "image" && "$reference_kind" != "video" && "$reference_kind" != "audio" ]]; then
    print -u2 "unsupported H3 bake-off reference kind: $reference_kind"
    exit 64
  fi
  if [[ "$reference_path" == *$'\t'* || "$reference_path" == *$'\n'* ]]; then
    print -u2 "H3 bake-off reference paths cannot contain tabs or newlines"
    exit 64
  fi
  resolved_reference="${reference_path:A}"
  if [[ ! -f "$resolved_reference" ]]; then
    print -u2 "H3 bake-off reference is unavailable: $resolved_reference"
    exit 66
  fi
  reference_index=$((reference_index + 1))
  reference_bytes="$(stat -f '%z' "$resolved_reference")"
  reference_sha256="$(shasum -a 256 "$resolved_reference" | awk '{print $1}')"
  print -r -- "$reference_index"$'\t'"$reference_kind"$'\t'"$reference_bytes"$'\t'"$reference_sha256"$'\t'"$resolved_reference"$'\t'"$reference" >> "$reference_receipt"
done

if [[ -n "$reference_manifest" ]]; then
  reference_manifest="${reference_manifest:A}"
  if [[ ! -f "$reference_manifest" ]]; then
    print -u2 "H3 bake-off reference manifest is unavailable: $reference_manifest"
    exit 66
  fi
  if ! diff -u \
    <(awk -F '\t' 'NF && $1 !~ /^#/ && $1 != "order" { print $1 "\t" $2 "\t" $3 "\t" $4 }' "$reference_manifest") \
    <(awk -F '\t' 'NR > 1 { print $1 "\t" $2 "\t" $3 "\t" $4 }' "$reference_receipt") \
    > "$output_dir/reference-manifest.diff"; then
    print -u2 "H3 bake-off references do not match the pinned manifest: $reference_manifest"
    print -u2 "see $output_dir/reference-manifest.diff"
    exit 65
  fi
  rm -f "$output_dir/reference-manifest.diff"
fi

swap_usage="$(sysctl -n vm.swapusage)"
starting_swap_mib="$(print -r -- "$swap_usage" | awk '
  {
    for (field_index = 1; field_index <= NF; field_index++) {
      if ($field_index == "used" && $(field_index + 1) == "=") {
        raw = $(field_index + 2)
        unit = substr(raw, length(raw), 1)
        value = substr(raw, 1, length(raw) - 1) + 0
        factor = 1
        if (unit == "K") factor = 1 / 1024
        else if (unit == "G") factor = 1024
        else if (unit == "T") factor = 1024 * 1024
        else if (unit != "M") exit 2
        printf "%.3f\n", value * factor
        found = 1
        exit
      }
    }
  }
  END { if (!found) exit 2 }
')" || {
  print -u2 "could not parse starting swap usage: $swap_usage"
  exit 69
}

contaminant_pattern='/mere\.run |mere\.run-node|mlxfast|mlx-fast|MereRunPackageTests\.xctest|python.*(mlx|train|eval)|swift-build|swift-driver|swift-frontend|xcodebuild'
contaminant_pids="$(pgrep -if "$contaminant_pattern" | awk -v current="$$" -v parent="$PPID" '$1 != current && $1 != parent' || true)"
contaminants=""
if [[ -n "$contaminant_pids" ]]; then
  while IFS= read -r contaminant_pid; do
    process_executable="$(ps -p "$contaminant_pid" -o comm= | sed -E 's/^[[:space:]]+//' || true)"
    [[ -n "$process_executable" ]] || continue
    process_cwd="$(lsof -a -p "$contaminant_pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1 || true)"
    [[ -n "$contaminants" ]] && contaminants+=$'\n'
    contaminants+="$contaminant_pid"$'\t'"$process_executable"$'\t'"${process_cwd:--}"
  done <<< "$contaminant_pids"
fi
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  print -r -- "invocation_dir=$invocation_dir"
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
  print -r -- "contaminant_pattern=$contaminant_pattern"
  print -r -- "starting_swap_used_mib=$starting_swap_mib"
  print -r -- "max_starting_swap_mib=$max_starting_swap_mib"
  print -r -- "$swap_usage"
  if [[ -n "$contaminants" ]]; then
    print -r -- $'matched_processes:\npid\texecutable\tcwd'
    print -r -- "$contaminants"
  else
    print -r -- "matched_processes=none"
  fi
} > "$output_dir/start-gate.txt"

if [[ -n "$contaminants" ]]; then
  print -u2 "another build or ML workload is active; refusing a contaminated H3 algorithm bake-off"
  print -u2 -r -- $'pid\texecutable\tcwd'
  print -u2 -r -- "$contaminants"
  print -u2 "see $output_dir/start-gate.txt"
  exit 75
fi
if awk -v used="$starting_swap_mib" -v maximum="$max_starting_swap_mib" 'BEGIN { exit !(used > maximum) }'; then
  print -u2 "starting swap is ${starting_swap_mib} MiB; refusing an H3 bake-off above the ${max_starting_swap_mib} MiB evidence ceiling"
  print -u2 "see $output_dir/start-gate.txt"
  exit 75
fi

if [[ ! -x "$executable" ]]; then
  (cd "$repo_root" && swift build -c release)
fi
if [[ ! -x "$scorer" ]]; then
  print -u2 "H3 bake-off scorer is unavailable or not executable: $scorer"
  exit 69
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

capture_system_state() {
  local destination="$1"
  {
    date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
    pmset -g therm
    sysctl vm.swapusage
    vm_stat
  } > "$destination"
}

receipt="$output_dir/receipts.tsv"
quality_receipt="$output_dir/quality.tsv"
print -r -- $'arm\tstatus\texact_kernel_mode\twall_seconds\tmax_rss_bytes\tpeak_footprint_bytes\tmlx_peak_gib\tsha256\toutput' > "$receipt"
print -r -- $'arm\tstatus\tvideo_ssim\tvideo_psnr_db\tvideo_vmaf\taudio_zero_lag_correlation\taudio_relative_l2\treport' > "$quality_receipt"
passed_arms=()
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
  print -r -- "prompt_sha256=$(shasum -a 256 "$prompt_receipt" | awk '{print $1}')"
  print -r -- "prompt_receipt=$prompt_receipt"
  print -r -- "argument_receipt=$argument_receipt"
  print -r -- "reference_receipt=$reference_receipt"
  if [[ -n "$reference_manifest" ]]; then
    print -r -- "reference_manifest=$reference_manifest"
    print -r -- "reference_manifest_sha256=$(shasum -a 256 "$reference_manifest" | awk '{print $1}')"
  else
    print -r -- "reference_manifest=unverified"
  fi
  print -r -- "executable=$executable"
  print -r -- "executable_sha256=$(shasum -a 256 "$executable" | awk '{print $1}')"
  print -r -- "contaminant_pattern=$contaminant_pattern"
  print -r -- "starting_swap_used_mib=$starting_swap_mib"
  print -r -- "max_starting_swap_mib=$max_starting_swap_mib"
  print -r -- "competing_ml_processes=none"
  sw_vers
  uname -a
  /usr/sbin/system_profiler SPHardwareDataType
  pmset -g therm
  sysctl vm.swapusage
  vm_stat
  df -h "$output_dir"
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
  local time_log="$output_dir/$arm.time.log"
  local result_code

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

  capture_system_state "$output_dir/$arm.system-before.txt"
  local start="$EPOCHREALTIME"
  set +e
  /usr/bin/time -l -o "$time_log" \
    env MERERUN_H3_EXACT_KERNELS="$exact_kernel_mode" \
    MERERUN_H3_PROFILE_STEPS=1 \
    "$executable" video generate "$prompt" \
    --model "$model" \
    --width "$width" \
    --height "$height" \
    --num-frames "$frames" \
    --seed "$seed" \
    "${extra_arguments[@]}" \
    "$@" \
    --output "$output" > "$stdout_log" 2> "$stderr_log"
  result_code=$?
  set -e
  capture_system_state "$output_dir/$arm.system-after.txt"

  local wall_seconds
  wall_seconds="$(awk -v start="$start" -v finish="$EPOCHREALTIME" 'BEGIN { printf "%.3f", finish - start }')"
  local max_rss_bytes="-"
  local peak_footprint_bytes="-"
  local mlx_peak_gib="-"
  if [[ -f "$time_log" ]]; then
    max_rss_bytes="$(awk '/maximum resident set size/ { print $1; exit }' "$time_log")"
    peak_footprint_bytes="$(awk '/peak memory footprint/ { print $1; exit }' "$time_log")"
    [[ -n "$max_rss_bytes" ]] || max_rss_bytes="-"
    [[ -n "$peak_footprint_bytes" ]] || peak_footprint_bytes="-"
  fi
  if [[ -f "$stderr_log" ]]; then
    mlx_peak_gib="$(sed -nE 's/.*peak_gib=([0-9.]+).*/\1/p' "$stderr_log" | sort -nr | head -1)"
    [[ -n "$mlx_peak_gib" ]] || mlx_peak_gib="-"
  fi
  if (( result_code != 0 )); then
    print -r -- "$arm"$'\t'"failed:$result_code"$'\t'"$exact_kernel_mode"$'\t'"$wall_seconds"$'\t'"$max_rss_bytes"$'\t'"$peak_footprint_bytes"$'\t'"$mlx_peak_gib"$'\t-'$'\t'"$output" >> "$receipt"
    return "$result_code"
  fi
  local checksum
  checksum="$(shasum -a 256 "$output" | awk '{print $1}')"
  print -r -- "$arm"$'\tpassed\t'"$exact_kernel_mode"$'\t'"$wall_seconds"$'\t'"$max_rss_bytes"$'\t'"$peak_footprint_bytes"$'\t'"$mlx_peak_gib"$'\t'"$checksum"$'\t'"$output" >> "$receipt"
  passed_arms+=("$arm")
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
    print -r -- $'render-75\tskipped:incompatible-32px-grid\tdisabled\t-\t-\t-\t-\t-\t-' >> "$receipt"
  fi
fi
if arm_enabled render-625; then
  if valid_scaled_canvas 5 8; then
    run_arm render-625 \
      --h3-acceleration quality \
      --h3-render-width "$((width * 5 / 8))" \
      --h3-render-height "$((height * 5 / 8))"
  else
    print -r -- $'render-625\tskipped:incompatible-32px-grid\tdisabled\t-\t-\t-\t-\t-\t-' >> "$receipt"
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
if arm_enabled exact-boundary-layout; then
  run_arm exact-boundary-layout --exact-kernel-mode boundary-layout --h3-acceleration quality
fi

quality_failures=0
if [[ -f "$output_dir/quality.mp4" ]]; then
  for arm in "${passed_arms[@]}"; do
    report="$output_dir/$arm.quality.json"
    set +e
    score_line="$("$scorer" "$output_dir/quality.mp4" "$output_dir/$arm.mp4" \
      --json "$report" \
      --contact-sheet "$output_dir/$arm.contact.png" \
      --expected-width "$width" \
      --expected-height "$height" \
      --expected-frames "$frames" \
      --expected-fps 24 \
      --expected-sample-rate 32000 \
      --expected-channels 2)"
    score_code=$?
    set -e
    if (( score_code == 0 )); then
      print -r -- "$score_line" >> "$quality_receipt"
    else
      print -r -- "$arm"$'\t'"failed:$score_code"$'\t-\t-\t-\t-\t-\t'"$report" >> "$quality_receipt"
      quality_failures=$((quality_failures + 1))
    fi
  done
else
  print -r -- $'quality\tmissing-baseline\t-\t-\t-\t-\t-\t-' >> "$quality_receipt"
  quality_failures=$((quality_failures + 1))
fi

print -r -- "$receipt"
print -r -- "$quality_receipt"
if (( quality_failures != 0 )); then
  exit 1
fi
