#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
kernel_lab="$script_dir/h3-kernel-lab.sh"
output_dir=""
resume=0
modes=()

usage() {
  print -u2 "usage: scripts/h3-kernel-suite.sh [--output DIR] [--resume] [MODE ...]"
}

while (( $# > 0 )); do
  case "$1" in
    --output)
      if (( $# < 2 )); then
        usage
        exit 64
      fi
      output_dir="$2"
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      modes+=("$@")
      break
      ;;
    -*)
      print -u2 "unsupported H3 kernel-suite option: $1"
      usage
      exit 64
      ;;
    *)
      modes+=("$1")
      shift
      ;;
  esac
done

if (( ${#modes} == 0 )); then
  modes=(
    gate-adaln
    gate-adaln-int8
    qkv-layout
    qkv-direct
    affine-oproj
    affine-ffn
    buffer-alias
    exact-ref2va
  )
fi

typeset -A seen_modes
for mode in "${modes[@]}"; do
  if [[ ! "$mode" =~ '^[a-z0-9-]+$' ]]; then
    print -u2 "invalid H3 kernel-suite mode: $mode"
    exit 64
  fi
  if (( ${+seen_modes[$mode]} )); then
    print -u2 "duplicate H3 kernel-suite mode: $mode"
    exit 64
  fi
  seen_modes[$mode]=1
done

if [[ -z "$output_dir" ]]; then
  output_dir="$repo_root/.build/h3-kernel-suite/$(date -u '+%Y%m%dT%H%M%SZ')"
fi
if [[ "$output_dir" == *$'\t'* || "$output_dir" == *$'\n'* ]]; then
  print -u2 "H3 kernel-suite output paths cannot contain tabs or newlines"
  exit 64
fi
output_dir="${output_dir:A}"
receipt="$output_dir/suite.tsv"
current_commit="$(git -C "$repo_root" rev-parse HEAD)"
mode_key="${(j:,:)modes}"

if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  print -u2 "H3 kernel-suite evidence requires a clean worktree"
  exit 75
fi

if [[ -e "$receipt" && "$resume" != "1" ]]; then
  print -u2 "H3 kernel-suite receipt already exists; pass --resume to reuse it: $receipt"
  exit 73
fi
if [[ -e "$output_dir" && ! -d "$output_dir" ]]; then
  print -u2 "H3 kernel-suite output exists but is not a directory: $output_dir"
  exit 73
fi
if [[ -d "$output_dir" && ! -e "$receipt" && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  print -u2 "H3 kernel-suite output directory is not empty: $output_dir"
  exit 73
fi
if [[ "$resume" == "1" && ! -e "$receipt" ]]; then
  print -u2 "H3 kernel-suite receipt is unavailable for resume: $receipt"
  exit 66
fi
if [[ "$resume" == "1" ]]; then
  if [[ ! -f "$output_dir/commit.txt" || ! -f "$output_dir/mode-key.txt" ]]; then
    print -u2 "H3 kernel-suite identity receipts are unavailable for resume: $output_dir"
    exit 66
  fi
  recorded_commit="$(<"$output_dir/commit.txt")"
  if [[ "$recorded_commit" != "$current_commit" ]]; then
    print -u2 "H3 kernel-suite resume commit mismatch: expected $recorded_commit, found $current_commit"
    exit 65
  fi
  recorded_mode_key="$(<"$output_dir/mode-key.txt")"
  if [[ "$recorded_mode_key" != "$mode_key" ]]; then
    print -u2 "H3 kernel-suite resume mode mismatch: expected $recorded_mode_key, found $mode_key"
    exit 65
  fi
fi

mkdir -p "$repo_root/.build"
lock_dir="$repo_root/.build/h3-kernel-suite.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  stale_owner_pid=""
  if [[ -f "$lock_dir/owner.tsv" ]]; then
    stale_owner_pid="$(awk -F '\t' 'NR == 2 { print $1; exit }' "$lock_dir/owner.tsv")"
  fi
  if [[ "$stale_owner_pid" =~ '^[0-9]+$' ]] && ! kill -0 "$stale_owner_pid" 2>/dev/null; then
    print -u2 "recovering stale H3 kernel-suite lock from exited PID $stale_owner_pid"
    rm -f "$lock_dir/owner.tsv"
    rmdir "$lock_dir" 2>/dev/null || true
    if ! mkdir "$lock_dir" 2>/dev/null; then
      print -u2 "could not reacquire H3 kernel-suite lock after stale-owner recovery"
      exit 75
    fi
  else
    print -u2 "another H3 kernel suite owns $lock_dir"
    if [[ -f "$lock_dir/owner.tsv" ]]; then
      sed -n '1,20p' "$lock_dir/owner.tsv" >&2
    fi
    exit 75
  fi
fi

cleanup_lock() {
  rm -f "$lock_dir/owner.tsv"
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

{
  print -r -- $'pid\tstarted_utc\toutput'
  print -r -- "$$"$'\t'"$(date -u '+%Y-%m-%dT%H:%M:%SZ')"$'\t'"$output_dir"
} > "$lock_dir/owner.tsv"

mkdir -p "$output_dir"
if [[ ! -e "$receipt" ]]; then
  print -r -- $'mode\tattempt\tstatus\texit_code\tstarted_utc\tended_utc\tstdout\tstderr\tstart_gate' > "$receipt"
  print -r -- "$current_commit" > "$output_dir/commit.txt"
  print -r -- "$mode_key" > "$output_dir/mode-key.txt"
  print -r -- $'started_utc\tresume\tmodes' > "$output_dir/invocations.tsv"
  {
    date -u '+created_utc=%Y-%m-%dT%H:%M:%SZ'
    print -r -- "commit=$current_commit"
    print -r -- "worktree=clean"
    sw_vers
    sysctl -n machdep.cpu.brand_string
    sysctl hw.memsize
    sysctl vm.swapusage
    pmset -g therm
  } > "$output_dir/environment.txt"
fi

rm -f "$output_dir/completed.txt"
print -r -- "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"$'\t'"$resume"$'\t'"$mode_key" >> "$output_dir/invocations.tsv"

overall_code=0
for mode in "${modes[@]}"; do
  mode_root="$output_dir/$mode"
  passed_marker="$mode_root/passed"
  if [[ "$resume" == "1" && -f "$passed_marker" ]] && \
      grep -qx "commit=$current_commit" "$passed_marker"; then
    print -r -- "skip $mode: existing pass marker"
    continue
  fi

  attempt_number="$(awk -F '\t' -v selected="$mode" '
    NR > 1 && $1 == selected { attempts++ }
    END { print attempts + 1 }
  ' "$receipt")"
  mode_dir="$mode_root/attempt-$attempt_number"
  mkdir -p "$mode_dir"
  stdout_log="$mode_dir/stdout.log"
  stderr_log="$mode_dir/stderr.log"
  start_gate="$mode_dir/start-gate.txt"
  started_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  print -r -- "run $mode"

  set +e
  MERERUN_H3_LAB_START_GATE_RECEIPT="$start_gate" \
    "$kernel_lab" "$mode" > "$stdout_log" 2> "$stderr_log"
  mode_code=$?
  set -e
  ended_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  if (( mode_code == 0 )); then
    mode_status="passed"
    {
      print -r -- "commit=$current_commit"
      print -r -- "attempt=$attempt_number"
      print -r -- "ended_utc=$ended_utc"
    } > "$passed_marker"
  elif (( mode_code == 75 )); then
    mode_status="start-gate-rejected"
  else
    mode_status="failed"
    overall_code=1
  fi

  print -r -- "$mode"$'\t'"$attempt_number"$'\t'"$mode_status"$'\t'"$mode_code"$'\t'"$started_utc"$'\t'"$ended_utc"$'\t'"$stdout_log"$'\t'"$stderr_log"$'\t'"$start_gate" >> "$receipt"

  if (( mode_code == 75 )); then
    print -u2 "H3 kernel suite stopped at the clean-host gate for $mode"
    print -u2 "see $start_gate"
    exit 75
  fi
done

if (( overall_code == 0 )); then
  date -u '+completed_utc=%Y-%m-%dT%H:%M:%SZ' > "$output_dir/completed.txt"
fi
exit "$overall_code"
