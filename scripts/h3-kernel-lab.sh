#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
cd "$repo_root"

mode="${1:-quick}"
export MERERUN_DIT_BENCH=1
export CFFIXED_USER_HOME="${MERERUN_H3_LAB_HOME:-${TMPDIR:-/tmp}/mere-run-h3-kernel-home}"
mkdir -p "$CFFIXED_USER_HOME"

start_gate_receipt="${MERERUN_H3_LAB_START_GATE_RECEIPT:-$repo_root/.build/h3-kernel-lab/start-gate.txt}"
max_starting_swap_mib="${MERERUN_H3_LAB_MAX_STARTING_SWAP_MIB:-1024}"
if [[ ! "$max_starting_swap_mib" =~ '^[0-9]+([.][0-9]+)?$' ]]; then
  print -u2 "MERERUN_H3_LAB_MAX_STARTING_SWAP_MIB must be a nonnegative number"
  exit 64
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
mkdir -p "${start_gate_receipt:h}"
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  print -r -- "mode=$mode"
  print -r -- "commit=$(git -C "$repo_root" rev-parse HEAD)"
  if [[ -z "$(git -C "$repo_root" status --porcelain)" ]]; then
    print -r -- "worktree=clean"
  else
    print -r -- "worktree=dirty"
  fi
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
} > "$start_gate_receipt"

if [[ -n "$contaminants" ]]; then
  print -u2 "another build or ML workload is active; refusing a contaminated H3 kernel benchmark"
  print -u2 -r -- $'pid\texecutable\tcwd'
  print -u2 -r -- "$contaminants"
  print -u2 "see $start_gate_receipt"
  exit 75
fi
if awk -v used="$starting_swap_mib" -v maximum="$max_starting_swap_mib" 'BEGIN { exit !(used > maximum) }'; then
  print -u2 "starting swap is ${starting_swap_mib} MiB; refusing an H3 kernel benchmark above the ${max_starting_swap_mib} MiB evidence ceiling"
  print -u2 "see $start_gate_receipt"
  exit 75
fi

run_release_test() {
  local filter="$1"
  local release_root="$repo_root/.build/arm64-apple-macosx/release"
  local test_binary_root="$release_root/MereRunPackageTests.xctest/Contents/MacOS"
  local test_binary="$test_binary_root/MereRunPackageTests"
  local default_metallib="$release_root/Resources/default.metallib"
  local stale_source=""

  if [[ -f "$test_binary" ]]; then
    stale_source="$(find Package.swift Sources Tests -type f -newer "$test_binary" -print -quit)"
  fi
  if [[ ! -f "$test_binary" || -n "$stale_source" || "${MERERUN_H3_LAB_REBUILD:-0}" == "1" ]]; then
    swift build --build-tests -c release -Xswiftc -enable-testing -Xswiftc -DDEBUG
  fi
  if [[ ! -f "$test_binary_root/mlx.metallib" ]]; then
    if [[ ! -f "$default_metallib" ]]; then
      default_metallib="$release_root/magentart.framework/Resources/default.metallib"
    fi
    mkdir -p "$test_binary_root"
    cp "$default_metallib" "$test_binary_root/mlx.metallib"
  fi
  xcrun xctest -XCTest "$filter" "$release_root/MereRunPackageTests.xctest"
}

case "$mode" in
  quick)
    export MERERUN_H3_BENCH_SEARCH=coordinate
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_CHUNKS="${MERERUN_H3_BENCH_CHUNKS:-1024,1536,2048,2560,3072,4096}"
    export MERERUN_H3_BENCH_HEAD_CHUNKS="${MERERUN_H3_BENCH_HEAD_CHUNKS:-14,28,56}"
    export MERERUN_H3_BENCH_EVAL_BATCHES="${MERERUN_H3_BENCH_EVAL_BATCHES:-1,2,4,8}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-2}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3AttentionChunkSizes
    ;;
  attention)
    export MERERUN_H3_BENCH_SEARCH="${MERERUN_H3_BENCH_SEARCH:-grid}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3AttentionChunkSizes
    ;;
  projections)
    run_release_test DiTShapeBenchTests/testMiniMaxH3QmmVsResidentBF16
    ;;
  mpp-projections)
    export MERERUN_H3_MPP_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test \
      MiniMaxH3MPPProjectionTests/testProductionShapeReleaseBenchmark
    ;;
  modulation)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-29018}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3AdaLNRunModulation
    ;;
  gate-adaln)
    export MERERUN_H3_FUSED_KERNEL_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test \
      MiniMaxH3FusedKernelTests/testGateAttentionAndPrepareFeedForwardReleaseBenchmark
    ;;
  gate-adaln-int8)
    export MERERUN_H3_FUSED_INT8_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test \
      MiniMaxH3FusedKernelTests/testGateAttentionAndQuantizeFeedForwardReleaseBenchmark
    ;;
  qkv-layout)
    export MERERUN_H3_QKV_LAYOUT_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test \
      MiniMaxH3FusedKernelTests/testPrepareHeadMajorQKVReleaseBenchmark
    ;;
  qkv-direct)
    export MERERUN_H3_QKV_DIRECT_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test \
      MiniMaxH3FusedKernelTests/testProjectHeadMajorQKVAffineInt8ReleaseBenchmark
    ;;
  affine-oproj)
    export MERERUN_H3_AFFINE_OPROJ_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test \
      MiniMaxH3FusedKernelTests/testProjectHeadMajorAttentionAffineInt8ReleaseBenchmark
    ;;
  affine-ffn)
    export MERERUN_H3_AFFINE_FFN_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test \
      MiniMaxH3FusedKernelTests/testFusedFeedForwardAffineInt8ReleaseBenchmark
    ;;
  buffer-alias)
    export MERERUN_H3_BUFFER_DONATION_BENCH=1
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    run_release_test \
      MiniMaxH3FusedKernelTests/testCompiledResidualBoundaryDonationReleaseBenchmark
    ;;
  exact-ref2va)
    default_ref2va_root="${HOME}/Library/Application Support/MereRun/models/video-minimax-h3-ref2va-mlx"
    export MERERUN_TEST_MLX_DEVICE=gpu
    export MERERUN_H3_EXACT_FULL_FORWARD=1
    export MERERUN_H3_EXACT_STAGE_DIAGNOSTICS="${MERERUN_H3_EXACT_STAGE_DIAGNOSTICS:-1}"
    export MERERUN_H3_EXACT_KERNEL_MODEL_ROOT="${MERERUN_H3_EXACT_KERNEL_MODEL_ROOT:-$default_ref2va_root}"
    run_release_test \
      MiniMaxH3Tests/testInstalledRef2VAExactKernelFullForwardWhenEnabled
    ;;
  block)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_QUERY_TOKENS="${MERERUN_H3_BENCH_QUERY_TOKENS:-1024}"
    export MERERUN_H3_BENCH_EVAL_BATCH="${MERERUN_H3_BENCH_EVAL_BATCH:-4}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-2}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3BlockExecutionSchedules
    ;;
  post)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-73470}"
    export MERERUN_H3_BENCH_QUERY_TOKENS="${MERERUN_H3_BENCH_QUERY_TOKENS:-640}"
    export MERERUN_H3_BENCH_HEADS="${MERERUN_H3_BENCH_HEADS:-8}"
    export MERERUN_H3_BENCH_EVAL_BATCH="${MERERUN_H3_BENCH_EVAL_BATCH:-1}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-2}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3BlockPostAttentionSchedules
    ;;
  dtype)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-37966}"
    export MERERUN_H3_BENCH_QUERY_TOKENS="${MERERUN_H3_BENCH_QUERY_TOKENS:-768}"
    export MERERUN_H3_BENCH_EVAL_BATCH="${MERERUN_H3_BENCH_EVAL_BATCH:-1}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-3}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3BlockDTypes
    ;;
  turnover)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_QUERY_TOKENS="${MERERUN_H3_BENCH_QUERY_TOKENS:-1024}"
    export MERERUN_H3_BENCH_EVAL_BATCH="${MERERUN_H3_BENCH_EVAL_BATCH:-4}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-2}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3BlockWeightTurnover
    ;;
  boundary)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_QUERY_TOKENS="${MERERUN_H3_BENCH_QUERY_TOKENS:-1024}"
    export MERERUN_H3_BENCH_EVAL_BATCH="${MERERUN_H3_BENCH_EVAL_BATCH:-4}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-2}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3TwoBlockBoundaryFusion
    ;;
  attention-block)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-37966}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-4}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3BlockAttentionSchedules
    ;;
  gemm)
    run_release_test DiTShapeBenchTests/testMiniMaxH3MetalGEMMProjectionShapes
    ;;
  gemm-block)
    run_release_test DiTShapeBenchTests/testMiniMaxH3BlockGEMMSchedules
    ;;
  vae)
    run_release_test DiTShapeBenchTests/testMiniMaxH3VideoVAETileSize
    ;;
  audio-parity)
    run_release_test MiniMaxH3Tests/testInstalledAudioVAEDecodeMatchesReference
    ;;
  *)
    print -u2 "usage: scripts/h3-kernel-lab.sh [quick|attention|attention-block|projections|mpp-projections|modulation|gate-adaln|gate-adaln-int8|qkv-layout|qkv-direct|affine-oproj|affine-ffn|buffer-alias|exact-ref2va|block|post|dtype|turnover|boundary|gemm|gemm-block|vae|audio-parity]"
    exit 64
    ;;
esac
