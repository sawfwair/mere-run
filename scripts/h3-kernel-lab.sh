#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
cd "$repo_root"

mode="${1:-quick}"
export MERERUN_DIT_BENCH=1
export CFFIXED_USER_HOME="${MERERUN_H3_LAB_HOME:-${TMPDIR:-/tmp}/mere-run-h3-kernel-home}"
mkdir -p "$CFFIXED_USER_HOME"

run_release_test() {
  local filter="$1"
  local release_root="$repo_root/.build/arm64-apple-macosx/release"
  local test_binary_root="$release_root/MereRunPackageTests.xctest/Contents/MacOS"
  local test_binary="$test_binary_root/MereRunPackageTests"
  local stale_source=""

  if [[ -f "$test_binary" ]]; then
    stale_source="$(find Package.swift Sources Tests -type f -newer "$test_binary" -print -quit)"
  fi
  if [[ ! -f "$test_binary" || -n "$stale_source" || "${MERERUN_H3_LAB_REBUILD:-0}" == "1" ]]; then
    swift build --build-tests -c release -Xswiftc -enable-testing -Xswiftc -DDEBUG
  fi
  if [[ ! -f "$test_binary_root/mlx.metallib" ]]; then
    mkdir -p "$test_binary_root"
    cp "$release_root/Resources/default.metallib" "$test_binary_root/mlx.metallib"
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
  modulation)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-29018}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3AdaLNRunModulation
    ;;
  block)
    export MERERUN_H3_BENCH_ROWS="${MERERUN_H3_BENCH_ROWS:-14958}"
    export MERERUN_H3_BENCH_QUERY_TOKENS="${MERERUN_H3_BENCH_QUERY_TOKENS:-1024}"
    export MERERUN_H3_BENCH_EVAL_BATCH="${MERERUN_H3_BENCH_EVAL_BATCH:-4}"
    export MERERUN_H3_BENCH_ROUNDS="${MERERUN_H3_BENCH_ROUNDS:-2}"
    run_release_test DiTShapeBenchTests/testMiniMaxH3BlockExecutionSchedules
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
    print -u2 "usage: scripts/h3-kernel-lab.sh [quick|attention|attention-block|projections|modulation|block|dtype|turnover|boundary|gemm|gemm-block|vae|audio-parity]"
    exit 64
    ;;
esac
