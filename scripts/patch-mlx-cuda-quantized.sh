#!/usr/bin/env bash
# Patch pinned MLX CUDA quantized kernels for synchronization and BF16 accuracy.
set -euo pipefail

cuda_backend="${1:?Usage: patch-mlx-cuda-quantized.sh CUDA_BACKEND_DIRECTORY}"
kernel="$cuda_backend/device/qmm_sm80.cuh"
launcher="$cuda_backend/quantized/qmm/qmm_sm80.cu"
qmv="$cuda_backend/quantized/qmm/qmv.cu"
marker=MERERUN_CUDA_QMM_SM80_EPILOGUE_BARRIER
qmv_marker=MERERUN_CUDA_BF16_QMV_FLOAT_ACCUMULATION

if grep -Fq "$marker" "$kernel" && grep -Fq 'mr_epilogue1' "$launcher" && grep -Fq "$qmv_marker" "$qmv"; then
  exit 0
fi

kernel_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-qmm-kernel.XXXXXX")"
launcher_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-qmm-launcher.XXXXXX")"
qmv_tmp="$(mktemp "${TMPDIR:-/tmp}/mererun-qmv.XXXXXX")"
trap 'rm -f "$kernel_tmp" "$launcher_tmp" "$qmv_tmp"' EXIT

if grep -Fq "$marker" "$kernel"; then
  cp "$kernel" "$kernel_tmp"
elif ! awk '
  $0 == "  // Epilogue." {
    count++
    print "  // MERERUN_CUDA_QMM_SM80_EPILOGUE_BARRIER: drain all asynchronous"
    print "  // copies before reusing the mainloop shared-memory union for output."
    print "  cp_async_wait<0>();"
    print "  __syncthreads();"
    print ""
  }
  { print }
  END { if (count != 1) exit 42 }
' "$kernel" >"$kernel_tmp"; then
  echo '[prepare-linux-native] error: unrecognized MLX SM80 QMM epilogue.' >&2
  exit 69
fi

# MLX caches PTX by module name. Change the name so an older cached kernel
# cannot mask the synchronization fix after rebuilding the bridge.
if grep -Fq 'mr_epilogue1' "$launcher"; then
  cp "$launcher" "$launcher_tmp"
elif ! awk '
  /"qmm_sm80_tn_\{\}_m\{\}_b\{\}_g\{\}_\{\}",/ {
    count++
    sub(/qmm_sm80_tn_\{\}_m\{\}_b\{\}_g\{\}_\{\}/,
        "qmm_sm80_tn_{}_m{}_b{}_g{}_{}_mr_epilogue1")
  }
  { print }
  END { if (count != 1) exit 42 }
' "$launcher" >"$launcher_tmp"; then
  echo '[prepare-linux-native] error: unrecognized MLX SM80 QMM cache key.' >&2
  exit 69
fi

if grep -Fq "$qmv_marker" "$qmv"; then
  cp "$qmv" "$qmv_tmp"
elif ! awk '
  $0 == "  cuda::std::conditional_t<(bits == 1 || bits >= 8), float, T>" {
    count++
    print "  // MERERUN_CUDA_BF16_QMV_FLOAT_ACCUMULATION: keep BF16 partial sums"
    print "  // in FP32, matching the tensor-core path and avoiding cancellation loss."
    print "  cuda::std::conditional_t<"
    print "      (bits == 1 || bits >= 8 || cuda::std::is_same_v<T, cutlass::bfloat16_t>),"
    print "      float, T>"
    next
  }
  { print }
  END { if (count != 1) exit 42 }
' "$qmv" >"$qmv_tmp"; then
  echo '[prepare-linux-native] error: unrecognized MLX QMV accumulator policy.' >&2
  exit 69
fi

# Replace files rather than writing through SwiftPM's read-only checkout modes.
cmp -s "$kernel_tmp" "$kernel" || mv -f "$kernel_tmp" "$kernel"
cmp -s "$launcher_tmp" "$launcher" || mv -f "$launcher_tmp" "$launcher"
cmp -s "$qmv_tmp" "$qmv" || mv -f "$qmv_tmp" "$qmv"
echo '[prepare-linux-native] patched MLX quantized synchronization, PTX cache key, and BF16 accumulation.'
