# mlx-swift fork policy and compiled-call overhead

mere-run pins the public `sawfwair/mlx-swift` fork based on upstream 0.32.1.
That fork carries the Linux package bridge and Prism low-bit compatibility
needed by this repository. Its embedded `sawfwair/mlx` revision also provides
native affine 1-bit CUDA quantize, dequantize, and QMV execution, plus the
Metal affine 1-bit path and the generation-17 NAX correctness gate. The fork
retains upstream's NVFP4 split-K fix and drops the old duplicate `qmv_wide`
implementation now supplied upstream. Changes stay scoped to their bit width,
group size, quantization mode, and backend so stock 2-bit and wider models keep
their existing paths.

A separate measured one-line compiled-call optimization exists on staging
branches but is **deliberately not included in the pin**. The rest of this
document records why, and what a safe version of that fix requires.

## The finding

`CompiledFunction.call` (Source/MLX/Transforms+Compile.swift) takes the global
`evalLock` and rebuilds a closure trampoline on every invocation. At decode
call rates this is a cliff, measured on an M4 Max (Gemma4 12B, 96 compiled
calls/token):

| | per-token wall | per-call cost |
|---|---|---|
| stock 0.31.4 | 70.8 ms (2.6× slower than uncompiled) | ~0.45 ms |
| lock removed | 28.2 ms (parity) | ~5 µs (= one raw op) |

Every MLXNN compiled activation (`geluApproximate`, swiglu helpers, …) pays
this cost per call, not just explicit `compile(...)` users. Confirmed present
through upstream v0.31.6 (the file is unchanged upstream since 2026-05-07).

## Why the one-line fix is not shipped

Removing the global lock is only proven safe for the paths our benchmarks
exercised: repeated calls to already-traced functions from one generator at a
time. It is **not** established safe for the path our architecture actually
exercises in production:

- A **first call** (or first call at a new shape) runs a trace:
  `mlx_detail_compile` consults and mutates the **global compile cache**, and
  tracing re-enters arbitrary user code. The global `evalLock` was the only
  cross-function serialization of that path; the per-function `NSLock` does
  not cover two *different* compiled functions racing their first traces.
- mlx-swift's own comment documents the lock as guarding eval re-entrancy,
  i.e. the maintainers treat this path as not thread-safe without it.
- mere-run runs concurrent generator actors (text serving while image/LoRA
  training, multiple model families resident) — exactly the workload that can
  first-trace independent compiled functions on different threads at once.

No runtime crash has been reproduced either way (a targeted
`testCompileThreadSafety` attempt died on an unrelated `default.metallib`
packaging issue, and all passing tests were effectively single-generator).
For a serving runtime the burden of proof is on the patch, and
"maintainer-documented not-thread-safe + globals on the unlocked path + our
architecture exercises it" is past the threshold to hold it back.

## The paired fix this needs (before re-attempting)

1. **Core-side thread safety**: bump/patch the vendored mlx core so compile
   tracing state is thread-local and the compile cache has its own narrow
   lock, making a cache-hit `mlx_closure_apply` genuinely lock-free-safe and
   first-traces mutually safe.
2. **Swift-side lock removal** (the existing one-liner), only on top of (1).
3. **A stress test that can actually fail**: concurrently first-trace many
   *independent* compiled functions (and new shapes of shared ones) from
   multiple threads. Repeated calls to a single closure — including the
   existing `MLXCompiledFunctionOverheadTests` micro-bench — cannot catch
   this class. (Prerequisite: fix the `default.metallib` packaging issue that
   currently blocks Metal-backed concurrency tests in the test bundle.)

Do **not** upstream or re-pin the lock removal alone.

## Staging branches (kept on the fork)

- <https://github.com/sawfwair/mlx-swift>
  - `mere-run/compiled-call-evallock` @ `7450480` — v0.31.4 + lock removal
  - `mere-run/compiled-call-evallock-0.31.6` @ `f004aa1` — v0.31.6 + same
- Measurement harness in-repo:
  `MERERUN_BENCHMARK_COMPILE_OVERHEAD=1 swift test --filter MLXCompiledFunctionOverheadTests`

## Scope notes

- Shipped performance in this repo does not depend on the patch: the
  compiled-segments decode path defaults off, and every benchmark number in
  the perf PR matches stock-dependency runs.
- The Linux/CUDA prebuilt path builds the exact embedded MLX revision selected
  by this pin; do not replace it with a floating checkout.
- The vendored Metal library is accepted only when its stamp matches the exact
  mlx-swift revision, MLX core version, and generated-kernel source hash
  compiled into `mere.run`. `scripts/check.sh` recomputes the same provenance
  from the clean pinned checkout.
- Never edit `.build/checkouts/` to change dependency behavior — checkout
  edits silently vanish on the next `swift package resolve/update`.
