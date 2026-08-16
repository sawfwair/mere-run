# mlx-swift fork policy and compiled-call overhead

mere-run pins the public `sawfwair/mlx-swift` fork at
`5bf3e46fecfb69cd3b559025fa99885ddd188731`. It is rebased onto upstream
`mlx-swift` `da318704cc0e972b61dcca43c62cd15e545362ae`, including the upstream
`MLXArray` finalizer fix and generated-source-list maintenance. The embedded
`sawfwair/mlx` revision is
`31af89c4c21642236b8a2bc1358438512d9521e3`, based on upstream MLX
`9ab977b5649154590d598ea5d545aa1b3c97f883` and retaining the 0.32.1 ABI.

The owned patch stack carries the Linux/CUDA package bridge, executor-safe
Swift streams, native affine 1-bit CUDA quantize/dequantize/QMV execution, the
Metal affine 1-bit path, custom quantized-kernel headers, NVFP4 staging, the
generation-17 NAX correctness gate, and M4/H3 tuning. The obsolete unaligned
1-bit fast-kernel tail was not replayed because upstream now requires aligned
fast dispatch; the fork instead tests matching host/kernel alignment directly.
Changes stay scoped to their bit width, group size, quantization mode, and
backend so stock 2-bit and wider models keep their existing paths.

The current pin also lets an MLXFast custom Metal kernel explicitly request
the core quantized helper headers. The source marker
`MLX_INCLUDE_FP_QUANTIZED_HEADERS` exposes the NVFP4 helpers, while
`MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS` exposes the affine helpers. Kernels
without either marker compile from the unchanged default header set. The NAX
attention and gather-tile optimizations now live in MLX core source, so
regenerating mlx-swift's AOT sources reproduces them instead of depending on
generated-file-only commits.

## Refresh procedure

Refresh the dependency chain from the bottom up and publish it in the same
order:

1. Record immutable upstream cutoffs for both repositories and classify every
   fork-only commit as replay, replace with upstream, or drop with a regression
   test. Rebase the MLX core fork first.
2. Build an installable MLX wheel from the rebased core, run the full core gate,
   and exercise every owned quantization mode on its actual backend. Do not
   treat source compilation as runtime proof.
3. Update mlx-swift's MLX submodule to the reviewed core revision, regenerate
   AOT sources, regenerate a second time to prove the tree is idempotent, and
   run the full Xcode test plan so the Metal shader bundle is present.
4. Publish the reviewed core revision, then pin that immutable revision from
   mlx-swift and publish the reviewed Swift revision. Never make mere.run depend
   on an unpushed or floating dependency ref.
5. Pin the immutable mlx-swift revision in `Package.swift` and
   `Package.resolved`, rebuild the vendored Metal library, and update all three
   provenance fields: Swift revision, core version, and generated-source hash.
6. Run `./scripts/check.sh`, followed by both supported runtime gates:
   `MERERUN_RUN_E2E=core ./scripts/check.sh` and
   `MERERUN_RUN_E2E=installed ./scripts/check.sh`. Include a real generation
   smoke for any model family implicated by the changed stream, quantization,
   or kernel paths.
7. Open dependency-ordered draft pull requests (MLX, mlx-swift, mere.run), link
   them explicitly, and merge only after the downstream pin and platform CI are
   green against the exact advertised SHAs.

Perform this audit at least once per upstream minor release, and sooner for a
security fix or a correctness/performance change in a path mere.run owns.

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
   this class. Run it through Xcode, which builds the Metal shader bundle;
   command-line SwiftPM alone cannot build that bundle.

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
