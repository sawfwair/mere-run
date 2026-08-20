# mlx-swift fork policy and compiled-call overhead

`mere.run` pins the public `sawfwair/mlx-swift` fork at
`7558b9cff75746e3ce25802aecbdc498b240af7f`. It contains upstream
`mlx-swift` through `97cf19efeaa4e929415e75982e999adb34f62c0d`. The embedded
`sawfwair/mlx` revision is
`11da2b33a51772c023e2f7d7bc4ba9b3ff7e03ef`, which contains the exact
upstream MLX v0.32.1 release commit
`3a6219917e4535575ce5bce2fc2ba27a483a709b`.

The owned patch stack carries the Linux/CUDA package bridge, executor-safe
Swift streams, native affine 1-bit CUDA quantize/dequantize/QMV execution, the
Metal affine 1-bit path, custom quantized-kernel headers, NVFP4 staging, the
generation-17 NAX correctness gate, M4/H3 tuning, and executor-safe compile
cache bridging. The obsolete unaligned
1-bit fast-kernel tail was not replayed because upstream now requires aligned
fast dispatch; the fork instead tests matching host/kernel alignment directly.
Changes stay scoped to their bit width, group size, quantization mode, and
backend so stock 2-bit and wider models keep their existing paths.

The pin also lets an MLXFast custom Metal kernel explicitly request
the core quantized helper headers. The source marker
`MLX_INCLUDE_FP_QUANTIZED_HEADERS` exposes the NVFP4 helpers, while
`MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS` exposes the affine helpers. Kernels
without either marker compile from the unchanged default header set. The NAX
attention and gather-tile optimizations now live in MLX core source, so
regenerating mlx-swift's AOT sources reproduces them instead of depending on
generated-file-only commits.

The vendored metallib also compiles MLX core's `dot.metal` directly. MLX 0.32.1
does not copy that always-required source into `mlx-generated/metal`, while the
host runtime still dispatches its FP32, FP16, and BF16 symbols. The provenance
digest therefore covers the core source and its transitive headers in addition
to the generated kernel tree, and `--verify-only` checks the resulting symbols.

## Refresh procedure

Refresh and publish the dependency chain in dependency order:

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
   mlx-swift and publish the reviewed Swift revision. Do not make `mere.run` depend
   on an unpushed or floating dependency ref.
5. Pin the immutable mlx-swift revision in `Package.swift` and
   `Package.resolved`, rebuild the vendored Metal library, and update its
   enforced provenance: Swift revision, core version and revision, upstream
   release tag and revision, and generated-source hash.
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

The previously staged compiled-call optimization is included. MLX v0.32.1
makes compile caches thread-local and cache erasure thread-safe, and the Swift
bridge retains the originating cache identities across executors. With that
core prerequisite in place, independent compiled functions no longer take the
global Swift evaluation lock. The regression suite covers concurrent first
traces, shape changes, numerical evaluation, and cross-thread cache erasure.

An eight-worker, 2,000-call-per-worker comparison on the same machine isolates that
lock change on the same v0.32.1 core. Across five alternating trials, median
compiled graph-build wall time fell from 7.56 us/call to 1.38 us/call (5.5x
higher concurrent call throughput); median build-plus-evaluation wall time fell
from 13.53 us/call to 7.61 us/call. Reproduce it in mlx-swift with
`MLX_SWIFT_BENCHMARK_COMPILE_CONCURRENCY=1 swift test --filter
TransformTests.testConcurrentCompiledCallOverheadMicrobench`.

## The finding

`CompiledFunction.call` (Source/MLX/Transforms+Compile.swift) takes the global
`evalLock` and rebuilds a closure trampoline on every invocation. At decode
call rates this is a cliff, measured on an M4 Max (Gemma4 12B, 96 compiled
calls/token):

| Variant | Per-token wall time | Per-call cost |
|---|---|---|
| stock 0.31.4 | 70.8 ms (2.6× slower than uncompiled) | ~0.45 ms |
| lock removed | 28.2 ms (parity) | ~5 µs (= one raw op) |

Every MLXNN compiled activation (`geluApproximate`, swiglu helpers, …) pays
this cost per call, including code that does not call `compile(...)` directly. Confirmed present
through upstream v0.31.6 (the file is unchanged upstream since 2026-05-07).

## Why the one-line fix was previously held back

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
  which means the maintainers treat this path as not thread-safe without it.
- mere-run runs concurrent generator actors (text serving while image/LoRA
  training, multiple model families resident) — exactly the workload that can
  first-trace independent compiled functions on different threads at once.

No runtime crash has been reproduced either way (a targeted
`testCompileThreadSafety` attempt died on an unrelated `default.metallib`
packaging issue, and all passing tests were effectively single-generator).
For a serving runtime the burden of proof is on the patch, and
"maintainer-documented not-thread-safe + globals on the unlocked path + our
architecture exercises it" is past the threshold to hold it back.

## Required paired fix

1. **Core-side thread safety:** Update the vendored MLX core so compile
   tracing state is thread-local and the compile cache has its own narrow
   lock, making a cache-hit `mlx_closure_apply` genuinely lock-free-safe and
   first-traces mutually safe.
2. **Swift-side lock removal:** Apply the existing one-line change only after
   the core-side change.
3. **A stress test that can detect the failure:** Concurrently first-trace many
   *independent* compiled functions (and new shapes of shared ones) from
   multiple threads. Repeated calls to a single closure — including the
   existing `MLXCompiledFunctionOverheadTests` micro-bench — cannot catch
   this class. Run it through Xcode, which builds the Metal shader bundle;
   command-line SwiftPM alone cannot build that bundle.

Do **not** replay the lock removal onto an older core without the cache-scoped
erase bridge and the concurrency regressions.

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
  mlx-swift revision, MLX core version and revision, incorporated upstream
  release, and generated-kernel source hash compiled into `mere.run`.
  `scripts/check.sh` recomputes the same provenance and release ancestry from
  the clean pinned checkout.
- Do not edit `.build/checkouts/` to change dependency behavior. Checkout
  edits disappear during the next `swift package resolve` or `swift package update`.
