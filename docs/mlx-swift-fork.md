# mlx-swift fork policy and compiled-call overhead

`mere.run` pins the public `sawfwair/mlx-swift` fork at
`4382a34136fdefe93e12ddb7430d5a5ab0dfee21`. It contains upstream
`mlx-swift` through `19601207e9a0de51e03ee6ec0c3c5f3784275075`. The embedded
`sawfwair/mlx` revision is
`9d094c52c4001a6afc0dec9a6c411c13ec59e4d5`, which contains the exact
upstream MLX v0.32.4 release commit
`36969c094354086e15d2a327a9c50feb20b9cf11`.

The owned patch stack carries the Linux/CUDA package bridge, executor-safe
Swift streams, native affine 1-bit CUDA quantize/dequantize/QMV execution, the
Metal affine 1-bit path, custom quantized-kernel headers, NVFP4 staging, the
generation-17 NAX correctness gate, M4/H3 tuning, and executor-safe compile
cache bridging. The obsolete unaligned
1-bit fast-kernel tail was not replayed because upstream now requires aligned
fast dispatch; the fork instead tests matching host/kernel alignment directly.
Changes stay scoped to their bit width, group size, quantization mode, and
backend so stock 2-bit and wider models keep their existing paths.

The pin exposes `Stream.Context` and `Stream.withDefaultStream` over upstream's
pooled task-local CPU and GPU streams. Companion streams are created lazily,
so CPU work does not require an available GPU. Long-running servers can lease
a context to each active operation, synchronize it when the operation ends,
and reuse it for a later operation. Callers must serialize graph construction
and evaluation on each context; overlapping operations need distinct contexts.
Released streams return to upstream's pool for later scopes.

The pin also lets an MLXFast custom Metal kernel explicitly request
the core quantized helper headers. The source marker
`MLX_INCLUDE_FP_QUANTIZED_HEADERS` exposes the NVFP4 helpers, while
`MLX_INCLUDE_AFFINE_QUANTIZED_HEADERS` exposes the affine helpers. Kernels
without either marker compile from the unchanged default header set. The NAX
attention and gather-tile optimizations now live in MLX core source, so
regenerating mlx-swift's AOT sources reproduces them instead of depending on
generated-file-only commits.

The refreshed generator copies `dot.metal` and the new gather-offset,
attention-VJP, gated-delta, and fast-fence kernels into `mlx-generated/metal`.
The bundle builder compiles each generated root source once. The provenance
digest covers the generated kernel tree plus the original dot source and its
transitive core headers; `--verify-only` checks the required FP32, FP16, and
BF16 dot symbols. Fast-fence kernels are guarded by Metal 3.2 for the fork's
baseline macOS 14 shader builds. mere.run's bundle targets macOS 15 and Metal 3.2.

## v0.32.4 integration

The [core patch inventory](https://github.com/sawfwair/mlx/blob/9d094c52c4001a6afc0dec9a6c411c13ec59e4d5/FORK_PATCHES.md)
classifies all 20 previous core commits. The
[Swift inventory](https://github.com/sawfwair/mlx-swift/blob/4382a34136fdefe93e12ddb7430d5a5ab0dfee21/FORK_PATCHES.md)
classifies all 19 previous Swift commits. Shared upstream 1-bit CPU/Metal
support and CUDA shutdown behavior replace overlapping implementations.
NVFP4 staging retains the upstream global-scale path; the owned fast path
remains scoped to non-global-scale weights. The retained CPU 1-bit transposed
matmul regression additionally requires float accumulation for FP16/BF16
inputs, without changing wider affine arithmetic or test tolerances.

Final Linux portability corrections exclude Metal-only cross-entropy and
gated-delta host files and spell the dependent CPU reduction accumulator
type with `typename`. Generated GPU sources and the bundled Metal binary
remain unchanged. Its provenance revisions are updated while retaining the
original build timestamp. The CPU spelling fix passes a fresh C++ build and
all 298 tests.

On an M5 Max, the final core passes 998 Python tests on Metal (57 skips),
998 on a CPU-only build (151 skips), 298 Metal-enabled C++ tests, and 269
CPU-only C++ tests. The full Swift Xcode test plan passes 1,063 tests with
three skips and no failures, including reusable contexts, concurrent first
traces, cross-thread cache erasure, lock ordering, and the NAX gate.
Regenerating the AOT sources and copied headers again produces identical bytes.
Native BF16 qualification also passes all three installed tests and 48 GPU
fixtures with no skips. Release API arithmetic, streaming completion, a tool
round trip, and retrieval from 24,042 prompt tokens pass under a 32K budget.
Plain streaming still exposes raw reasoning markup in `delta.content`.

The exact 127,242-token retrieval probe still fails in the native release
runtime with the same decoded reasoning, answer, and token counts as the
previous pin. It also fails upstream MLX-LM using the refreshed core fork,
while stock upstream MLX 0.32.4 passes. This points toward a difference in the retained fork or
build configuration; the cause has not been isolated. The generation-17 NAX
guard is preserved because its FP16/low-bit correctness purpose must be
validated before replacement. The refresh does not qualify reliable 128K
retrieval or broad numerical parity.

All three downstream check modes complete. The standard core and installed
runtime smoke presets are absent and skip six and eleven probes respectively;
those skips do not replace the separately executed BF16 qualification above.
CUDA execution and older Apple hardware/OS execution remain unvalidated on
this host; the dependency refreshes remain draft PRs.

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
core prerequisite in place, independent stateless compiled functions no longer
take the global Swift evaluation lock. Stateful calls use upstream's recursive
instance lock with the global evaluation lock acquired first. Namespaced C
bridge functions retain every originating core cache for finalization across
Swift executors, alongside the standard mlx-c compile API. The regression suite covers concurrent first
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
