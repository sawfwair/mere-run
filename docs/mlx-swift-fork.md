# mlx-swift fork policy and compiled-call overhead

`mere.run` pins the public `sawfwair/mlx-swift` fork at
`834636e7dcdc8f78d581e1e946ec7e6a3fee7765`. It contains upstream
`mlx-swift` through `19601207e9a0de51e03ee6ec0c3c5f3784275075`. The embedded
`sawfwair/mlx` revision is
`735d48ca1436dc2648073c548715af3aea5d32c7`, which contains the exact
upstream MLX v0.32.4 release commit
`36969c094354086e15d2a327a9c50feb20b9cf11`.

The owned patch stack carries the Linux/CUDA package bridge, executor-safe
Swift streams, native affine 1-bit CUDA quantize/dequantize/QMV execution, the
Metal affine 1-bit path, custom quantized-kernel headers, NVFP4 staging, the
generation-17 quantized NAX correctness gate, M4/H3 tuning, and executor-safe compile
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

Floating-point matmul, attention, and gated delta follow upstream NAX device
availability. The generation-17 workaround is local to quantized dispatch:
ordinary, gathered, sorted-RHS, and alignment decisions all require generation
18 or later. This preserves the previous quantized paths while restoring the
floating-point paths used by original BF16 checkpoints.

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

The [core patch inventory](https://github.com/sawfwair/mlx/blob/735d48ca1436dc2648073c548715af3aea5d32c7/FORK_PATCHES.md)
classifies all 20 previous core commits. The
[Swift inventory](https://github.com/sawfwair/mlx-swift/blob/834636e7dcdc8f78d581e1e946ec7e6a3fee7765/FORK_PATCHES.md)
classifies all 19 previous Swift commits. Shared upstream 1-bit CPU/Metal
support and CUDA shutdown behavior replace overlapping implementations.
NVFP4 staging retains the upstream global-scale path; the owned fast path
remains scoped to non-global-scale weights. The retained CPU 1-bit transposed
matmul regression additionally requires float accumulation for FP16/BF16
inputs, without changing wider affine arithmetic or test tolerances.

The C bindings pin the [portable I/O patch](https://github.com/sawfwair/mlx-c/pull/1)
at `dc502b106252d1899e93dd6f6da0a77a2366e80b`, based on upstream
`57731b00e8bed710e0750e7356da9f18f8eb3e20`. Error-message construction uses
`std::string` and `std::to_string`, preserving the diagnostics on Ubuntu
22.04/Jammy without requiring newer C++ library formatting support. The Swift
compile-cache bridge also omits an unused private umbrella header.

Final Linux portability corrections exclude Metal-only cross-entropy and
gated-delta host files and spell the dependent CPU reduction accumulator
type with `typename`. Generated GPU sources and the bundled Metal binary
remain unchanged. Its provenance revisions are updated while retaining the
original build timestamp. The CPU spelling fix passes a fresh C++ build and
all 298 tests.

On an M5 Max, the scoped fix passes 999 Python tests on Metal (56 skips),
including all 60 quantization cases (one CUDA-only skip), and 298 C++ tests.
The full Swift Xcode plan passes 1,064 tests with three skips and no failures,
including reusable contexts, concurrent first traces, cross-thread cache
erasure, lock ordering, and FP16/BF16 floating-point and quantized NAX checks.
The Python runner, Xcode plan, and downstream `scripts/check.sh` select
`MLX_ENABLE_TF32=0` for strict FP32 reference fixtures. Direct fixture runs
should use the same setting; production inference defaults are unchanged.
Regenerating the AOT sources and copied headers again produces identical bytes.
Native BF16 qualification also passes all three installed tests and 50 native
LFM2 fixtures with no skips. Earlier release API arithmetic, streaming completion, a tool
round trip, and retrieval from 24,042 prompt tokens pass under a 32K budget.
Plain streaming still exposes raw reasoning markup in `delta.content`.

The matched 127,242-token Python ablation isolates the former global NAX
guard: disabling NAX reproduces the old failure token-for-token, while upstream
availability returns the correct code twice with all 325 tokens matching
stock upstream. The scoped guard preserves generation-17 quantized dispatch.
A second mismatch was the terminal prompt-token forward: native prefill used
the whole final chunk for the first generation logits, while upstream leaves
one prompt token for a singleton forward. Substituting only the native
boundary into the passing Python runner breaks retrieval. Matching upstream's
boundary fixes the native release probe: all 127,242 prompt tokens are consumed,
the answer is exactly `TANGERINE-4827`, and generation stops after 295 tokens
within the unchanged 2,048-token budget. The run uses original BF16 weights,
production precision defaults, and a fresh server with no reused prompt cache.
Broader long-context retrieval quality and numerical parity remain unqualified.

The three installed native BF16 checks pass with production precision defaults.
All 50 native LFM2 fixtures pass with FP32 reference precision, including
terminal-token boundary, semantic-checkpoint, and resumed-prefill regressions.
All 22 application quantized/binary-linear GPU checks pass. The small-batch
Q4 kernel promotes BF16 inputs to FP32 before adding the bias sum, matching
upstream serial QMV arithmetic. This restores bit-exact small and wide batching;
a cancellation regression covers positive and negative residuals across batch
widths 2–9.

The final downstream local gate passes 5,561 XCTest cases (476 skips) and
all 196 Swift Testing cases. Earlier refresh runs also complete both E2E
check modes. The standard core and installed
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
