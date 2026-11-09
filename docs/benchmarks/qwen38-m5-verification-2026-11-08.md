# Qwen3.8 M5 matrix verification and fused feed-forward

The partitioned tiled M5 backend improves qualified greedy DFlash 2 decoding:
code reached 102.8 TPS versus 80.7 TPS for reference, and prose reached 30.8
versus 25.0 TPS. Reversed-order code also improved, reaching 107.3 versus
78.2 TPS. All timing conditions passed resource qualification. The backend
was opt-in during measurement because it changes the numerical reference.
The full repository gate passed on a direct rerun. The November 9 policy now
defaults to DFlash 2 and tiled M5 kernels for eligible Q4 text requests on
supported hardware with the installed draft; broader quality evaluation is
pending. See the [sampled report's policy follow-up](qwen38-sampled-dflash2-2026-11-08.md#default-policy-follow-up).

These results use temperature 0 with thinking disabled. They do not establish
throughput at Qwen's recommended sampled settings. At measurement time the Qwen
DFlash 2 decoder required greedy generation without active penalties. Sampled
verification has since been implemented with local distribution and cache
tests. Qualified code timings at the published presets appear in the
[sampled throughput report](qwen38-sampled-dflash2-2026-11-08.md), including code
and prose with both presets. Non-thinking prose regresses against target-only
decoding. The default accepts that tradeoff; broader workload and quality
evaluation remain pending.

[Qwen's recommended presets](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)
use temperature 1.0, top-p 0.95, and top-k 20 for thinking; non-thinking uses
temperature 0.7, top-p 0.80, top-k 20, and presence penalty 1.5. Both use min-p 0
and repetition penalty 1.0; thinking uses presence penalty 0.

## Qualified partitioned results

The final control uses tile-major Q4 buffers, 32-channel stages, interleaved
gate/up, and two K partitions for down and output projections with 5,120
outputs. Other supported projections use one partition. Each pair uses the
same release binary, pinned target and draft, prompt, and generation settings.
There is one 256-token warm-up and three measured requests per condition.

| Protocol / workload | Reference TPS, median (range) | Partitioned TPS, median (range) | Change |
| --- | ---: | ---: | ---: |
| Fixed 512 / code | 80.7 (76.2–88.2) | 102.8 (101.4–116.7) | +27.3% |
| Fixed 512 / code, reversed order | 78.2 (78.1–86.3) | 107.3 (98.0–111.4) | +37.3% |
| Fixed 512 / prose | 25.0 (24.4–25.2) | 30.8 (30.6–31.8) | +22.8% |
| 1,024 cap, EOS stopping / code | 70.6 (69.5–80.0) | 93.8 (91.2–104.2) | +32.9% |
| 1,024 cap, EOS stopping / prose | 24.8 (24.0–24.9) | 34.0 (33.9–35.8) | +37.1% |

The fixed code conditions produce the same rendered output with both backends,
use 75 verification rounds, and accept 437 of 519 draft tokens. Their gain
does not depend on better draft acceptance. Fixed prose changes continuation:
the partitioned backend uses 214 rounds versus 218 for reference. Both match
their own serial oracle. The prose oracle was contended and is retained only
for correctness; its timing is excluded.

EOS-stopping continuations also differ. Code generates 1,017 tokens with the
partitioned backend versus 970 for reference; prose generates 594 versus 601.
These request-throughput observations do not establish equivalent quality.
The earlier matched Splash code median was 121.5 TPS, compared with 93.8 TPS
here under the corresponding EOS protocol. Splash was not rerun in this timing
window, so these results show progress without establishing parity with Splash.

The final release SHA-256 is
`767036cca2bd44bc4b56c972e2fbe42466b2cc9b916eea1508e31d880edbec2b`.
The machine-readable summary retains every request, source hashes, resource
qualification, and the preceding controls. Later sections describe earlier
revisions and the changes that led to this result; their rates apply to those
revisions.

## Implementation

The backend uses Metal 4 `matmul2d` with BF16 activations and FP16 dequantized
weight tiles. It reads the existing MLX affine Q4 buffers, with group size 64,
without repacking the checkpoint. Two SIMD groups each compute an eight-row,
32-column tile, staging 64 input channels at a time in double-buffered
threadgroup memory. The kernel fetches the next packed tile before submitting
the current matrix operation. This runs on the GPU's matrix hardware; it does
not use the separate Apple Neural Engine.

Compatible dense feed-forward layers compute gate, up, and SwiGLU in one
dispatch. Gate and up accumulators stay in registers, and the epilogue retains
the BF16 operation boundaries of the existing compiled activation. This avoids
writing and rereading separate gate/up arrays. The down projection is separate.

The MPP operand construction follows
[Splash 1.6.0's staged tile implementation](https://github.com/incoai/splash/blob/1.6.0/runtime/metal/kernels/common/gguf_staged_tile.h),
with attribution in `THIRD_PARTY_NOTICES.md`. Splash also calibrates dispatch
by shape and GPU size and implements fused GDN and wider verification attention;
see its [internals](https://github.com/incoai/splash/blob/1.6.0/docs/internals.md).
This experiment changes matrix projections and dense feed-forward execution.
GDN and attention kernels are unchanged.

The data feed also differs. Splash's staged decoder reads tile-major weight
and metadata planes and uses 32-channel steps. This prototype reads MLX's
original row-major arrays and uses 64-channel steps, consuming twice the
threadgroup staging memory for the same column tile. Splash's fused gate/up
kernel interleaves both projections at each K step; this prototype completes
gate before starting up in the same dispatch. These differences are visible in
the [staged tile helpers](https://github.com/incoai/splash/blob/1.6.0/runtime/metal/kernels/common/gguf_staged_tile.h)
and [linear kernels](https://github.com/incoai/splash/blob/1.6.0/runtime/metal/kernels/shared/gguf_linear.metal).
They are plausible tuning targets, rather than established explanations for
the measured end-to-end gap.

Dispatch requires macOS 26 or later, an Apple10 GPU, BF16 inputs and metadata,
and a supported affine Q4 geometry. It applies to the selected Qwen3.8 target
projections and the shared target vocabulary head used to score draft
proposals. Independent DFlash 2 layers retain the reference backend. Other
shapes, dtypes, and devices use existing dispatch.

FP16 weight tiles introduce a new numerical reference. Serial and verification
calls use the same eight-row tile and split count, padding shorter inputs, to
keep their reduction arithmetic consistent. This does not promise identical
continuations to the previous FP32 QMV implementation.

## Component results

These measurements use a 40-core M5 Max with 128 GiB of memory, GPU architecture
`applegpu_g17s`, synthetic packed weights, and input shape `[1, 8, K]`.
Each cell is the median of nine batches of eight independent calls, after a
one-second warm-up per geometry and two discarded rounds. Mode order rotates
each round. Times include synchronized submission and launch overhead.

The matrix column uses one K partition, matching the current experimental
dispatch. The machine-readable summary also retains two-, four-, and
eight-partition controls and every recorded sample.

| K → N | Reference | M5 matrix, one partition |
| --- | ---: | ---: |
| 5120 → 17408 | 0.223 ms | 0.199 ms |
| 17408 → 5120 | 0.633 ms | 0.504 ms |
| 5120 → 16384 | 0.213 ms | 0.206 ms |
| 5120 → 14336 | 0.219 ms | 0.186 ms |
| 6144 → 5120 | 0.106 ms | 0.108 ms |
| 5120 → 248320 | 6.563 ms | 2.110 ms |

The vocabulary projection was 3.11 times as fast in this component run. Its
recorded range was 2.056–2.156 ms, compared with 5.865–6.787 ms for reference.
The down projection and `5120 → 14336` samples drifted substantially during
the run; their median differences need repeat measurements before informing
dispatch. The `6144 → 5120` matrix path was slightly slower. Component gains
do not establish an end-to-end gain.

For gate/up/SwiGLU at `5120 → 17408`:

| Implementation | Median | Range |
| --- | ---: | ---: |
| Reference projections and compiled activation | 0.436 ms | 0.435–0.439 ms |
| Separate M5 projections and compiled activation | 0.475 ms | 0.465–0.480 ms |
| Fused M5 gate/up/SwiGLU | 0.392 ms | 0.388–0.398 ms |

Fusion reduced the median by 10.1% relative to reference. Separate M5 calls
were slower, which makes fusion an important part of this experiment.

The collector admitted the run after 30 seconds without competing compute
submissions or compilation, with normal memory pressure and desktop GPU use
up to 50%. Thermal state was fair and low-power mode was off. Another API
server remained resident, but its GPU counters were unchanged throughout the
run. The collector recorded no exclusion reasons. No unrelated process was
stopped.

## Correctness and full-model controls

All four selected GPU tests passed. The arithmetic test compares the matrix
kernel with a dense FP32 reference of FP16-rounded dequantized weights and
checks exact serial-versus-verification equality at widths one, four, and
eight. The fused activation test checks exact BF16 equality against separate
M5 projections and the existing compiled SwiGLU, including the real
`5120 → 17408` feed-forward geometry.

The installed-model test passed two prompts at a 128-token budget. It compares
actual token IDs across repeated DFlash 2 generation and a target-only oracle
using the same M5 arithmetic. Both prompts generated 128 tokens and matched
exactly. It also asserts that the shared target head uses the M5 backend and
that independent draft projections retain the reference backend. These are
correctness checks, not throughput measurements or broad quality evaluations.

The all-projection main comparison uses one release binary for reference,
fused, and target-only conditions. It pins the target and draft
assets from the [matched Splash comparison](qwen38-splash-matched-2026-11-08.md),
uses greedy decoding with reasoning off, and runs a 256-token warm-up followed
by three measured requests. Main controls generate exactly 512 tokens with
EOS stopping disabled; a separate matched stage uses a 1,024-token cap and EOS
stopping. Rates are the CLI's generated-token count divided by decode wall
time. Serial-oracle checks use the same arithmetic as each speculative
condition. Competing compute or abnormal memory pressure excludes timing.

The qualified main conditions produced these results:

| Workload / backend | Median TPS | Range | Matches its serial oracle |
| --- | ---: | ---: | --- |
| Code / reference | 80.4 | 74.3–89.7 | Yes |
| Code / M5 matrix and fused feed-forward | 65.8 | 64.4–69.4 | Yes |
| Prose / reference | 26.1 | 26.0–26.8 | Yes |
| Prose / M5 matrix and fused feed-forward | 22.9 | 22.7–23.2 | Yes |

Both code conditions used 75 verification rounds, with 437 of 519 draft tokens
accepted. Their continuations differ, but each matches the serial oracle for
its own arithmetic. The M5 condition was 18.2% slower by median. The M5
target-only code oracle also reached only 12.7 TPS in its single request.
The fixed eight-row matrix layout has a substantial cost for serial decoding.
Component gains have not translated into faster full-model decoding.

Prose was also slower, by 12.2% at the median. Further all-projection stages
were cancelled in favor of a head-only experiment, which preserves reference
dispatch in the target body. The latest completed cross-engine comparison
remains the earlier matched Splash run. These results do not establish that
mere.run has closed that gap.

## Head-only controls and phase profiles

The head-only control uses a rebuilt release binary with explicit projection
scope selection. Both its reference and experimental conditions use that same
binary and the same fixed 512-token protocol. Target body projections and
independent draft layers retain reference dispatch.

| Workload / backend | Median TPS | Range | Matches its serial oracle |
| --- | ---: | ---: | --- |
| Code / reference | 84.4 | 80.2–93.8 | Yes |
| Code / M5 vocabulary head | 84.2 | 79.6–92.2 | Yes |
| Prose / reference | 20.8 | 20.7–20.9 | Yes |
| Prose / M5 vocabulary head | 22.5 | 21.3–23.5 | Yes |

Within each workload, both conditions produced the same continuation and
acceptance counts. Code used 75 verification rounds and accepted 437 of 519
draft tokens; prose used 218 rounds and accepted 293 of 1,522. Code is tied.
The head-only prose median was 8.0% faster in this order. Absolute rates vary
substantially across collection windows, so this single comparison does not
establish a general gain. The head-only serial oracle reached 32.3 TPS in one
code request, which removes much of the serial cost of the broader matrix
backend.

Reversing code order did not confirm a benefit: head-only reached 65.1 TPS
versus 75.2 TPS for reference, a 13.4% regression by median. Both conditions
passed resource qualification and matched their serial oracle. The absolute
rates also differ substantially from the first order. This evidence does not
support enabling the head-only backend by default.

The EOS-stopping controls differ by less than 1%. Code reached 71.6 TPS
with the M5 head versus 71.2 TPS for reference, generating 970 tokens in every
request. Prose reached 24.7 versus 24.5 TPS, generating 601 tokens in every
request. Each condition has three qualified requests. Together with the
reversed code result, this does not establish a general throughput gain.

A further real-checkpoint test confirms actual head dispatch. On eight real
hidden states, the selected head's maximum difference from the matrix primitive
was zero. A reference projection over the same arrays differed by 0.0625.
Both installed prompts passed this assertion and repeated-generation and
serial-oracle token checks. This rules out fallback arithmetic as the
explanation for the tied code result.

Separate synchronized profiles use one 256-token warm-up and one measured
512-token code request per condition. All three passed resource qualification
and their serial-oracle checks. The following wall times sum 75 speculative
rounds:

| Backend | Draft graph | Draft wait | Verification graph | Verification wait | Cache completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 0.043 s | 0.730 s | 0.186 s | 5.634 s | 0.153 s |
| M5 head only | 0.053 s | 0.700 s | 0.222 s | 5.273 s | 0.167 s |
| M5 all supported projections | 0.048 s | 0.634 s | 0.242 s | 6.457 s | 0.156 s |

Profiling evaluates and synchronizes the draft before verification and
synchronizes phase boundaries. These runs change scheduling and are excluded
from normal throughput results. The head-only profile reduced verification
wait by 6.4% in this single control; broader matrix dispatch increased it.
This supports keeping the next experiment narrow, but does not establish an
uninstrumented throughput gain. The phase timings include host scheduling and
GPU waits; they are not isolated GPU kernel times.

## Tiled revision

A further opt-in revision implements tile-major MLX Q4 codes and metadata,
32-channel stages, and interleaved gate/up accumulations. Projection kernels
use 8 KiB of threadgroup staging instead of 16 KiB; paired feed-forward kernels
use one SIMD group and 4 KiB. Cached packed buffers accompany the original
parameters, which remain available to prefill and fallback. Packing has a
one-time preparation cost and increases resident memory.

The tiled revision passed offline Metal compilation, strict SwiftLint, focused
CPU checks, and two GPU numerical tests. The numerical tests cover K values
512, 2048, 5120, and 17408, exact serial-versus-verification agreement, and
interleaved feed-forward equality with separate tiled projections at the real
`5120 → 17408` geometry. Both installed prompts also passed repeated-generation,
serial-oracle token checks, and the actual matrix-dispatch assertion.

The first tiled release used one K partition throughout. Its main code control
reached 60.8 TPS versus 77.3 TPS for reference; both used 75 rounds and accepted
437 of 519 draft tokens. Reversed code reached 73.9 versus 85.5 TPS. Prose
reached 20.3 versus 16.8 TPS, but used different continuations and acceptance
counts. All these conditions passed resource qualification and their serial
oracle checks. The mixed results do not establish a general TPS gain.

A separate partition control now uses two K partitions for `17408 → 5120` and
`6144 → 5120`, retaining one elsewhere. Without partitioning, these projections
launch 80 threadgroups on the 40-core GPU. The new schedule is fixed by shape
for both serial and verification calls. It passed numerical tests at one,
two, and four partitions, both installed prompts, and actual head dispatch.
The installed probe recorded about 29.4 GiB of active MLX allocations with
tiled mode enabled; this includes model, draft, and probe state and is not a
peak-memory measurement. Process RSS alone does not account for all of those
allocations.

The partition control's release build and throughput comparison passed. Its
qualified results are reported first in this document. Contended attempts
were retained separately and excluded from those rates.
The new tiled arithmetic can produce different continuations. Enable it with
`MERERUN_Q38_M5_TILED_WEIGHTS=1` alongside the M5 backend. Default dispatch
remains unchanged.

## Validation and reproduction

The native SwiftPM release builds, focused policy tests, GPU numerical tests,
and installed-model token checks passed. The final direct run of
`./scripts/check.sh`, using the native SwiftPM wrapper, passed: strict SwiftLint
reported no violations in 2,743 files; build, CLI help, and hygiene checks
passed; XCTest ran 5,841 cases with 491 skips and zero failures; all 197 Swift
Testing checks passed.

An earlier gate reported 52 assertions involving protected run-record files.
The collector then waited on the console lock flag. That flag still reported
locked during the successful direct rerun, so it was not a reliable admission
check. The protected-file suites passed without changing file protection or
skipping those tests.

Enable the experiment with `MERERUN_Q38_M5_VERIFICATION=1`. Fused feed-forward
execution is enabled with that backend; set `MERERUN_Q38_M5_FUSED_FFN=0` for
the separate-projection control. Set `MERERUN_Q38_M5_PROJECTIONS=head` to use
M5 dispatch only for the vocabulary head. These settings are documented in
[configuration](../configuration.md#mererun-q38-m5-verification).

The measured winning configuration sets `MERERUN_Q38_M5_VERIFICATION=1`,
`MERERUN_Q38_M5_TILED_WEIGHTS=1`, `MERERUN_Q38_M5_PROJECTIONS=all`, and
`MERERUN_Q38_M5_FUSED_FFN=1`, with `MLX_ENABLE_TF32=0` and the pinned draft
configured through `MERERUN_Q35_DFLASH2_PATH`. The reference control sets
`MERERUN_Q38_M5_VERIFICATION=0`. Exact workload text is in the `code` and
`prose` entries of `scripts/benchmark-q35-generation.py`.

Compile before admitting an idle GPU window, then run the numerical tests:

```sh
MERERUN_TEST_MLX_DEVICE=gpu MLX_ENABLE_TF32=0 \
  swift test --build-system native --skip-build \
  --filter M5TiledAffineQ4Tests
```

For the original row-major component timing methods, select
`M5AffineQ4VerificationTests` and set
`MERERUN_TEST_M5_Q4_COMPONENT_BENCHMARK=1`. The tests themselves do not reserve the GPU or monitor other
processes. Retain resource receipts and exclude contended timing runs.

The [machine-readable summary](qwen38-m5-verification-2026-11-08-summary.json)
contains raw component samples, medians and ranges, source hashes, binary
hashes, qualification details, and the completed full-model comparisons.
