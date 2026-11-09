# Qwen3.8 exact Q4 verification: eight-row experiments

The first component experiment favors sharing weights across eight inputs for
the large vocabulary projection, but not across every projection. Production
dispatch remains unchanged. There is no qualified full-model speedup yet.

`SmallBatchAffineQMV` processes an eight-row verification block as two
groups of four inputs. The experiment keeps the same FP32 arithmetic and serial
QMV reduction order and reuses each weight read across all eight inputs. It
tests four, two, and one output rows per SIMD group to vary register use. It
reads the existing packed arrays without making additional weight copies.

## Component results

These are synchronized wall times per projection call on a 40-core M5 Max with
128 GiB of memory. Inputs have shape `[1, 8, K]`. Packed weights, scales, and
biases are synthetic and use affine Q4, group size 64, and BF16 activations.
Each cell is the median of nine batches of eight calls, after two unrecorded
rounds. Layout order rotates each round. Submission and launch overhead are
included; these are not isolated GPU kernel timings.

| K → N | Current, four inputs | Eight inputs, four outputs | Eight inputs, two outputs | Eight inputs, one output | MLX native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5120 → 17408 | 0.287 ms | 0.232 ms | 0.310 ms | 0.399 ms | 0.292 ms |
| 17408 → 5120 | 0.221 ms | 0.232 ms | 0.264 ms | 0.375 ms | 0.222 ms |
| 5120 → 16384 | 0.211 ms | 0.218 ms | 0.234 ms | 0.322 ms | 0.205 ms |
| 5120 → 14336 | 0.190 ms | 0.196 ms | 0.210 ms | 0.287 ms | 0.182 ms |
| 6144 → 5120 | 0.104 ms | 0.108 ms | 0.114 ms | 0.148 ms | 0.096 ms |
| 5120 → 248320 | 6.605 ms | 5.648 ms | 10.155 ms | 15.513 ms | 7.068 ms |

The eight-input, four-output layout reduced the vocabulary projection's median
by 14.5%. Its observed range was 4.903–5.959 ms, compared with
5.857–6.865 ms for the current layout. Most other shapes regressed. The first
geometry also had substantial variation: the current layout ranged from
0.216 to 0.293 ms. Its median difference needs a longer warm-up and repeat
before informing dispatch. Smaller output tiles were slower in this run.

The collector admitted the component test after 30 seconds without compilation
or competing compute, using per-process GPU counters and allowing desktop GPU
use up to 50%. It recorded normal memory pressure, fair thermal
state, and no competing compute submissions during the test. Earlier attempts
at full-model timing were excluded when other inference or compilation resumed.
No unrelated process was stopped.

## Partial full-model arithmetic controls

These separate runs compare the existing exact Q4 dispatch with MLX's native
wide arithmetic. They do not test the new head layout. The toggle affects
eligible draft and target projections, so this is not a target-only experiment.
Both conditions use the pinned target and draft from the matched Splash run,
reasoning off, greedy decoding, an 8,192-token context limit, and a fixed
512-token output budget with EOS stopping disabled. Each condition has one
256-token warm-up and three measured requests. Rates exclude the first emission.

| Workload / arithmetic | Median TPS (range) | Matches serial output |
| --- | ---: | --- |
| Code / exact | 75.1 (73.3–76.7) | Yes |
| Code / MLX native | 71.7 (67.4–79.8) | Yes |
| Prose / MLX native | 24.3 (24.3–25.1) | No |

Each row passed the resource qualification checks. The code ranges overlap;
there is no established speed benefit from switching to native arithmetic.
The native prose continuation differs from serial decoding. An exact prose
attempt matched the serial output but was excluded from timing because another
compute process submitted GPU work. Its replacement and reversed-order code
controls remain queued. The head-layout A/B follows those controls and a release
rebuild. These partial results do not justify changing default dispatch.

## Correctness and validation

The GPU parity test passed all 16 comparisons: the current layout and three
experimental layouts at four geometries, including Qwen's gate/up and down
projections. Maximum FP32 difference against concatenated serial QMV rows was
zero. This checks synthetic projection arithmetic; it does not establish
real-checkpoint generation parity. The MLX-native component is a timing
reference and has no serial-equivalence claim.

Lint and the native SwiftPM build passed. The full gate ran 5,831 XCTest cases
(485 skipped) and 197 Swift Testing checks. The Swift Testing checks passed;
XCTest reported 52 failures involving protected run-record files. An isolated
rerun reproduced the file permission failures, and the host reported that its
screen was locked. Run-record writes use `completeUnlessOpen` protection, so a
clean gate requires an unlocked session. File protection was unchanged.
The CLI help sweep, repository hygiene scans, and `git diff --check` passed
separately. The remaining arithmetic controls and head-layout A/B wait for idle
hardware. The new head layout has no qualified full-model
throughput result, and default dispatch is unchanged.

The [machine-readable summary](qwen38-exact-q4-verification-2026-11-08-summary.json)
retains all component medians and ranges, resource qualification, source hashes,
the test binary hash, and validation limits. The
[matched Splash comparison](qwen38-splash-matched-2026-11-08.md) remains the
latest completed cross-engine throughput comparison.

## Reproduce

Build with the checkout's native SwiftPM workaround, then run the parity test:

```sh
MERERUN_TEST_MLX_DEVICE=gpu MLX_ENABLE_TF32=0 \
  swift test --build-system native \
  --filter SmallBatchAffineQMVTests.testSharedEightRowsPreserveSerialArithmetic
```

For component timing, first admit an idle compute window and run separately:

```sh
MERERUN_TEST_MLX_DEVICE=gpu MLX_ENABLE_TF32=0 \
  MERERUN_TEST_Q4_COMPONENT_BENCHMARK=1 \
  swift test --build-system native --skip-build \
  --filter SmallBatchAffineQMVTests.testEightRowComponentBenchmark
```

The test itself does not reserve the GPU or monitor other processes. Compile
before the timing window, retain the resource receipts, and exclude any run
with concurrent inference or compilation.

For an experimental whole-model run, set
`MERERUN_Q38_EXACT_Q4_HEAD_OUTPUT_ROWS=4` alongside the DFlash 2 configuration
from the matched comparison. This selects eight-input sharing for the
`5120 → 248320` vocabulary head only. Values `2` and `1` select the slower
output-tile controls; `0` restores the default. Keep
`MERERUN_Q38_EXACT_WIDE_Q4` enabled: disabling exact Q4 dispatch also disables
this experiment. Whole-model output and throughput qualification are pending.
