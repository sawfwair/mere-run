# SOV custom Metal expert qualification — 2026-11-13

The custom Q8 Metal expert kernel reduced the captured verification expert block
from **0.903 ms to 0.742 ms (17.8%)**. Paired resident inference with DFlash showed
a median latency reduction of **4.4% on 46.57 seconds of speech** and **1.6% on
3.52 seconds** relative to the existing MLX Q8 expert path. The kernel preserves
the packed parameters and matched MLX output exactly in the tested cases.

This is the experimental `--experimental-decoder q8-metal` benchmark policy.
Ordinary transcription retains the existing default. The results qualify two
English recordings on one M5 Max; they do not establish general recognition
accuracy or performance across Apple Silicon devices.

The [machine-readable receipt](./sov-transcribe-metal-moe-2026-11-13.json) includes
all measured sequences, first inference results, paired ratios, operator samples,
diagnostic profiles, rejected experiments, and source and binary hashes.

## Configuration

- Apple M5 Max, 128 GiB unified memory; macOS 27.2 (26B123), Xcode 27.2 (27B52).
- SOV-Transcribe-1 revision `505d8fae243924c0d8acdc622f7f0abab344eba3`.
- SOV-Transcribe-1-DFlash revision `8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`.
- MLX Swift revision `1e077ba968b291e0237d1ab522b271d433fa1713`;
  Python operator screen used MLX 0.32.4.
- GPU encoder/projector, DFlash enabled, prefill chunk size 1024, maximum 448 tokens.
  Dynamic-int8 encoder disabled. Combining that encoder with Q8 experts remains
  unqualified after the earlier long-clip token mismatch.
- Affine Q8 experts, group size 64; BF16 inputs, scales, biases and outputs, with
  FP32 accumulation. Prepared MLX active allocation was **29.11 GiB** for both
  expert kernels. This is not total process or system memory.
- Release build: `swift build -c release --product mere.run -Xswiftc -disable-cmo`.
  Each study compared one binary internally. The standalone study and later
  paired study used different binaries; both hashes are retained in the receipt.

The recordings were `Tests/MereRunCoreTests/Fixtures/D1/released/speech.wav`
(3.5183125 seconds, 13 frozen tokens) and
`Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav`
(46.57 seconds, 138 frozen tokens). Audio SHA-256 values are recorded in the JSON.

## Kernel and dispatch

The decoder has 128 experts and selects eight routes per token. The shipped
kernel shares packed weight loads across two adjacent routes belonging to the
same expert. It handles expert boundaries, odd route counts and the 704-wide
down projection tail. Parameters are passed on every call; the cached shader
does not capture weights.

Inspection of the actual MLX dispatcher corrected an initial hypothesis:
**stock Q8 verification already uses GEMV**. The grouped matrix path requires at
least four routes per expert on average; the 16-token verification block has
128 routes across 128 experts. Matrix tile waste was therefore not the cause of
this Q8 bottleneck. Sharing loads across nearby GEMV routes was the useful change.

The custom path applies only to sorted, small BF16 affine-Q8 group-64 batches
with at most 128 routes and fewer than four routes per expert on average.
Unsupported shapes and larger prefill retain MLX dispatch. This also preserves
the matrix path's arithmetic: its BF16 dequantization differs from the FP32
affine sums used by the vector path. Existing route sorting, activation,
scatter and mixing remain in MLX.

The shared production shader was screened against real native Q8 activations.
Thirty timed samples per variant alternated execution order after warmup:

| Verification implementation | Median expert block | Exact output |
| --- | ---: | --- |
| MLX Q8 | 0.903 ms | reference |
| Custom, one route per load | 0.773 ms | yes |
| **Custom, two routes per load** | **0.742 ms** | **yes** |
| Custom, four routes per load | 0.899 ms | yes |

The captured input was `[1, 16, 2816]`, selecting 39 distinct experts. These
times exclude preparing sorted input routes but include both projections,
activation, inverse scatter and mixing. They are not full transcription times.
The unchanged `[1, 779, 2816]` prefill fallback measured 3.532 ms in this screen.

Small BF16 TensorOps matrix tiles were bit exact but slower than MLX (best tested
tile 1.386 ms versus 1.295 ms). Q8 matrix tiles were slower (best 1.687 ms versus
0.932 ms) and changed output (relative RMSE about 0.00482). Routing fusion was
also tested; its extra benefit beyond the simpler projection kernel was small
and inconsistent. Those prototypes were not integrated. Their independent
screens used different captures and cannot be compared directly with the table.

## Resident inference with DFlash

The primary comparison loads one Q8 model per process and toggles only expert
dispatch. Each cycle measures both modes, reversing their order on alternating
cycles. A second session starts with Metal and reverses recording order. There
are two warmups and ten measurements per mode in each process: **80 measured
transcriptions and 40 adjacent pairs** across four processes.

Timing starts after the dispatch switch and includes synchronized inference and
request cleanup. It excludes checkpoint loading and audio-file reading. The
compiled shader is cached, parameters are not repacked, and each inference
creates fresh decode caches. No benchmark, profiler or compiler job from this
task ran alongside these measurements.

| Recording | Session | MLX Q8 median | Custom median | Median paired reduction | Custom wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.52 s | 1 | 256.42 ms | 252.66 ms | 1.63% | 9/10 |
| 3.52 s | 2 | 358.11 ms | 357.32 ms | 1.62% | 7/10 |
| 46.57 s | 1 | 1372.65 ms | 1310.30 ms | 4.66% | 10/10 |
| 46.57 s | 2 | 1515.30 ms | 1389.65 ms | 4.38% | 10/10 |

The headline uses the median of adjacent per-pair latency reductions: **1.63%**
and **4.45%**, respectively. It is not the ratio of pooled medians. Session
drift is substantial, especially on the short recording, so adjacent pairs are
the more useful estimate. Long-clip generation medians fell from 943 to 887 ms
in the first session and from 1029 to 932 ms in the second.

An earlier standalone comparison loaded each variant in a separate process,
with two warmups and five measurements per recording. The second session
reversed both variant and recording order. Its 60 measured sequences gave:

| Decoder experts | 3.52 s pooled resident median | 46.57 s pooled resident median |
| --- | ---: | ---: |
| BF16 MLX | 310.04 ms | 1498.80 ms |
| Q8 MLX | 270.06 ms | 1322.37 ms |
| Q8 custom Metal | 264.45 ms | 1317.29 ms |

That standalone study did **not** establish a robust total-latency gain from the
custom kernel. Long-clip generation improved, while unchanged encoder and
prefill timings drifted upward. It motivated the paired comparison; all results
are retained rather than discarded. BF16-to-Q8 packing benefits were already
established separately in the [decoder qualification](./sov-transcribe-decoder-2026-11-13.md).

## What still dominates

Real-checkpoint diagnostic profiles instrumented the target verifier, draft,
heads and individual decoder layers. In the stock Q8 profile, the target body
accounted for about 80% of the staged generation work: median 60.31 ms versus
4.16 ms for the draft body, 1.26 ms for draft context append, 4.85 ms for the
draft head/argmax and 3.21 ms for the target head/argmax.

With the custom kernel, summed per-layer synchronized categories were:

| Category | Diagnostic stage sum |
| --- | ---: |
| Attention and input normalization | 35.12 ms |
| Experts and their normalization | 30.01 ms |
| Dense MLP and normalization | 12.00 ms |
| Router | 11.55 ms |
| Final normalization and residual | 5.82 ms |
| Attention residual | 5.33 ms |

These are **debug, forced-barrier diagnostics**, not release latency. Barriers
alter overlap and add overhead; these sums cannot be added to the unbarriered
target body time. Profiles ran sequentially, so changes in untouched stages
cannot be credited to the kernel. The supported finding is that attention and
expert computation still dominate the verifier, while draft and head work is
smaller. The paired release measurements above establish the performance claim.

## Parity and validation

- All **140 measured sequences and 16 first inference sequences** matched the
  frozen BF16+DFlash token IDs: 13 tokens on the short recording and 138 on the
  long recording. Warmups are excluded from this count.
- Native Q8 versus custom Q8 candidate IDs and the complete captured target
  verifier hidden state `[1, 16, 2816]` were bit identical, with zero maximum error.
- The operator screen produced zero relative RMSE for all three vector variants.
- **35 focused tests passed**, covering kernel tails, expert boundaries, odd
  route counts, single rows, updated parameters, prefill and dense-route
  fallback, packing identity across dispatch toggles, SOV runtime and weights,
  CLI parsing, paired-mode validation and generated command documentation.
- Two opt-in real-checkpoint diagnostic profile tests passed. The release build,
  scoped strict SwiftLint and `git diff --check` passed. The full repository
  gate was deferred at the user's request.

Exact output on these fixtures is stronger than transcript-only comparison but
does not replace multilingual recognition evaluation. The Q8 policy and custom
kernel remain experimental.

## Reproduce

Run the custom policy directly:

```bash
.build/release/mere.run model benchmark sov \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --model-root .build/qualification/sov/target \
  --draft-model .build/qualification/sov/draft \
  --experimental-decoder q8-metal --warmups 2 --repetitions 10 --json
```

Compare dispatch on the same resident Q8 weights:

```bash
.build/release/mere.run model benchmark sov \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --model-root .build/qualification/sov/target \
  --draft-model .build/qualification/sov/draft \
  --experimental-decoder q8 --interleave-metal \
  --warmups 2 --repetitions 10 --json
```

Repeat with `--metal-first` to reverse the starting order. Warmups and repetitions
apply per mode. JSON aligns `measurementKernels`, `residentSeconds` and
`measurements` by index. The existing all-ASR harness also accepts
`--experimental-decoder q8-metal` for the SOV backend.

The shared-shader screen is `scripts/model-conversion/profile_sov_moe_metal.py`.
It takes `--checkpoint`, `--activations`, `--output` and `--repetitions`.
The opt-in Swift profile test supplies native activation captures; the checked-in
receipt retains the resulting timings and exact shader hash.
