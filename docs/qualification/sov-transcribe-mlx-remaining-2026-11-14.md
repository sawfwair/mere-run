# SOV MLX cache, draft, and prefill experiments — 2026-11-14

The remaining optimization experiments compare against the already-pipelined
SOV+DFlash implementation. Transactional KV storage reduces long-clip resident
latency by about 3% and passes the longer window-wrap check. Confidence truncation
helps on the original long fixture but its refinement fails token parity on the
93-second recording and is slower there.
Shorter fixed blocks and the calibrated draft vocabulary add verification rounds
and lose performance. Compiling the MoE segment after attention is neutral on the
long recording. These policies remain benchmark options; ordinary transcription
keeps the existing pipeline.

The [JSON receipt](./sov-transcribe-mlx-remaining-2026-11-14.json) contains the raw
CLI reports, adjacent-pair ratios, source and binary hashes, calibration IDs,
route captures, matrix-kernel screen, trace metrics, and cleanup manifests.

## Repeated Q8 comparison

Positive reductions mean lower latency. Each row combines two fresh processes,
two warmups per mode, and six plus eight measured pairs. The second process
reverses policy, recording, and starting-mode order. Measurements alternate
between the pipeline baseline and one candidate on the same resident model.
Resident timing includes synchronization and cleanup, and excludes model loading
and audio I/O. Generation is a substage of the same measured call.

| Policy | Audio | Resident reduction | Generation reduction | Wins/pairs |
| --- | --- | ---: | ---: | ---: |
| Rollback cache | 46.57 s | 3.02% | 3.97% | 13/14 |
| Rollback cache | 3.52 s | -0.89% | -0.34% | 5/14 |
| Compiled MoE segment | 46.57 s | -0.30% | -1.50% | 6/14 |
| Compiled MoE segment | 3.52 s | 1.72% | 0.11% | 9/14 |
| Block length 4 | 46.57 s | -43.72% | -63.45% | 0/14 |
| Block length 4 | 3.52 s | -27.94% | -147.18% | 1/14 |
| Block length 8 | 46.57 s | -4.51% | -6.28% | 3/14 |
| Block length 8 | 3.52 s | -13.45% | -76.86% | 1/14 |
| Block length 12 | 46.57 s | -2.70% | -4.69% | 4/14 |
| Block length 12 | 3.52 s | -13.86% | -88.88% | 1/14 |
| Confidence truncation | 46.57 s | 2.92% | 4.30% | 12/14 |
| Confidence truncation | 3.52 s | -1.82% | -11.75% | 4/14 |
| Calibrated 32k draft head | 46.57 s | -51.17% | -74.18% | 0/14 |
| Calibrated 32k draft head | 3.52 s | -16.01% | -99.13% | 1/14 |

All 420 recorded outputs in this screen match the existing frozen token IDs.
Unrecorded warmups are excluded from that count. Absolute timings drifted across
processes, so the table uses medians of adjacent-pair ratios. Background machine
activity remained; no compiler or profiler from this task ran during these
qualified measurements.

<!-- final-confirmation -->

## Final confirmation

The final binary adds confidence truncation after the first round and records
actual proposal lengths. Q8 refinement uses eight pairs per recording; BF16 uses
six pairs. The 93.14-second recording repeats the long fixture
twice and uses four pairs per policy, with a 1,529-token prompt that exceeds
the sliding window. Its baseline is newly measured, rather than a preexisting
frozen transcript. Baseline tokens are also checked across processes.

| Configuration | Policy | Audio | Resident reduction | Generation reduction | Wins/pairs | Token match |
| --- | --- | --- | ---: | ---: | ---: | --- |
| BF16 | `adaptive-after-first` | 46.57 s | 1.30% | 0.41% | 4/6 | Yes |
| BF16 | `adaptive-after-first` | 3.52 s | 3.62% | 2.62% | 4/6 | Yes |
| BF16 | `compiled` | 46.57 s | -1.94% | -3.79% | 2/6 | Yes |
| BF16 | `compiled` | 3.52 s | -1.22% | -2.81% | 3/6 | Yes |
| BF16 | `rollback` | 46.57 s | 3.14% | 3.31% | 6/6 | Yes |
| BF16 | `rollback` | 3.52 s | 12.31% | 19.22% | 4/6 | Yes |
| Q8 Metal/attention | `adaptive-after-first` | 46.57 s | 6.03% | 10.04% | 8/8 | Yes |
| Q8 Metal/attention | `adaptive-after-first` | 3.52 s | -0.77% | -1.49% | 3/8 | Yes |
| Q8 Metal/attention | `adaptive-after-first` | 93.14 s | -14.63% | -20.85% | 1/4 | No |
| Q8 Metal/attention | `rollback` | 93.14 s | 2.11% | 3.54% | 4/4 | Yes |

All 532 recorded outputs on the original two fixtures match
their frozen tokens. On the longer recording, all rollback outputs match the
pipeline baseline; all four measured adaptive candidates differ. Overall,
546/550 recorded outputs match their comparison tokens.
The five recorded trace outputs are counted separately.

The adaptive failure is deterministic across its measured repetitions: the
296-token transcript substitutes “Calico” for “Callico,” first differing at
zero-based token 60. Both modes finish below the 448-token output limit.
Changing verification lengths changes attention and expert dispatch and
floating-point grouping. That is a plausible explanation, rather than a
proven attribution from this trace. Target verification alone does not prove
bit-identical tokens across different block shapes. The adaptive refinement
fails qualification and remains available only as a benchmark experiment.
The original BF16 and Q8 rollback checks, including the wrapped window, pass.
The shorter-clip gains do not override this parity failure. Short-recording
timing varies substantially between configurations and does not establish a
consistent gain. All new policies remain opt-in.

<!-- final-confirmation-end -->

## What changed

`rollback` uses an SOV-only linear KV buffer. It reserves space before a trial,
appends speculative rows, and commits by changing the valid length. Sliding
storage retains the current window plus spare capacity and compacts before a
trial when needed. Full-attention storage grows in 256-row increments. This
avoids trial forks and copying accepted prefixes back to the base cache.
Synthetic tests cover partial and zero commits, windows smaller than proposals,
compaction, multiple heads, unequal K/V widths, and isolated snapshots.

Pinned MLX attention accepts these batch-one K/V views when the last dimension
is contiguous. Its vector path also checks batch/head layout; the full path
checks only the last stride. A sliced sequence dimension alone does not force a
contiguous copy in this case. This source check does not measure other copies
or buffer donation in the complete inference. The chronological rollback layout
can differ from the reference ring's physical order for a single query after
wrap, so this cache remains a DFlash research policy.

`compiled` passes module parameters as observed runtime inputs to MLX compilation
and passes the layer scalar explicitly. Each compiled graph covers the complete
MoE segment after attention: normalization, residuals, dense MLP, router, sparse
experts, and output scaling. Module and custom-kernel replacement invalidate its
cached graph. The native checkpoint's first verification logits match reference
exactly. This experiment does not compile attention, mutable caches, or the whole
decoder into one graph.

`adaptive` computes the maximum draft softmax probability in FP32, then truncates
at the first probability below 0.5, keeping at least one proposal. Reading the
cut point introduces a host dependency before verification. The
`adaptive-after-first` refinement keeps the first full block and applies this
policy only on later rounds. JSON `proposedLengths` and `acceptedLengths` expose
the actual proposal and agreeing-prefix counts for every verification round.

Fixed block lengths include the anchor. For the long recording, block 8 needs
21 rounds and block 12 needs 17, versus the baseline's 16. The short recording
needs two rounds at block 8 or 12 versus one at length 16. Smaller expert unions
do not offset these extra rounds. All-layer route capture recomputes each length
rather than assuming a prefix has identical floating-point dispatch:

| Verification length | Mean distinct experts per layer | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| 4 | 19.93 | 16 | 23 |
| 8 | 27.83 | 22 | 34 |
| 12 | 33.07 | 27 | 41 |
| 16 | 37.50 | 30 | 51 |

## Calibrated vocabulary

The draft-only 32k head selects rows from 7,000
[FLEURS training transcripts](https://huggingface.co/datasets/google/fleurs):
1,000 each from `en_us`, `fr_fr`, `de_de`, `es_419`, `it_it`, `pt_br`, and `nl_nl`.
FLEURS is licensed CC BY 4.0. Calibration uses training text and excludes the
benchmark recordings. The tokenizer observes 28,354 distinct token IDs; EOS IDs
are forced in and ascending unused IDs fill the remaining capacity. The compact
head maps local predictions back to global IDs; target verification retains its
full vocabulary and original parameters.

After selection, the frozen recordings have vocabulary coverage of 114/138 and
12/13 tokens. This is a coverage diagnostic, not an accuracy score. Long-clip
verification grows from 16 to 29 rounds, and mean matching proposal length drops
from 7.94 to 3.97. The smaller head loses more through acceptance than it saves
in projection. The script `scripts/model-conversion/calibrate_sov_vocabulary.py`
reproduces selection; the receipt preserves the selected IDs and corpus hashes
without publishing the training transcripts.

## M5 prefill matrix kernel

The research shader `scripts/model-conversion/SOVPrefillQ8.metal` tests Metal
TensorOps tiles 8×32, 16×32, 16×64, and 32×64 with a K tile of 64. Each has a
BF16 dequantization variant and a group-affine variant. Native packed layer-zero
weights and routed activations are captured before screening in Python MLX
0.32.4. The Python MLX baseline is bit identical to the captured native output
for both prefill and verification.

For 779 prefill tokens, native-equivalent MLX takes 3.515 ms and the fastest
custom tile takes 19.414 ms, 5.52× slower despite bit-identical output.
The best verification tile takes 1.690 ms versus 0.939 ms, with numerical
drift. Group-affine alternatives also fail to improve the result. Route sorting,
gathering, and tile-job preparation are excluded from custom timings, making
these optimistic screens. The shader remains a reproducible research artifact
and is not dispatched by the runtime. These layer-zero screens do not establish
whole-model prefill performance.

## Metal System Trace

A successful Metal System Trace records the pipeline baseline through model
preparation, two warmups, and four measured inferences. All five recorded outputs
match frozen tokens. Filtering GPU intervals to the launched process gives a
10.199-second active span, 7.201 seconds of overlapping GPU work, and 2.998
seconds with no GPU work from that process. There are 599 gaps above 100 µs and
299 above 1 ms.

These intervals include preparation and whole inferences. They are not isolated
generation gaps, and other processes can use the GPU while the target is idle.
The exported template provides encoder-level labels, without shader names, so
it cannot attribute a precise cost to KV copy kernels or identify every host
dependency. The trace supports further host-scheduling investigation but does
not establish a measured per-round copy budget. The earlier failed trace is
excluded: its inference was rejected by the disk-space reserve.

## Build storage

The worktree's `.build` directory fell from about **106 GiB to 80 GiB** after
removing obsolete Core ML source packages and superseded compiled exports,
deleting an extracted ASR download cache, and losslessly archiving old traces.
The manifest records deleted paths and reasons. SOV checkpoints, native ASR
models, active compiled encoders, raw benchmark receipts, and source changes
were preserved.
About 50 GiB is the original SOV checkpoint and 12 GiB is Swift build output.
Directory sizes do not imply the same change in physical APFS free space.

## Validation and reproduction

- Apple M5 Max, 128 GiB; macOS 27.2 (26B123), Xcode 27.2 (27B52).
- Target revision `505d8fae243924c0d8acdc622f7f0abab344eba3`; draft revision
  `8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`; MLX Swift revision
  `1e077ba968b291e0237d1ab522b271d433fa1713`.
- MLX encoder/projector, native DFlash, 1,024-token prefill chunks, output limit
  448. Dynamic-int8 encoder and fused head normalization are disabled.
- Q8 measurements use group-64 Metal experts and experimental shared-mask,
  direct-RoPE attention. BF16 checks use original experts and reference attention.
- Thirty-six focused GPU tests passed; 16 CPU tests passed and one GPU-only
  test was skipped. Independent FP32 fixture checks disable TF32; release
  checkpoint comparisons use default precision.
- A strengthened GPU and CPU retest replaces MLP weights and the layer scalar
  after compilation, confirms changed output, and matches the interpreted model.
- Release build and CLI documentation inventory passed. The full repository
  gate remains deferred at the user's request.
- Strict SwiftLint, whitespace checks, Python syntax checks, and both research
  script help commands passed. Style-lint findings were reviewed: numeric
  comparisons, the Metal System Trace product name, and concurrent activity.

```bash
swift build -c release --product mere.run -Xswiftc -disable-cmo
.build/release/mere.run model benchmark sov \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --model-root .build/qualification/sov/target \
  --draft-model .build/qualification/sov/draft \
  --experimental-decoder q8-metal --experimental-attention \
  --generation-experiment rollback --generation-baseline pipeline \
  --interleave-generation --warmups 2 --repetitions 8 --json
```

Substitute another research policy from the table and add `--generation-first`
to reverse starting order. For `vocabulary-32k`, pass the calibrated ID array with
`--draft-vocabulary path.json`. Omit the experimental expert and attention flags
for original BF16. The optional native generation profile captures routing and
packed weights when `MERERUN_PROFILE_SOV_REMAINING` is set; the matrix screen
accepts that capture prefix through `--capture`.

To reproduce the window-wrap comparison, repeat the long recording twice:

```bash
ffmpeg -stream_loop 1 \
  -i Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  -c:a pcm_s16le /tmp/sov-wrapped.wav
```

Use `/tmp/sov-wrapped.wav` as the benchmark input with the same flags. Its
SHA-256 is `cf5ae3118a31db300a4e8243f2d33ddb8896cbfe6b92f89ae19f71bc2ac0a565`.
`adaptive-after-first` reproduces the failed comparison; `rollback` passes it.

These fixtures and one machine do not qualify multilingual recognition quality
or performance across Apple Silicon devices.
