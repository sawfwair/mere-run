# SOV shader profiling and byte unpacking — 2026-11-14

Byte-vector unpacking improves resident latency by **0.64% on the 46.57-second
recording** and **0.72% on the 93.14-second recording**. Generation improves by
about 1.0–1.2%. The short recording has no reliable gain. All 112 measured
comparison outputs match frozen token IDs and DFlash acceptance counts.
The change is retained in the existing opt-in Q8 Metal path. Ordinary BF16
transcription keeps its current policy.

The [JSON receipt](./sov-transcribe-shader-profile-2026-11-14.json) contains raw paired reports, token IDs,
per-round host timings, trace metrics, shader counters, source and binary
hashes, microbenchmarks, and storage relocation totals.

## Resident comparison

Both modes use one resident target and draft, the same saved Q8 group-64 weights,
optimized attention, rollback KV storage, and native full audio padding. The
control extracts bytes with integer shifts; the candidate reinterprets each
word as `uchar4` before FP32 conversion. Kernel selection happens outside timing
and does not replace parameters. Two warmups per mode precede measurement.
Two fresh processes per clip reverse clip and starting-mode order. Adjacent
inferences alternate modes, with order reversed on alternate cycles.

| Audio | Shift-unpack median | Byte-unpack median | Paired resident reduction (95% CI) | Paired generation reduction | Resident wins/pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.52 s | 0.2350 s | 0.2358 s | 0.08% (-0.59 to 0.33%) | 0.73% | 9/16 |
| 46.57 s | 1.2043 s | 1.1977 s | 0.64% (0.41 to 1.03%) | 1.05% | 19/24 |
| 93.14 s | 2.2678 s | 2.2450 s | 0.72% (0.61 to 0.97%) | 1.19% | 13/16 |

Percentages are medians of adjacent-pair ratios, so they need not equal the
ratio of the displayed latency medians. Confidence intervals bootstrap pairs
within each process. Both processes show a positive median reduction for each
long recording. The short recording's interval crosses zero. These are small
measured improvements, not a new throughput breakthrough.

Resident time includes features, encoder/projector, prefill, generation,
synchronization, and cleanup. It excludes loading, admission queueing, and
reading the audio file. The candidate processes the 46.57-second recording in
about 1.20 seconds resident, approximately 39 times real time, on this machine.
These are local Apple M5 Max measurements, not a hardware maximum.

To repeat one paired comparison:

```bash
.build/release/mere.run model benchmark sov /path/to/recording.wav \
  --model-root /path/to/original-sov \
  --draft-model /path/to/dflash \
  --packed-weights /path/to/saved-q8-g64 \
  --experimental-decoder q8 --experimental-attention \
  --generation-experiment rollback --interleave-metal \
  --metal-baseline reuse --warmups 2 --repetitions 12 --json
```

Add `--metal-first` for the reversed starting order. The default Metal comparison
baseline remains MLX; `--metal-baseline reuse` selects the original shift-based
route-reusing specialization. The separate Python shader screen accepts
`--legacy-bit-unpack` for that specialization.

## Where the round time goes

The host profile, recorded before the unpacking change, observes existing
synchronization points without adding GPU barriers. Across 48 rounds, median
host graph construction is 0.382 ms for the
draft and 1.893 ms for target verification. Draft submission takes 0.825 ms,
prediction materialization waits 56.176 ms, and commit takes 0.080 ms. The median
round lasts 59.710 ms. Stage medians are not additive, and prediction wait alone
does not distinguish GPU work from scheduling gaps.

A separate Metal System Trace attaches after warmup and records round signposts.
For **134 complete rounds**, unioned active GPU-execution intervals from this
process cover a median **97.67%** of the round. Median round duration is
47.521 ms; median time without this process's GPU work is **1.104 ms**. Eight
rounds contain an individual gap above 1 ms. Other-process GPU work covers less
than 0.95% of any sampled round and has a zero median. This trace indicates that
the existing pipeline keeps the GPU fed; it leaves little exposed idle time
for a host-only scheduling change to remove.

GPU-active intervals include work stalled on memory. They are not a claim of
97.67% ALU utilization. The traced and untraced runs are separate; their wall
times must not be compared as an optimization result.

## Shader attribution

Before the unpacking change, a programmatic capture covers native decode round
three after two warmups, without adding evaluation barriers. Apple `gpudebug`
collects an overlapping
counter replay at maximum GPU state, sampling four of forty cores. It reports
46.165 ms GPU time and marks consistent state as `no`.

| Shader | Reported GPU cost | Invocations |
| --- | ---: | ---: |
| Q8 expert gate/up, K=2816, N=1408 | 30.95% | 30 |
| Q8 expert down, K=704, N=2816 | 20.53% | 30 |
| BF16 RMS normalization | 0.88% | 32 |
| FP32 route sorting | 0.49% | 30 |

The two expert shaders account for **51.47% of reported GPU cost**. Their
compiled code uses 60 and 47 temporary registers, respectively, with no spills.
The replay reports average bandwidth of 318.69 GiB/s, kernel occupancy of
22.52%, and memory-request stalls as the top occupancy influence for 17.37% of
the timeline. These are replay counters, not native wall-time percentages.

The named shader table accounts for only 56.36% of reported cost. Unnamed
matrix dispatches contribute additional cost. The largest unnamed dispatch has
12.51% reported cost and binds a 1.38 GiB weight buffer and a 7.50 MiB output
buffer, consistent with the 15-row draft vocabulary projection. That attribution
is inferred from buffer sizes and call order. The expert table is not a complete
model cost breakdown.

The capture and raw counter data live on the external SSD at
`/Volumes/SALVATION/MereRunProfiles/sov-decode-2026-11-14/`.
The receipt's binary and source hashes identify the final comparison build;
the earlier diagnostic build's binary hash was not recorded.
An initial launch trace hit its time limit before measured inference and is
excluded. The attached trace's 24 recorded transcriptions match frozen tokens
and acceptance counts. The original baseline, host profile, and capture add
nine matching outputs, for **145 checked outputs** in this pass.

## Kernel change and rejected candidate

The retained specialization reinterprets existing Q8 words as four unsigned
bytes. Packed files, scales, biases, products, FP32 reduction order, and BF16
output rounding are unchanged. Larger or denser prefill batches retain MLX's
matrix path. Template specialization preserves a shift-unpack control for paired
measurement. No checkpoint conversion is required.

On captured layer-zero verification activations, byte unpacking reduces isolated
expert-block time by 3.02% and 4.98% in two screens, with exact outputs. The
native generation gain is smaller, as shown above. A second candidate skips
arithmetic for unused rows; it is 24.52% slower in the isolated paired screen
and is rejected. It is not in the runtime.

## Validation and storage

Release build passes. The focused GPU/decoder/CLI/documentation selection runs
37 tests with zero failures and one opt-in residency-budget test skipped because
its receipt path is unset. Both Metal specializations match MLX exactly across
reduction tails and expert boundaries. Tests also cover current-parameter reads,
matrix fallback, decode acceptance/rejection, window rollover, instrumentation
parity, and CLI validation. Strict lint with its cache disabled, Python syntax,
and `git diff --check` pass. The full repository gate remains deferred at the
user's request.

Other jobs were active on the shared machine; admission explicitly queued a run
while four permits were active. This task's builds, profiling, and file copies
finished before the paired timing runs. Pairing and reversal limit drift but do
not turn the machine into an exclusive benchmark host.

Research assets, prior push artifacts, and the draft checkpoint totaling
6.66 GiB were copied to the external SSD and verified file by file with
SHA-256 before removing the originals. Symlinks preserve their original paths.
The relocation manifest is retained beside the external assets. Target weights
and packed-file contents are unchanged.

This pass identifies device work as the dominant remaining cost. It does not
prove a performance ceiling: GPU-active time can still improve through better
expert and vocabulary kernels. It provides no evidence for a large remaining
host-scheduling gain.
