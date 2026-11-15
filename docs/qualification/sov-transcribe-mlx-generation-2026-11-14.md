# SOV MLX generation scheduling — 2026-11-14

Ordinary SOV+DFlash transcription uses a pipelined generation loop. Draft IDs
stay on the GPU while the host constructs verification, and verified predictions
supply the next anchor token. The target and draft retain their original weights.
The latest Q8 Metal/attention baseline measured a **5.14% median paired resident
latency reduction** on the 46.57-second fixture and **1.19%** on the 3.52-second
fixture. The original BF16 configuration also passed the final confirmation below.

All **436 recorded CLI outputs** matched frozen token IDs. The pipeline and
shared-cache first-verification hidden states were bit identical to reference.
The [JSON receipt](./sov-transcribe-mlx-generation-2026-11-14.json) includes raw
inference reports, pair summaries, stage diagnostics, memory measurements, and
source and binary hashes. Other warmups are excluded from that output count.

## Paired results

Positive percentages mean lower candidate latency. Resident timing includes
request-stream synchronization and cleanup, and excludes loading and audio I/O.
Generation is a substage of that same inference, not a separate throughput test.

| Configuration and change | Audio | Resident reduction | Generation reduction | Resident wins/pairs |
| --- | --- | ---: | ---: | ---: |
| Original BF16, pipeline | 46.57 s | 9.16% | 11.59% | 8/8 |
| Original BF16, pipeline | 3.52 s | -1.11% | 3.90% | 4/8 |
| Q8 Metal/attention, pipeline | 46.57 s | 5.14% | 7.90% | 17/20 |
| Q8 Metal/attention, pipeline | 3.52 s | 1.19% | 5.13% | 13/20 |
| Q8 Metal/attention, Q8 draft head | 46.57 s | 2.17% | 2.14% | 15/20 |
| Q8 Metal/attention, Q8 draft head | 3.52 s | 0.62% | 2.64% | 13/20 |
| Q8 Metal/attention, shared KV storage | 46.57 s | 2.45% | 3.18% | 15/20 |
| Q8 Metal/attention, shared KV storage | 3.52 s | -0.42% | 0.66% | 7/20 |

The Q8 study used two fresh processes per experiment and recording: two warmups
per mode, then eight pairs in the first session and 12 in the second. Adjacent
pairs alternate order; the second session reverses recording, experiment, and
starting-mode order. Each pair uses one resident target with identical parameters.

The final BF16 confirmation uses the final binary, original experts, reference
attention, two warmups per mode, and eight measured pairs per recording. The
JSON also retains an earlier four-pair BF16 pilot. The final binary changes the
generator's default scheduling and embedded documentation; the explicitly
selected pipeline algorithm matches the Q8 study's binary.

The short recording spends most of its time in the encoder and prefill and uses
one verification round. Its end-to-end results do not establish a consistent
gain. The repeated long-clip comparisons support the pipeline change.

Background CPU activity and substantial stage drift remained. No compiler or
profiler from this task ran during release measurements. The repository's
machine admission serialized its CLI benchmarks. These are results on this
machine and these recordings, with paired comparisons used to limit timing drift.

## Implementation

The [decoder](https://github.com/sawfwair/mere-run/blob/400cc5f2be5ce7b7ccb12ffab62d90f155d7949c/Sources/AudioSOVTranscribeModel/SOVTranscribeDecoder.swift)
starts draft evaluation asynchronously, builds target input IDs from the device
proposal array, and reads proposals and predictions together. It carries the
verified next prediction on the host after acceptance or prefix reuse. Replay
still obtains its anchor from the replayed logits, preserving the reference
path's finite-precision behavior. Intermediate committed-cache evaluations
become dependencies of the next round. Request stream synchronization remains
part of the measured call, including cancellation cleanup.

Ordinary transcription selects this pipeline automatically with DFlash. The
specialized `model benchmark sov` command defaults to `reference` scheduling;
select `--generation-experiment pipeline` to measure the production policy.
Ordinary decoder weights remain BF16. The Q8 expert policy is a separate benchmark
selection, as in the earlier [MoE qualification](./sov-transcribe-metal-moe-2026-11-13.md).

The optional `shared-cache` path uses one sliding-attention concatenation and
retains a view of its valid suffix. Existing cache forks and verified-prefix
commits preserve isolation and history. Synthetic tests cover multiple heads,
unequal K/V widths, wrapped windows, oversized chunks, and mutations after forks.
Neither released audio fixture fills the target's 1,024-token window; real-checkpoint
qualification of wrapped shared storage remains outside these measurements.

The optional `draft-q8` path prepares a separate group-64 Q8 vocabulary head and
keeps the target embedding and output head unchanged. It adds **0.7305 GiB** of
prepared active memory. Ten paired isolated head/argmax samples measured
**3.63 ms BF16 versus 2.28 ms Q8**, a 37.30% median paired reduction. This debug
Swift diagnostic synchronizes each operator and excludes the draft body; the
end-to-end gains in the table are much smaller. Quantized proposals can change
acceptance and verification grouping, so frozen-token checks remain required.
Acceptance vectors were unchanged on these recordings.

JSON `acceptedLengths` records raw agreeing proposal prefixes. `emittedDraftTokens`
counts drafts included before EOS. The long Q8 recording has 127 raw
agreements and 122 emitted drafts; the short recording has 15 and 12. The
existing `acceptedDraftTokens` counter retains its original raw-agreement meaning.

The pinned [upstream DFlash source](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1-DFlash/blob/8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05/modeling_dflash.py)
explicitly selects causal sliding layers and bidirectional full layers, matching
the native mask. No attention-mask correction was needed.

## Allocator and combined experiments

`retained-allocator` retains MLX's buffer cache across requests instead of clearing
it during cleanup. `combined` selects pipeline scheduling, the Q8 draft head,
shared KV storage, and allocator retention. Separate processes are necessary
because a cache-clearing reference changes the following request's starting state.
The second session reverses recording and policy order.

| Session | Recording | Reference seconds | Retained allocator seconds | Combined seconds |
| --- | --- | ---: | ---: | ---: |
| A | long | 2.302 | 2.133 | 1.881 |
| A | short | 0.453 | 0.449 | 0.446 |
| B | long | 1.533 | 1.688 | 1.770 |
| B | short | 0.305 | 0.277 | 0.507 |

These between-process comparisons reverse direction and do not establish a stable
additional end-to-end gain. Both policies remain opt-in. Steady cached buffers
occupied about 0.8–0.9 GiB on the short recording and 4.2–5.2 GiB on the long one.
The experiment respects MLX's configured process-wide cache limit and releases
buffers on unload. Reported peak active allocation includes model preparation;
it is not the sum of active and cached resident memory.

## Configuration and validation

- Apple M5 Max, 128 GiB; macOS 27.2 (26B123), Xcode 27.2 (27B52).
- Target revision `505d8fae243924c0d8acdc622f7f0abab344eba3`; draft revision
  `8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`.
- Pinned MLX Swift revision `1e077ba968b291e0237d1ab522b271d433fa1713`.
- MLX encoder/projector, DFlash, prefill chunks of 1,024, output limit 448.
  Dynamic-int8 encoder and fused head normalization disabled.
- Q8 study: group-64 custom Metal experts, shared attention masks and direct
  partial RoPE. BF16 confirmation: original experts and reference attention.
- Fixtures: `Tests/MereRunCoreTests/Fixtures/D1/released/speech.wav`, 13 frozen
  tokens; `Fixtures/Nemotron3/returning-speaker.wav`, 138 frozen tokens. The
  second fixture path is relative to `Tests/MereRunCoreTests/`.
- Thirty-one focused GPU tests and 12 CPU tests passed. One GPU-only test was
  skipped on CPU. Independent FP32 NumPy fixtures used `MLX_ENABLE_TF32=0`;
  release checkpoint parity and timings used default precision.
- CLI inventory generation and the checkpoint diagnostic passed. The complete
  verifier hidden state `[1,16,2816]` matched exactly for device-resident IDs and
  shared storage. Four diagnostic full decodes per profile also matched frozen
  tokens and acceptance lengths.
- Release build, scoped strict SwiftLint, and whitespace checks passed. The full
  repository gate remains deferred at the user's request.

Two English fixtures do not qualify multilingual recognition quality, longer
recordings, or other Apple Silicon devices.

## Reproduce

```bash
swift build -c release --product mere.run -Xswiftc -disable-cmo
.build/release/mere.run model benchmark sov \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --model-root .build/qualification/sov/target \
  --draft-model .build/qualification/sov/draft \
  --experimental-decoder q8-metal --experimental-attention \
  --generation-experiment pipeline --interleave-generation \
  --warmups 2 --repetitions 10 --json
```

Add `--generation-first` to reverse starting order. For original BF16 and reference
attention, omit `--experimental-decoder q8-metal --experimental-attention`.
Substitute `draft-q8` or `shared-cache` for individual paired experiments.
Allocator retention and `combined` require separate resident runs.

The optional `SOVGenerationProfileTests` requires `MERERUN_TEST_MLX_DEVICE=gpu`,
`MERERUN_TEST_SOV_PATH`, `MERERUN_TEST_SOV_DFLASH_PATH`,
`MERERUN_PROFILE_SOV_EMBEDDINGS`, and an output prefix in
`MERERUN_PROFILE_SOV_GENERATION`. It checks hidden-state and token parity and
writes host-stage diagnostics plus the isolated draft-head samples. These
profiles are separate from the release end-to-end measurements.
