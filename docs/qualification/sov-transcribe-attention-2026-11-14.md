# SOV shared masks and direct partial RoPE — 2026-11-14

The target decoder now has an opt-in attention scheduling path that shares
identical masks across layers and avoids proportional partial-RoPE slicing and
concatenation. It retains the existing attention matrix operations,
normalization, weights and mask values.

The confirmation study observed **3.11% lower median paired resident latency on
the 46.57-second recording** and **1.35% on the 3.52-second recording**, relative
to the existing Q8 Metal expert path with DFlash. Background compilation and
Node workers made host load variable. These are provisional results under the
recorded load; they do not establish idle-system latency or a new peak speed.
The path remains experimental and off by default.

All **160 measured transcriptions and eight recorded first inferences** matched
the frozen token IDs. The captured native verifier hidden state was bit
identical. The [JSON receipt](./sov-transcribe-attention-2026-11-14.json) includes
every measured result, both benchmark series, operator screens, diagnostic
profiles, host observations, and source and binary hashes.

## What the profile found

On an M5 Max with 128 GiB memory, macOS 27.2 (26B123) and Xcode 27.2 (27B52),
the target verifier body accounted for about 80% of the staged generation work.
Instrumenting the first sliding-attention layer and first full-attention layer
showed substantial mask and RoPE overhead:

| Synchronized diagnostic stage | Sliding layer 0 | Full layer 5 |
| --- | ---: | ---: |
| Input normalization | 0.159 ms | 0.141 ms |
| Q/K/V projections | 0.295 ms | 0.274 ms |
| Head normalization | 0.222 ms | 0.220 ms |
| RoPE | 0.182 ms | 0.445 ms |
| Cache append | 0.228 ms | 0.236 ms |
| Mask construction | 0.395 ms | 0.347 ms |
| Attention | 0.304 ms | 0.310 ms |
| Output projection | 0.232 ms | 0.275 ms |

These are debug diagnostics with synchronization after each stage. They include
barrier overhead, change overlap, and must not be summed into release latency.
The baseline and optimized diagnostic captures used different builds; their
target-body timings are not the basis for the performance claim.

The existing full-attention heads are 512 wide. Splitting verification queries
into smaller groups to select vector attention was tested on captured native
inputs. It was slower and changed output: relative RMSE was about 0.0078 for
sliding attention and 0.00014 for full attention. Boolean and implicit causal
masks matched output in this screen but did not offer a consistent isolated
operator gain. Those dispatch changes were not integrated.

## Implemented changes

`Gemma4AttentionMaskCache` lives for one target forward call. It keys masks by
query length, query offset, key length, active window and dtype. Layers can reuse
the same array; inactive windows share the causal mask. On these recordings,
the 30 target layers can reuse one position mask per forward call. Padding and
vision masks bypass reuse. A later call builds a new cache, preserving position
and window changes.

Direct proportional RoPE supplies a full-width frequency vector to the existing
native MLX kernel. Infinite frequencies give the untouched portions of both
head halves zero angular velocity. This preserves the trained frequency basis
and half layout while eliminating the intermediate packing and concatenations.
For captured full-attention queries at offset 779, isolated operator time fell
from 0.168 to 0.146 ms; keys fell from 0.165 to 0.156 ms. All eight captured
query/key and offset cases were bit identical. These Python operator screens
use stored tensor values and are separate from native inference timings.

`setExperimentalAttention` changes these two policies without changing expert
dispatch or parameters. Ordinary transcription and other Gemma runtimes retain
their existing defaults. The existing custom expert kernel is described in the
[Metal MoE qualification](./sov-transcribe-metal-moe-2026-11-13.md).

## End-to-end method and results

Both modes used the original checkpoints, Q8 group-64 Metal experts, the MLX
encoder/projector, DFlash, a prefill chunk size of 1024 and a 448-token limit.
Dynamic-int8 encoder inference was disabled. Target revision was
`505d8fae243924c0d8acdc622f7f0abab344eba3`; draft revision was
`8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05`.

Each process loaded one target and one recording. Two warmups and ten measured
inferences ran per mode. Each adjacent pair reversed order on alternating
cycles. The second session started optimized and reversed recording order.
Dispatch switches occurred outside the timer. External synchronized resident
time includes request cleanup and excludes checkpoint loading and file reading.

The first series coincided with another checkout's Swift build and CPU-heavy
Node workers. Unchanged feature extraction, encoder and cleanup times drifted
substantially. A complete confirmation series repeated the protocol on the same
binary. Background workers remained active; this was not an idle-machine run.

| Confirmation recording | Session | Reference median | Optimized median | Median paired reduction | Optimized wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.52 s | 1 | 425.0 ms | 420.5 ms | 1.35% | 9/10 |
| 3.52 s | 2 | 410.5 ms | 400.9 ms | 1.42% | 8/10 |
| 46.57 s | 1 | 1852.8 ms | 1783.8 ms | 4.68% | 9/10 |
| 46.57 s | 2 | 1839.5 ms | 1803.5 ms | 2.98% | 7/10 |

The confirmation headline is the median of the 20 adjacent per-pair latency
reductions for each recording, rather than a ratio of pooled mode medians.
The original series yielded 0.86% on the short recording and 5.27% on the long
recording. Its long-clip session medians varied from 11.16% to 1.56%; that
unstable large result is not used as the headline. All samples are retained.

The supported finding is a modest benefit under the recorded host conditions.
A quiet-machine confirmation remains necessary before claiming a new peak
resident latency. Absolute times must not be compared directly with earlier
qualification runs, and percentage gains from different studies must not be
added together.

## Parity and validation

- Both recordings matched the previously frozen BF16+DFlash token IDs in all
  160 measured runs and eight recorded first inferences: 13 tokens for
  `Fixtures/D1/released/speech.wav`, 138 for
  `Fixtures/Nemotron3/returning-speaker.wav`. Other warmups are excluded.
- DFlash verification rounds, accepted draft tokens, rejected rounds and reused
  verification rounds were unchanged between modes within each recording.
- Native candidate IDs `[1, 16]` and the complete captured BF16 verifier hidden
  state `[1, 16, 2816]` were bit identical.
- Thirty-nine focused tests passed, including partial RoPE across FP32, BF16
  and FP16, strided inputs, extra tail dimensions, multiple offsets, mask keys,
  window crossings, shared KV layers, padding masks, expert kernels, SOV runtime
  and weights, and CLI validation and documentation contracts.
- Two real-checkpoint diagnostic profile tests passed. Release builds used
  `swift build -c release --product mere.run -Xswiftc -disable-cmo`.
  Scoped strict SwiftLint and `git diff --check` passed. The full repository gate
  remains deferred at the user's request.

These two English recordings do not establish multilingual recognition quality.
The changes preserve tested outputs; they remain opt-in pending broader accuracy
and quiet-machine performance qualification.

## Reproduce

Compare attention policies on the same resident Q8 Metal target:

```bash
.build/release/mere.run model benchmark sov \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --model-root .build/qualification/sov/target \
  --draft-model .build/qualification/sov/draft \
  --experimental-decoder q8-metal --interleave-attention \
  --warmups 2 --repetitions 10 --json
```

Repeat with `--attention-first` to reverse starting order. JSON aligns
`measurementKernels`, `residentSeconds` and `measurements`; optimized labels end
in `+attention`. For standalone optimized runs, replace `--interleave-attention`
with `--experimental-attention`.

`scripts/model-conversion/profile_sov_attention.py` reproduces the isolated
attention and RoPE screen from native captures. The opt-in
`SOVDecoderKernelProfileTests` accepts `MERERUN_PROFILE_SOV_ATTENTION=1` to capture
attention inputs and substages, and `MERERUN_PROFILE_SOV_OPTIMIZED_ATTENTION=1`
to enable the new target path.
