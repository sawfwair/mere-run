# SOV decoder and checkpoint-loader qualification — 2026-11-13

The native fused Q8 expert path reduces resident latency on both recordings,
with exact agreement against the frozen original-target tokens. Direct
file-backed checkpoint materialization also removes a large temporary memory
copy from the default BF16 path. Ordinary transcription retains BF16 experts;
Q8 remains an explicit benchmark/API experiment.

| Decoder with DFlash | 3.52 s recording | 46.57 s recording | Prepared MLX active memory | Long verification rounds |
| --- | ---: | ---: | ---: | ---: |
| BF16 reference | 306.2 ms | 1.390 s | 49.05 GiB | 15 |
| BF16 verification GEMV | 325.3 ms | 1.310 s | 49.05 GiB | 15 |
| Fused Q8 · group 64 | 254.7 ms | 1.233 s | 29.11 GiB | 16 |
| Fused Q8 · group 32 | 256.5 ms | 1.335 s | 30.44 GiB | 17 |

Q8 group 64 reduces median latency by 16.8% short and
11.3% long against the same-build BF16 reference. Prepared model memory falls
40.7%. These measurements use the M5 GPU, not the standalone ANE.

## Checkpoint memory

The former SOV loader read each tensor through Foundation `Data` before copying
it into MLX. The replacement uses the existing file-backed, materialized loader
with the same key mapping, shape checks, missing-weight checks, and original
dtypes. Router tensors remain FP32. No weights are truncated or omitted.

In the forward BF16 sessions, maximum process footprint fell from
103.50 to 54.41 GiB; prepared MLX active memory stayed at
49.05 GiB. Observed preparation time fell from 19.13 to
5.10 seconds. Filesystem cache state was not controlled, so these
load times are observations rather than a cold-disk speed guarantee.
Q8 packs one layer at a time and releases retired allocator buffers between
layers. Its maximum process footprint is 49.75 GiB in the clean study,
including full-precision preparation. The 29.11 GiB active figure does not
represent total application memory or establish support for a 32 GiB Mac.

## Expert path

Captured layer-0 routes select 39 distinct experts for a 16-token verification
block and 102 for a 779-token prefill. The isolated sorted-expert screen reduced
verification time from 1.206 to 0.907 ms with Q8 group 64. Router/top-k and route
sorting are outside that microprofile. It is a Python MLX screen, not a native
end-to-end result.

Native preparation fuses gate/up weights, packs affine 8-bit groups once, and
replaces the BF16 expert modules. Inference uses MLX gathered quantized matrix
multiplication. Activations remain BF16; FP32 routing, dense layers, attention,
DFlash, the vocabulary head, and the audio encoder retain their original
parameters. No vendored runtime changes are required.

The smaller quantization group does not guarantee faster speculative decoding:
group 32 requires 17 verification rounds here, group 64 requires 16, and BF16
requires 15. All produce the same final tokens. BF16 GEMV improves the long
median but its short timings vary considerably, so it remains experimental.

## Parity and measurement scope

All 80 measured sequences and 16 first-recorded sequences match the frozen
13-token short and 138-token long references. All variants retain 0/11 short
and 2/129 long word errors under the existing normalization. Both new-loader
mel tensors and complete projected embeddings are bit identical to the frozen
original-loader tensors: maximum absolute error is zero. Full padded windows
and all projected audio tokens remain present.

An earlier experiment combined Q8 experts with the custom dynamic-int8 audio
encoder. Its long transcript changed five token positions and increased word
errors to 5/129. That combination is not qualified for exact parity. The raw
initial study is retained separately in the receipt; its noisier timing data
is not pooled with the clean results.

Each clean variant runs in a fresh process, loads once, and stays resident for
both recordings. There are two warmups and five measured runs per recording
in each of two sessions. The second session reverses variant and recording
order. `caffeinate -di` holds temporary idle-sleep assertions. No compiler or
profiling job runs during measured inference. Loading and audio-file reads are
recorded separately; resident latency includes features, encoder/projector,
full prefill, DFlash verification, cleanup, and synchronization.

| Decoder | Short range, ten runs | Long range, ten runs |
| --- | ---: | ---: |
| BF16 reference | 290.2–334.6 ms | 1.362–1.447 s |
| BF16 verification GEMV | 285.3–378.7 ms | 1.290–1.496 s |
| Fused Q8 · group 64 | 246.2–283.1 ms | 1.209–1.290 s |
| Fused Q8 · group 32 | 247.7–262.8 ms | 1.283–1.379 s |

Two English recordings do not establish multilingual accuracy, general WER,
or parity on other inputs. The model's Blackwell serving measurements use a
different hardware/software setup; these results do not reproduce that serving
benchmark. Earlier mere.run receipts also use different binaries, so the
same-build ablation above is the comparison that isolates this experiment.

## Reproduction and checks

The qualification executable was built with:

```bash
swift build -c release --product mere.run -Xswiftc -disable-cmo
```

Two ordinary release rebuilds were stopped during prolonged Swift optimizer
passes. Cross-module optimization is disabled only in the recorded
qualification command; package defaults are unchanged. All clean variants use
one executable with release optimization and the same GPU runtime and kernel
compilation settings.

```bash
caffeinate -di .build/release/mere.run model benchmark asr \
  Tests/MereRunCoreTests/Fixtures/D1/released/speech.wav \
  Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --backend sov --model-root /path/to/SOV-Transcribe-1 \
  --draft-model /path/to/SOV-Transcribe-1-DFlash \
  --experimental-decoder q8 --warmups 2 --repetitions 5 --json
```

Omit `--experimental-decoder` for BF16, or select `bf16-gemv` or `q8-g32`.
Repeat with reversed variant and recording order. `model benchmark sov` exposes
the same expert policies and can dump embeddings with `--projected-output`.

The focused checks pass 31 Swift tests covering expert packing against
independent per-expert projections, router preservation, loader validation,
frontend numerics, cache verification, CLI policy scoping, and documentation
inventory. Strict scoped lint passes. The full repository gate was not run.

The [JSON receipt](./sov-transcribe-decoder-2026-11-13.json) contains every measured
sequence and stage timing, initial study, loader tensor checks, checkpoint
revisions, source and binary hashes, route capture, and operator screening data.
