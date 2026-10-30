# Native DRAMA-3 checkpoint qualification

The extended bounded probes pass batch speech, reference conditioning, short and
paragraph streaming, and cancellation on a 36 GiB system. See
[extended results](#extended-results-after-the-sampler-correction) for scope,
memory peaks, latency, and remaining quality limits.

The released checkpoint is pinned to
`fishaudio/drama-3@dd64a85cd82e9521f7202610ede97b367fc83946`.
All 18 source files, totaling 41,502,593,827 bytes, were verified against the
pinned Hub LFS SHA-256 or Git blob digest before conversion. Built with Fish Audio.
The original checkpoint remains subject to the Fish Audio Community License.

## Q4 export

The native CLI exported all 31,525 model tensors, packing routed experts as
affine Q4/group-64 and preserving other model dtypes and the original codec.
The standalone directory contains 99 files, including 88 safetensor shards,
totaling 27,408,212,020 bytes. The export completed in 199.92 seconds with a
2,624,113,256-byte peak physical footprint reported by macOS `time -l`.
These are conversion measurements, not inference memory or throughput results.

The `drama3-quantization.json` receipt records source and output SHA-256 hashes.
Its 11 source entries match the independently verified download manifest.
The output also preserves `LICENSE` and adds the required attribution `Notice`.

External storage exposed two allocation problems: safe mapping copied entire
shards on the removable volume, and long Foundation read/hash loops retained
autoreleased buffers. The native reader now reads only headers and requested
tensors, and both tensor reads and artifact hashing drain temporary buffers
within each iteration. The completed export validates the bounded working set.

## Codec comparison

The independent reference uses the pinned SGLang-Omni DAC graph at
`2cfb3b67becb437b70751defe37e63d3af4eedb0`, evaluated with PyTorch on CPU.
Every learned state-dict key matched after excluding regenerated rotary and
causal-mask buffers. Native inference remains Swift/MLX.

- Decoding 32 deterministic acoustic frames produced 65,536 waveform samples:
  RMSE `0.00019327203335706145`, mean absolute error
  `0.00007681941497139633`, and maximum absolute error `0.003674132749438286`.
- Encoding a one-second, 220 Hz sine wave produced identical codes for all
  220 positions across ten codebooks. The native encoder pads partial input
  to the complete 2,048-sample acoustic frame, matching released `DAC.encode`.

These synthetic component probes do not establish speech quality or Q4 quality
loss. Whole-file SHA-256 verification precedes restricted codec archive parsing.

## Local contract validation

`MERERUN_SWIFT_DISABLE_INDEX_STORE=1 ./scripts/check.sh` passed 5,445 XCTest
cases with 451 skips and zero failures, plus 193 Swift Testing checks. Strict
lint, model boundaries, all three package-policy variants, CLI help, and hygiene
checks passed. `pnpm docs:build` passed. Hosted CI and inference on other
platforms were not verified in this run.

## Reproduce

```bash
mere.run model optimize /path/to/drama-3-source --output /path/to/drama-3-q4
MERERUN_TEST_DRAMA3_ROOT=/path/to/drama-3-q4 \
MERERUN_TEST_DRAMA3_EVIDENCE=/path/to/evidence \
  swift test --filter Drama3InstalledCheckpointTests/testReleasedCheckpointSpeechAndMemory
```

The installed-checkpoint tests are opt-in and write artifacts outside the
repository. Full speech qualification acquires the normal shared inference
lease. The batch result below covers one short seeded English request on a
36 GiB system. The catalog's 32 GB minimum and broader 36 GB recommendation
remain estimates; other workflows need separate qualification.

## Runtime memory follow-up

The October 16, 2026 batch speech attempt passed normal admission and reached
autoregressive generation. It peaked at 39,699,848,680 bytes of physical
footprint. System swap grew to 23,754.94 MiB and internal disk headroom fell to
7.7 GiB. The attempt was stopped below the existing 8 GiB disk floor, before
producing a completed waveform or an ASR result. The Q4 export alone does not
establish that the runtime fits a 36 GiB system.

Batch synthesis now separates three phases: encode reference speech, generate
integer acoustic codes, and decode those codes into a waveform. It releases the
text/tower checkpoint and tokenizer before final codec loading. Streaming
requires both graphs and has a separate memory boundary. Audio-tower masks
preserve the loaded embedding dtype, avoiding promotion of BF16 matrix products
to FP32. The runtime clears inactive Metal allocation buffers between frames.

The opt-in speech probe writes `native-speech-stages.json`, recording active,
cached, and peak MLX memory and process footprint at phase and frame progress
events. It verifies that batch synthesis releases the text/tower checkpoint
before codec decoding. Repeat the released-checkpoint probe through normal
shared admission before claiming an improved measured peak or successful
36 GiB qualification. The Q4 artifact and codec arithmetic remain unchanged.

The October 23, 2026 follow-up passed admission but was stopped by the disk
headroom guard after reaching 35,490,109,896 bytes of physical footprint. No
completed waveform or ASR result exists for that attempt. Inspection found
that first-use MoE fusion retained the original gate/up expert banks alongside
the fused banks. For this checkpoint, retaining both forms can add 11.25 GiB
of packed weights and quantization metadata.

The DRAMA loader now uses the shared MoE ownership-transfer path: prepare the
fused gate/up bank for one loaded layer, release its source banks, and clear
inactive buffers before loading the next layer. Prompt processing also clears
inactive buffers between backbone layers. A loader regression checks that the
source banks are released, and a synthetic BF16 probe covers both attention
families. Phase memory evidence is written after each progress event so a
controlled stop retains the completed phase observations. The memory peak of
this ownership fix is measured in the following batch probe.

## Released-checkpoint batch result

The default fused-expert runtime at commit `4be05a235d27a56c365e2b63905e7249ca8065bf`
completed the seeded English batch probe on October 23, 2026, through normal
shared admission on a system with 38,654,705,664 bytes of physical RAM (36 GiB).
The Q4 checkpoint and codec files were unchanged.

- Input: `[excited] Hello! The native speech engine is running on this laptop.`
  Temperature: `0.6`; seed: `42`.
- Output: 81 acoustic frames, 165,888 mono samples at 44,100 Hz, or
  3.761633 seconds of PCM16 audio. Sample peak: `0.494446`; clipped samples: zero.
- Wall time: 105.52 seconds, including cold checkpoint loading and waveform
  decoding. This is not a warm generation-throughput measurement.
- Peak process physical footprint: 26,165,418,296 bytes (24.37 GiB), compared
  with 39,699,848,680 bytes in the original stopped attempt. Peak MLX allocation:
  25,796,160,132 bytes. Active MLX allocation before codec decoding fell to
  20 bytes, verifying release of the text/tower checkpoint.
- Sampled total system swap rose from 3.61 GiB to 6.89 GiB. These system totals
  include other apps; this result does not establish operation without swap.
- Native Parakeet returned the exact expected ten-word transcript: zero word
  errors. Score the plain `--no-timestamps` transcript; the default export also
  repeats text in timestamped alignment lines.

This qualifies the short batch probe on the tested 36 GiB system. It does not
establish broader speech quality, Q4 quality loss against the original weights,
long requests, reference cloning, streaming capacity, or other-platform behavior.
Batch requests release resident resources; subsequent calls reload the model.

After the ownership fix, the full local gate passed 5,447 XCTest cases with
451 skips and zero failures, plus 193 Swift Testing checks. The focused runtime
and loader suite passed, and Metal tests preserved both BF16 attention paths
and expert output before and after fusion. The documentation build passed.
Hosted CI remains unverified.

## Extended qualification and semantic sampling

The first 57-word paragraph probe reached the 750-frame acoustic limit and
correctly published no final WAV. Its peak physical footprint was
26,165,778,768 bytes; memory fit did not establish successful long-text synthesis.
The same runtime completed a 16-word reference-conditioned request with zero
word errors. Remaining old-sampler probes were stopped after identifying the
upstream semantic sampling correction.

The architecture pin used top-k 30 for the main semantic draw. Upstream revision
`444001b0e7bbf0f87a344a13b86a63da21b70395` removes that default cap because it can
make generation fall silent after its first phrase, and retains the token that
crosses the nucleus threshold for semantic draws. The native implementation
now follows that correction, including the repetition-aware redraw. Acoustic
heads retain top-k 30 and exclude the crossing token. The 750-frame limit and
normal admission floors are unchanged. Deterministic regression probes check
the semantic candidate range and both nucleus-filtering policies.

## Extended results after the sampler correction

All seven bounded probes passed through normal shared admission on the same
36 GiB system. Runtime source: `cac7aa6fe72aa04e54eb0ee51537e7073e5d836e`.
The original checkpoint and expert-only Q4 files were unchanged.
The [structured receipt](./extended-results.json) records hashes, requests,
transcripts, event counts, waveform comparisons, and memory measurements.

| Probe | Audio (s) | Word errors | Peak footprint (GiB) | Wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| Paragraph batch | 23.173515 | 0/57 | 25.70 | 142.21 |
| Short batch | 4.365351 | 0/10 | 24.37 | 104.92 |
| Reference clone batch | 5.804989 | 0/16 | 24.37 | 107.63 |
| Short stream | 4.365351 | 0/10 | 27.94 | 105.98 |
| Reference clone stream | 5.804989 | 0/16 | 28.35 | 131.21 |
| Paragraph stream through CLI | 23.173515 | 0/57 | 32.86 | 183.52 |
| Cancel after first chunk | None published | Cancelled | 26.92 | 133.15 |

The paragraph contains 57 words and produces 499 acoustic frames. All 57 words,
including the final sentence, are recovered exactly. The short baseline is
regenerated after the sampling correction, and its audio plus exact ten-word
transcript conditions the clone cases. The clone reads a different 16-word
sentence. This synthetic reference verifies the conditioning path and content;
it does not establish perceptual speaker identity or human-reference likeness.

Both short streams and the paragraph stream preserve the seeded batch sample
counts. Their cumulative chunk counts increase monotonically, and each final
file matches its stream. Maximum sample differences against PCM16 batch exports
are `0.00004119`, `0.00004208`, `0.00004709`, respectively,
below the frozen `0.0002` threshold. Streaming WAVs retain float32 samples;
batch exports use PCM16. All completed waveforms are finite and have zero
clipped samples.

Cold times to first audio are 96.93, 113.01, and 96.07 seconds
for the short, clone, and paragraph streams. These runs include model loading
in a fresh process; the filesystem cache was not flushed. They do not establish
warm throughput or interactive latency.

Cancellation after the first chunk releases the graphs in 3.716 seconds,
leaving 32 bytes of active MLX allocation, no published WAV, and no temporary
audio file. Batch phase traces verify model release before codec decoding.
System swap remains in use: totals include other applications, so these results
do not establish operation without swap.

Reproduce an extended installed-checkpoint case with:

```bash
MERERUN_TEST_DRAMA3_ROOT=/path/to/drama-3-q4 \
MERERUN_TEST_DRAMA3_EVIDENCE=/path/to/new-case-evidence \
MERERUN_TEST_DRAMA3_CASE=paragraph \
MERERUN_TEST_MLX_DEVICE=gpu \
  swift test --disable-index-store --filter Drama3ExtendedCheckpointTests
```

Cases are `paragraph`, `clone`, `stream`, `cloneStream`, and `cancel`. Clone
cases also require `MERERUN_TEST_DRAMA3_REFERENCE` pointing to the short baseline
WAV with the transcript fixed in the test. Streaming cases accept
`MERERUN_TEST_DRAMA3_COMPARISON` pointing to their matching seeded batch WAV.
The public CLI paragraph probe uses `speech synthesize` with
`--model speech-tts-drama-3 --seed 42 --temperature 0.6 --stream` and
`--stream-chunk-tokens 25 --progress-json --receipt`, then independently checks
its waveform, cumulative chunk counts, and plain native ASR transcript.

The full local gate passed 5,450 XCTest cases with 452 expected skips and zero
failures, plus 193 Swift Testing checks. Focused Metal sampling regressions and
the documentation build passed. Hosted CI remains unverified because the
review-checks connector requires an account connection. Delivery follows the
protected merge queue after the integration checks below.

These results qualify the recorded English requests, the tested synthetic
reference, and their streaming paths. Requests that reach the 750-frame cap
still fail without publishing a truncated final file. Broader languages,
multi-speaker quality, perceptual voice similarity, quality loss against the
original checkpoint, maximum-length references, and other-platform inference
remain unqualified.

## Integration with current main

The branch integrates main at `5850dc4f75a9aede3a018647a5db29e4618da178`.
Local integration source: `e1efec52091acac725b4778a15ba47120666ee13`.
Conflict resolution preserves Phonon, Breeze, Qwen3-TTS, and DRAMA-3 for managed
IDs and local paths. A regression checks those distinct routes; the Phonon
language control remains available while DRAMA-3 ignores language hints.
Canonical CLI inventories and Studio scope fixtures regenerate without drift.

The full local gate passes 5,539 XCTest cases with 462 expected skips and zero
failures, plus 195 Swift Testing checks. Three Metal probes cover semantic
sampling and BF16 through both backbone attention families. The debug macOS app
bundle builds and passes ad-hoc signature verification. The documentation build
passes.

The repeated released-checkpoint smoke produces the exact ten-word transcript
and a byte-identical WAV to the corrected-sampler short baseline above. Its peak
physical footprint is 26,165,549,392 bytes. The
[integration receipt](./main-integration-20261023.json) records the matching hash
and local gate results. This repeats the short case on the integrated branch;
the seven-probe receipts above retain their original source revision. Hosted
checks and the protected merge queue remain separate delivery gates.

## Public Q4 distribution

The measured export is published as
[Sawfwair/DRAMA-3-MLX-4bit](https://huggingface.co/Sawfwair/DRAMA-3-MLX-4bit/tree/543e9d0829856cc0657fa0fd6bb2903015c094fb)
at immutable revision `543e9d0829856cc0657fa0fd6bb2903015c094fb`. All 104 publication
files were verified by size and SHA-256 before public release, then verified
anonymously at that revision. The bundle totals 27,408,247,610 bytes.
The original 99-file measured export is unchanged on disk; publication preserves
its README as `UPSTREAM_MODEL_CARD.md` and adds usage, modification, checksum,
and qualification documents. No weights, tokenizer files, or codec bytes changed.

The [publication receipt](./huggingface-publication.json) records every file's
hash and size. The managed `speech-tts-drama-3` download uses this Q4 artifact,
while source provenance and Fish Audio usage terms retain the upstream pin.
This distribution check adds no speech-quality or inference-capacity claims.

A clean native `model pull speech-tts-drama-3` also completed from the public
revision, using an empty model store and cache with registered locations excluded.
All 104 downloaded publication files matched their expected sizes and SHA-256
hashes. The [native download receipt](./native-public-download-20261024.json)
records the source commit and full local gate: 5,540 XCTest cases, 462 expected
skips, zero failures, and 195 Swift Testing checks. Three focused catalog and CLI
tests and the documentation build also passed. This verifies distribution and
installation; speech-quality and inference-capacity limits remain unchanged.
