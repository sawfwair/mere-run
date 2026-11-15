# Native SOV-Transcribe-1

SOV-Transcribe-1 transcribes audio files locally with a Whisper large-v3 encoder
and Gemma 4 26B A4B decoder. It supports English, French, German, Spanish,
Italian, Portuguese, and Dutch. Natural-language instructions control transcript
style and supply names or vocabulary as context.

The Swift/MLX implementation uses the original BF16 checkpoint and keeps the
MoE router in FP32. It runs without Python, Transformers, vLLM, or remote model
code. English fixture transcription and numerical parity have been checked with
the original weights on an M5 Max. See the [qualification report](../qualification/sov-transcribe-ane-2026-11-13.md)
for resident inference, CLI startup, actual ANE execution, and memory observations.
Broad accuracy and memory capacity limits remain unqualified. Target weights
total about 53.3 GB; the catalog's experimental estimates are 64 GB minimum and
96 GB recommended. The qualification used 128 GiB and observed higher peak
process footprints.

## Install and transcribe

Install the model explicitly. Transcription does not download either checkpoint
automatically.

```bash
mere.run model pull speech-asr-sov-transcribe-1
mere.run speech transcribe ./recording.wav --backend sov --max-tokens 8192
```

Selecting the model also selects its native backend:

```bash
mere.run speech transcribe ./recording.wav \
  --model speech-asr-sov-transcribe-1 --language French \
  --instruction "Transcribe the French audio. Preserve names and write numbers as digits." \
  --max-tokens 8192 --output ./transcript.txt
```

`--model` accepts the managed ID, the upstream repository ID, or an installed
local checkpoint directory. `--instruction` replaces the default verbatim
instruction. Without an instruction, a language hint supplies the model's
language-specific transcription prompt. Without either, the prompt is:

> Transcribe the audio verbatim, outputting only spoken words in sequence.

The backend decodes mono audio at 16 kHz, pads it to complete 30-second windows,
and normalizes Whisper log-mel features over the complete padded recording.
Each window produces 375 projected audio tokens. Recordings must contain audio
and be at most 30 minutes long. The prompt, audio, and requested output must
fit the decoder context.

Decoding is greedy and stops on the checkpoint's EOS or turn-ending tokens.
The existing `--max-tokens` default is 448; increase it for longer recordings.
The result contains transcript text and duration. Word timestamps, translation,
streaming stdin and microphone listening are unsupported.
The model is available through the CLI; it is not registered as an API model.

## Enable DFlash

Install and select the companion explicitly:

```bash
mere.run model pull speech-asr-sov-transcribe-1-dflash
mere.run speech transcribe ./recording.wav --backend sov \
  --draft-model speech-asr-sov-transcribe-1-dflash --max-tokens 8192
```

`--draft-model` also accepts the upstream draft repository ID or a local draft
directory. Omitting it uses target-only decoding. The draft cannot transcribe
by itself and is hidden from the primary model inventory.

The five-layer drafter conditions on outputs of target layers 1, 6, 11, 17, 22,
and 27, using zero-based indices. It proposes up to 15 tokens after an anchor.
Its first four layers use causal sliding-window attention; its final layer uses
bidirectional attention. The drafter reads unscaled target embedding rows.

The target verifies every proposal. On rejection, the decoder restores the
committed prefix and replays accepted tokens before continuing with the target's
prediction. Draft blocks never enter the retained draft context. No speedup or
released-checkpoint transcript parity is claimed from the synthetic tests.

## Save and reuse Q8 weights

To opt into affine-Q8 MoE experts, select a packed-weight directory. The first
request reads and quantizes one source tensor at a time and saves native shards.
Later requests load those shards directly, without loading BF16 expert weights
or running quantization again. Use an external SSD to keep the pack outside the
build directory:

```bash
mere.run speech transcribe ./recording.wav --backend sov \
  --model /path/to/sov-checkpoint --draft-model /path/to/sov-dflash \
  --sov-packed-weights /Volumes/SSD/MereRunModels/sov-q8-g64
```

Q8 changes target weights and can change recognition output. Omitting
`--sov-packed-weights` retains BF16. The source directory is still required for
the encoder, projector, configuration, and tokenizer; the draft stays separate.
The pack stores target decoder weights, affine scales and biases, source
identity, quantization settings, and a versioned fused gate/up layout. GPU
shaders and derived Q6 prefill buffers are rebuilt rather than serialized.

A pack is published only after all expected target tensors have been saved.
A changed source, precision recipe, packer version, or shard metadata causes an
error; select a new directory to convert again. Startup validation checks source
configuration/index hashes and shard headers, sizes, and modification times.
It does not scan the complete tensor payloads for corruption on every launch.
Concurrent conversions use separate staging directories; only one can publish
the requested destination.

The benchmark accepts `--packed-weights PATH` with a quantized
`--experimental-decoder` policy, including calibrated mixed recipes. Its JSON
reports cache reuse, packed bytes, source identity, loading time, and load-time
MLX and process memory peaks. A pack depends on precision
and group size, so `q8` and `q8-metal` share one format.
See the [packed-weight qualification](../qualification/sov-transcribe-packed-weights-2026-11-14.md)
for startup, peak RAM, and full-checkpoint parity measurements.

## Core ML encoder and projector

The optional Core ML provider runs the Whisper encoder and stacked-frame
projector as one fixed 30-second graph. Waveform features remain native Swift/MLX;
Gemma and DFlash remain on MLX. The provider selects CPU and Neural Engine
compute units. Device selection alone does not prove ANE execution.

Convert the pinned original checkpoint on macOS 15 or newer:

```bash
uv run scripts/model-conversion/convert_sov_coreml.py /path/to/sov-checkpoint /path/to/sov-coreml
mere.run speech transcribe ./recording.wav --backend sov \
  --model /path/to/sov-checkpoint --provider coreml --coreml-encoder /path/to/sov-coreml \
  --draft-model /path/to/sov-dflash
```

The converter tiles queries and feed-forward work in blocks of 128 frames,
retaining the original normalization and attention orientation. Each query tile
attends to all 1500 keys. `--optimization baseline` reproduces the initial graph.
See the [optimization follow-up](../qualification/sov-transcribe-ane-optimization-2026-11-13.md)
for the measured bottlenecks, rejected precision-changing candidates, and updated
parity and end-to-end results.
For the faster 8-bit configuration, export the palettized preset:

```bash
uv run scripts/model-conversion/convert_sov_coreml.py /path/to/sov-checkpoint /path/to/sov-coreml-p8 --optimization palettized
```

This preset uses per-head attention with 500-query tiles, scaled channel
normalization, 128-frame feed-forward tiles, and learned 8-bit convolution
weight palettes. It preserves the full 30-second key context and the existing
375-token projector interface. The default converter preset remains FP16
`tiled`; the speech execution provider remains MLX.
See the [Core ML research follow-up](../qualification/sov-transcribe-coreml-research-2026-11-13.md)
for repeated token parity, actual ANE execution, published-encoder controls,
e-means measurements, and paired DFlash timings. Qualification covers two
English recordings, rather than a multilingual accuracy benchmark.

The [kernel follow-up](../qualification/sov-transcribe-coreml-kernels-2026-11-13.md)
tests delayed attention normalization, calibrated activation quantization, and
direct ANE execution. It includes export commands, rejected candidates, and
separate resident timing and hardware-execution receipts.

The converter verifies the pinned audio/projector shard checksum and exports
`encoder.mlmodelc` plus `sov-coreml.json`. The native loader verifies the compiled
file closure, checksums, revision, and input/output shapes. It processes windows
one at a time and casts projected embeddings to the decoder's precision.
Python and Core ML Tools are conversion dependencies only.

Profile a prepared model in an optimized build:

```bash
swift build -c release
.build/release/mere.run model benchmark sov ./recording.wav \
  --model-root /path/to/sov-checkpoint --draft-model /path/to/sov-dflash \
  --prefill-chunk-size 1024 --warmups 1 --repetitions 3 \
  --projected-output /tmp/sov-mlx.safetensors --json > /tmp/sov-mlx.json
.build/release/mere.run model benchmark sov ./recording.wav \
  --model-root /path/to/sov-checkpoint --draft-model /path/to/sov-dflash \
  --artifact /path/to/sov-coreml --warmups 1 --repetitions 3 \
  --prefill-chunk-size 1024 \
  --projected-output /tmp/sov-coreml.safetensors --json > /tmp/sov-coreml.json
uv run scripts/model-conversion/compare_sov_providers.py /tmp/sov-mlx.json /tmp/sov-coreml.json \
  --mlx-embeddings /tmp/sov-mlx.safetensors --coreml-embeddings /tmp/sov-coreml.safetensors \
  --output /tmp/sov-comparison.json
```

The report separates audio-file decoding and model loading from resident
inference and retains the first inference in each process separately. Resident timings include synchronized features, encoder/projector
and transfer, prompt/prefill, generation, and cache cleanup. It retains greedy
tokens and DFlash verification/acceptance counts. Tensor dumps happen after the
timed work. The comparison requires matching tokens, finite projected tensors,
relative RMSE at most 5%, and cosine similarity at least 0.999.

Prefill defaults to 1024 tokens per chunk. Partially accepted draft blocks reuse
the target verification's accepted KV rows and hidden states, preserving the
base cache's sliding-window history. Benchmark JSON includes
`reusedVerificationRounds`; a smaller value than `rejectedRounds` identifies
rounds that needed replay. Provider comparisons require matching prefill sizes.

Feature extraction skips FFT work on the known zero waveform tail and restores
the same normalized silence columns. The encoder still receives every mel
frame and retains its full 30-second attention context.

Inspect anticipated placement separately from actual execution:

```bash
uv run scripts/model-conversion/inspect_sov_coreml.py /path/to/sov-coreml \
  --output /tmp/sov-placement.json --require-ane
xcrun xctrace record --template 'Core ML' --output /tmp/sov-coreml.trace \
  --launch -- "$PWD/.build/release/mere.run" model benchmark sov ./recording.wav \
  --model-root /path/to/sov-checkpoint --draft-model /path/to/sov-dflash \
  --artifact /path/to/sov-coreml --warmups 1 --repetitions 1 --json
```

Use Instruments to correlate Core ML execution with Neural Engine activity and
the `SOVEncoderProjector` interval. Keep instrumented runs separate from latency
measurements. A compute-plan receipt describes anticipated placement; it cannot
establish actual execution or an end-to-end speedup.

## Ownership and validation

- `AudioSOVTranscribeModel` owns the encoder, projector, DFlash model, and greedy
  verifier. `MereRunGemmaModel` supplies target computation and attention caches.
- `AudioSTT/SOVTranscribe` owns typed processor checks, tokenizer input, checkpoint
  mapping, waveform features, and request lifecycle.
- Core registers pinned model sources and validates installed assets. AudioCore
  retains transcription options in durable requests. The CLI owns presentation.

```bash
swift test --filter SOV
./scripts/check.sh
```

Numerical tests compare the encoder, stacked projection, and mixed-mask draft
against independent NumPy fixtures. Synthetic generation covers acceptance,
rejection, EOS, budgets, and sliding-cache rollover. Routing tests cover managed
and local models, language policy, option forwarding, and unsupported requests.

The [qualification receipt](../qualification/sov-transcribe-ane-2026-11-13.md)
records original-weight numerical parity, exact DFlash token agreement across
audio providers, actual ANE execution, and short/long latency measurements.
The local test gate uses fixtures; installed-checkpoint tests are opt-in. To run the installed
cross-provider token regression with DFlash:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_SOV_PATH=/path/to/sov-checkpoint \
MERERUN_TEST_SOV_DFLASH_PATH=/path/to/sov-dflash \
MERERUN_TEST_SOV_COREML_PATH=/path/to/sov-coreml \
swift test --filter SOVCoreMLE2ETests
```

## Sources and license

Both checkpoints and their reference modeling code are Apache 2.0 licensed:

- [SOV-Transcribe-1 at the pinned revision](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1/tree/505d8fae243924c0d8acdc622f7f0abab344eba3)
- [SOV-Transcribe-1-DFlash at the pinned revision](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1-DFlash/tree/8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05)

See [third-party notices](https://github.com/sawfwair/mere-run/blob/main/THIRD_PARTY_NOTICES.md) for attribution.

The M5 GPU also has an experimental custom Metal FFN path. Compare it with the
reference using the same checkpoint, audio, prefill size, and DFlash settings:

```bash
.build/release/mere.run model benchmark sov ./recording.wav \
  --model-root /path/to/SOV-Transcribe-1 \
  --draft-model /path/to/SOV-Transcribe-1-DFlash \
  --experimental-dynamic-int8 --warmups 2 --repetitions 5 \
  --projected-output /tmp/sov-int8.safetensors --json
```

This option requires an M5 GPU and macOS 26.2 or later, and cannot be combined
with `--artifact`. It uses custom int8 matrix multiplication, live activation
quantization, and fused Hadamard rotation for the second FFN projection. Static
weight packing is included in model preparation; activation packing is included
in every encoder measurement. Receipts identify the experimental arithmetic.
The original BF16 path remains the default. Qualify embedding error and exact
tokens on your evaluation audio before using these experimental results.

The resident benchmark accepts `--experimental-decoder q8` to pack MoE expert
weights to affine 8-bit groups of 64 and fuse the gate/up projections.
`q8-g32` uses groups of 32; `bf16-gemv` retains BF16 weights and selects GEMV only
for verification blocks of at most 16 tokens. The default is `bf16`.
Preparation time includes weight packing. Resident measurements include routing
and the quantized matrix operations. Routers remain FP32; attention, dense
layers, DFlash, and the vocabulary head retain their original weights.
Q8 changes target arithmetic and requires both quality evaluation and token
comparison with the original target. A DFlash result verified by a Q8 target
does not establish agreement with the original BF16 target.

The benchmark also accepts `q4`, `q6`, and `mixed`. The first two use affine
groups of 64; `mixed --expert-recipe path.json` selects Q4/Q6/Q8 by decoder
layer. Recipes contain `schemaVersion: 1`, `groupSize` (32 or 64), and
`layerBits`, with one precision per layer. They are validated before checkpoint
loading. Group-64 Q8 layers keep custom Metal verification; Q4/Q6 use MLX.
`--interleave-experts` alternates the candidate with a separately resident Q8
Metal control, preserving generation settings and reporting both models' memory.
Use `--experts-first` to reverse starting order. Weight changes require held-out
recognition testing; exact agreement on a few fixtures is insufficient.

`q6-matrix` caches a second encoding of the same Q6 weight codes for the Q8
matrix backend during prefill. Verification retains compact Q6 storage. This
trades higher resident memory for accelerated prefill; it does not quantize the
weights a second time. The cache refreshes after packed-weight replacement.
Matrix arithmetic differs, so recognition and token comparisons remain required.

Residency controls are separate from expert precision. `--wire-models` requests
a bounded Metal residency budget during the benchmark. `--interleave-wired`
pairs default and explicitly wired runs using one model; `--wired-first` reverses
starting order. The managed ticket restores the previous limit after success or
failure. Its transitions are excluded from resident timings. JSON records the
requested budget, bounded by the GPU's recommended working set. Ordinary SOV
transcription keeps its existing policy.

The [calibrated expert and residency qualification](../qualification/sov-transcribe-mlx-lowbit-2026-11-14.md)
records 140 multilingual evaluation clips, precision and memory tradeoffs,
repeated end-to-end comparisons, and the native budget-application check.
These experiments did not establish a faster replacement for the Q8 Metal path.

The [decoder qualification](../qualification/sov-transcribe-decoder-2026-11-13.md)
records the expert ablation, exact token checks, and the file-backed loader's
memory reduction. Q8 group 64 improves both recorded resident latencies and
uses less prepared model memory. Preparation still loads the original BF16
weights, so use the recorded peak process memory when sizing a machine.
Combining Q8 experts with the custom int8 encoder changed the long fixture's
transcript; that combination is not qualified for exact parity.

`--experimental-decoder q8-metal` uses the same Q8 group-64 weights with a
custom Metal vector kernel for small expert batches. Two adjacent routes to
the same expert share their weight loads. Larger prefill batches continue to
use MLX. This policy requires an Apple GPU and remains experimental.

For a paired comparison on one resident set of Q8 weights, use
`model benchmark sov ... --experimental-decoder q8 --interleave-metal --json`.
Warmups and repetitions apply to each kernel. The command alternates execution
order; `--metal-first` reverses the starting order. JSON records a kernel label
and synchronized resident time for every measured inference. Kernel switching
does not repack weights and is outside the timed interval.
Add `--metal-baseline byte` to compare the optimized eight-column, route-first
layout with the prior four-column byte-unpack layout. `--metal-baseline reuse`
also restores the original shift-based unpacking. All variants use the same
packed buffers and FP32 reduction order.
Add `--experimental-expert-fusion --metal-baseline layout` to pair the fused
gate/up projection and activation against the current optimized layout. This
experiment preserves BF16 rounding at every native activation stage and remains
disabled by default. It requires Q8 Metal dispatch; unsupported batches retain
the existing projection and activation operations.
The [exact-fusion qualification](../qualification/sov-transcribe-expert-fusion-2026-11-14.md)
records bitwise array parity and 112 matching native outputs, with no reliable
resident improvement over the current layout.

Q8 Metal inference automatically selects the distributed expert reduction on
the qualified Apple M5 Max GPU (`applegpu_g17s`); no extra reduction flag is
needed. Other GPU names or architectures, CPU execution, and unsupported
projection widths retain the existing path. Expert precision stays as selected.
The benchmark's `--experimental-expert-reduction` flag remains a force override
for research. Pair it with `--interleave-metal --metal-baseline layout` to
compare against the original SIMD reduction. JSON measurement labels include
`+expert-reduction` whenever that policy is selected, including automatic use.
The [automatic-selection qualification](../qualification/sov-transcribe-automatic-reduction-2026-11-15.md)
records device matching, fallbacks, and no-flag native parity checks.
The [distributed-reduction qualification](../qualification/sov-transcribe-distributed-reduction-2026-11-14.md)
records modest resident gains on the short and wrapped fixtures, an inconclusive
long-fixture resident change, and 160 matching outputs. On the long fixture,
the candidate reduces resident latency by 7.05% against local MLX Q8 dispatch;
this is a separate comparison from the current custom kernel.
The [expert-layout qualification](../qualification/sov-transcribe-expert-layout-2026-11-14.md)
records the latest paired results and the vocabulary experiments rejected by
the exact-parity and performance screens.
The [shader-profile qualification](../qualification/sov-transcribe-shader-profile-2026-11-14.md)
records a small resident improvement on the long recordings and no reliable
change on the short recording.
See the [Metal MoE qualification](../qualification/sov-transcribe-metal-moe-2026-11-13.md)
for staged profiles, exact hidden-state checks, and end-to-end results.

`model benchmark sov --experimental-attention` shares identical position masks
within one target forward call and uses a direct full-width RoPE call for
proportional partial rotation. Packed weights, head normalization and attention
arithmetic are unchanged. The option is independent of expert selection and
remains off by default.

Use `--experimental-decoder q8-metal --interleave-attention --json` to compare
reference and optimized attention on the same resident target. Warmups and
repetitions apply per mode; `--attention-first` reverses the starting order.
JSON labels optimized measurements with `+attention`. The attention and expert
interleaving options are mutually exclusive, so each comparison changes one
policy. See the [attention qualification](../qualification/sov-transcribe-attention-2026-11-14.md).

`model benchmark sov --experimental-head-norm` enables a fused BF16 Q/K/V
head-normalization and RoPE kernel in the MLX target decoder. Each normalization
and learned scale retains its intermediate BF16 rounding. Supported 256- and
512-wide heads use the kernel for at most 16 query tokens; prefill and unsupported
dtypes or shapes retain their existing path. No weights are repacked by this
option. Current qualification found no meaningful end-to-end benefit; this
remains a diagnostic option.

With `q8-metal` and `--experimental-attention` selected, use
`--interleave-head-norm --json` to compare fused heads on one resident model. `--head-norm-first` reverses starting order.
Warmups and repetitions apply per mode; optimized JSON labels end in `+heads`.
Only one interleaving option can be active. See the
[MLX head-fusion qualification](../qualification/sov-transcribe-mlx-heads-2026-11-14.md).

### Generation experiments

`model benchmark sov --generation-experiment pipeline` keeps DFlash proposal
IDs on the GPU, starts drafting before constructing target verification, and
reads proposals and predictions together. It reuses the verified next token
and defers intermediate cache evaluations. Target and draft weights are unchanged.
Ordinary transcription uses this pipeline automatically when DFlash is selected.
The benchmark defaults to `reference` scheduling so comparisons remain explicit.

The other explicit experiments are `draft-q8` (a separate group-64 Q8 vocabulary
head used only for drafting), `shared-cache` (retained sliding KV state shares
the attention concatenation), and `retained-allocator` (retain MLX's allocator
cache across requests). `combined` selects all four. The allocator experiment
uses MLX's configured cache limit and reports cache bytes and peak allocation;
it does not change that process-wide limit. These three experiments remain opt-in.

For a paired comparison, select `pipeline`, `draft-q8`, or `shared-cache` and add
`--interleave-generation --json`. Warmups and repetitions apply per mode;
`--generation-first` reverses starting order. Other interleaving modes and
`--projected-output` cannot be combined with this comparison. Allocator retention
requires separate resident runs because alternating with a cache-clearing policy
changes the following run's starting state.

JSON `acceptedLengths` records raw matching draft prefixes per round, including
any suffix after EOS. `emittedDraftTokens` counts only the draft tokens included
in the transcript. Quantized drafting can change acceptance and verification
shapes; frozen-token parity still requires validation.

See the [generation scheduling qualification](../qualification/sov-transcribe-mlx-generation-2026-11-14.md)
for paired BF16 and Q8 results, memory costs, and the allocator experiment's limits.

Additional research policies compare against the production pipeline with
`--generation-baseline pipeline --interleave-generation`:

| Policy | Change |
| --- | --- |
| `rollback` | Reserve linear KV storage and commit speculative rows by valid length. |
| `compiled` | Compile each layer's complete MoE segment after attention; weights remain runtime inputs. |
| `block-4`, `block-8`, `block-12` | Limit verification length, including the anchor. |
| `adaptive` | Truncate proposals at the first confidence below 0.5, keeping at least one. |
| `adaptive-after-first` | Use confidence truncation after the first verification round. |
| `vocabulary-32k` | Draft through a restricted vocabulary head; retain the target's full head. |

For a calibrated vocabulary, pass `--draft-vocabulary vocabulary.json`, a JSON
array of at most 32,768 distinct global token IDs. Predictions from the compact
head map back to those IDs. Without this file, the experiment uses the first
32,768 vocabulary rows. `calibrate_sov_vocabulary.py` in `scripts/model-conversion/`
selects token rows from FLEURS training text across the seven supported languages.
Calibration is separate from measured inference and recognition qualification.
JSON `proposedLengths` records the verified proposal count per round.

The [cache, draft, and prefill qualification](../qualification/sov-transcribe-mlx-remaining-2026-11-14.md)
compares these policies with the existing pipeline, including calibrated
vocabulary selection, all-layer routing, an M5 matrix-kernel screen, and trace
limitations.

These policies remain experiments. Confidence truncation after the first block
fails token parity with the pipeline on the 93-second qualification recording;
changing verification shapes can change floating-point dispatch and token
selection even though the target verifies each proposal.

### Decode diagnostics

For a DFlash run, `model benchmark sov --profile-decode --json` adds
`decodeRounds` to each measured transcription. Each round records host graph
construction, submission, prediction materialization, commit, and elapsed time.
The same phases emit signposts under subsystem `run.mere.sov`, category `decode`.
These timings observe existing evaluation boundaries; they do not add GPU
barriers. Warmups run without these diagnostics.

To capture one measured round with Metal, launch with `MTL_CAPTURE_ENABLED=1`
and pass `--metal-capture /path/to/round.gputrace`. `--capture-round 3` selects
the third DFlash round by default. The destination must be new. Capture applies
only to the first measured transcription, and the command fails if that round
is never reached. Use an external SSD with enough space for the model's buffers.
Capturing can substantially increase latency and disk use, so use a separate
run without capture or profiling flags for performance comparisons. Decode
diagnostics cannot be combined with paired benchmark modes or projected output.

## Projected-padding experiment

The pinned upstream processor inserts 375 decoder audio tokens per padded
30-second window. `model benchmark sov --audio-token-margin 25` retains all
complete windows and slices only the final projected tail, keeping enough
frames to cover the real waveform plus 25 projected frames. The encoder and
projector still process the complete padded recording. This changes decoder
input and token positions, so recognition quality requires separate validation.
Ordinary transcription retains full padding.

Use `--interleave-audio-tokens` to compare full padding and trimming on one
resident model. `--audio-tokens-first` reverses the initial order; subsequent
cycles alternate. JSON includes `fullAudioTokens`, `audioTokens`, `promptTokens`,
acceptance lengths, prefill, generation, and end-to-end timings. A margin of
375 retains all tokens and provides a no-op control. Allowed margins are 0–375.

```bash
mere.run model benchmark sov ./recording.wav \
  --model-root /path/to/sov-checkpoint --draft-model /path/to/sov-dflash \
  --experimental-decoder q8-metal --packed-weights /Volumes/SSD/MereRunModels/sov-q8-g64 \
  --experimental-attention --generation-experiment rollback \
  --audio-token-margin 25 --interleave-audio-tokens \
  --warmups 1 --repetitions 4 --json
```

Projected tensor exports retain full `audio_embeddings` and also store
`decoder_audio_embeddings`, the prefix supplied to the decoder.
Exports and interleaved token-margin measurements run separately.

The tested trim settings did not qualify for production: aggressive margins
caused omissions and language changes, while the most conservative timing
screen did not show a worthwhile gain. See the
[projected-padding qualification](../qualification/sov-transcribe-padding-2026-11-14.md)
for paired timings, multilingual quality, and exact projected-tensor checks.
