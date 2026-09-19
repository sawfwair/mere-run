# AuK native qualification — September 18, 2026

The native base and Flash FP32 paths passed trained-weight comparisons and
**16 of 16 bounded audio checks** on Apple Silicon. The largest model-chain
waveform error against upstream MLX was **0.0155% relative L2**. Direct PyTorch
CPU component checks also passed.

The audio checks establish the specific results below. Controls are approximate;
passing this batch does not establish exact pitch/gain control or subjective
speech quality. [Machine-readable evidence](/qualification/auk-native-2026-09-18.json)
includes asset hashes, the tested binary/source hashes, numerical errors, audio
statistics, timing, and local ASR transcripts.

## Environment and sources

- Apple M4 Max, 128 GiB unified memory, macOS 26.5.2.
- Native debug build with Swift 6.3 and MLX core 0.32.1.
- Reference environment: Python 3.12.13, MLX 0.32.2, PyTorch 2.7.1,
  transformers 4.57.6. PyTorch comparisons run on CPU.
- AuK source: `Tencent-Hunyuan/AuK` at
  `6943a1e967409e8c73139a7a345f2a611cfb3dd6`.
- Base weights: `tencent/AuK` at `790742b71a4430120daf2b2099192abae449eb9f`.
- Flash weights: `tencent/AuK-Flash` at `575b92f0895f75180bf2cbd35f2e176c5732b8ed`.
- Encoder: `Qwen/Qwen2.5-Omni-3B` at `f75b40e3da2003cdd6e1829b1f420ca70797c34e`.
  The pinned index places all Thinker text and audio tensors in shards 1 and 2.
  Shard 3 is not part of the managed AuK companion download.

The native runtime reads the original checkpoint layouts. Python is used only
for independent qualification, not by the public CLI inference path. The
reference converter retains converted tensors in memory instead of creating a
second checkpoint set on disk.

## Completed checks

The native tokenizer matches the Hugging Face processor token for token for
text-only and reference-audio prompts. Mel features match within a maximum
absolute error of `3.5643578e-05`, including an audio clip ending between frames.

Qualification exposed and corrected two integration issues:

- The shared non-power-of-two Fourier transform rounded phase angles too early.
  Computing phases in double precision before rounding the basis reduced the
  maximum mel error from `0.0012059212` to `0.000035643578`.
  A synthetic high-frequency leakage test protects the correction.
- The generic model validator applied image-pipeline component requirements to
  AuK. AuK now validates its original flat checkpoint layout; regression tests
  cover base, Flash, and encoder installs with missing-file rejection.

The downloaded base transformer and VAE match their published SHA-256 hashes.
Native reference encoding, waveform decoding, and guided transformer execution
pass on Metal with the original weights. These checks alone do not establish
speech quality.

## Trained-weight parity

All checks passed with identical input tensors and initial noise. Relative L2
errors against the pinned upstream MLX implementation were:

| Comparison | Base relative L2 | Flash relative L2 |
| --- | ---: | ---: |
| Text and audio conditioning | 6.43e-7 | 6.47e-7 |
| Reference VAE encoding | 1.38e-6 | 1.38e-6 |
| Diffusion forward pass | 9.33e-7 | 1.87e-6 |
| Complete diffusion trajectories | 2.96e-6 | 8.20e-5 |
| VAE decoding of common latents | 1.91e-6 | 1.29e-6 |
| Complete native model chain | 7.42e-5 | 1.55e-4 |

The model-chain comparison starts from common token IDs, mel features, audio
samples, and initial noise. Frontend tokenization and mel extraction are checked
separately. Both variants include text-only and reference-audio paths. Base uses
32 steps with guidance 2; Flash uses its fixed four-step grid without guidance.
The predefined tolerance is 0.1% relative L2 for components and latents and 1%
for the final model-chain waveform.

Direct PyTorch CPU checks of the VAE encoder, decoder, and transformer forward
pass also passed. Their largest relative L2 errors were `2.91e-6` for base and
`4.16e-6` for Flash. These are component comparisons, not a complete PyTorch
sampling run or CUDA qualification.
Reference encoding uses the deterministic posterior mean, matching the upstream
MLX path. Default stochastic PyTorch reference encoding is not compared as an
identical-output operation.

## Audio results

All cases use seed 42. Inputs and instructions come from the pinned upstream
cookbook, with an additional Chinese text-only request. The source clips range
from 4.99 to 11 seconds. English ASR uses local Parakeet TDT 0.6B v3 BF16;
Chinese ASR uses local Qwen3-ASR 1.7B 8-bit. Their asset hashes are in the evidence.

| Check | Base | Flash |
| --- | --- | --- |
| Reference-conditioned English TTS | Target sentence; 6.25% ASR WER | Same |
| Text-only English TTS | Exact target words | Exact target words |
| Phrase replacement | New phrase present; old phrase absent | Same |
| Pitch, requested +200 cents | +290 cents | +234 cents |
| Speed, cookbook duration 6.86 s | 1.60× duration compression; 0% ASR word difference | 1.60×; 4.55% ASR word difference |
| Volume, requested +10 dB | +9.87 dB | +7.58 dB |
| Enhancement proxy | Quiet-frame RMS −62.31 dB; 7.14% ASR character difference | Same |
| Chinese TTS | Exact target characters | Exact target characters |

The reference-conditioned TTS WER reflects “honour” versus the target's “honor.”
Pitch and volume sources yielded no recognized words, so no speech-preservation
claim is made for those cases. The speed instruction asks for 1.5×, but the
upstream cookbook's duration setting compresses the current 11-second asset to
6.86 seconds; the table reports the actual ratio.

Criteria were set before running the batch: TTS word/character error at most
20%; complete replacement phrase present and old phrase absent; pitch shift
between 80 and 330 cents; duration ratio between 1.35 and 1.7 with speech content
retained within 20% ASR word error; gain between 7 and 13 dB; and quiet-frame RMS
reduction greater than 3 dB with at most 30% ASR character difference. These are
coarse, automatic checks. Enhancement has no paired clean-audio ground truth,
and quieter pauses alone do not establish perceptual quality.

## Runtime measurements

Each measurement runs the public CLI in a fresh process and includes model
loading and WAV writing. Downloads and ASR are excluded. Measurements use a
debug build on an active development machine, not an isolated release benchmark.

| Case | Output seconds | Base wall seconds | Flash wall seconds |
| --- | ---: | ---: | ---: |
| 1.1-zeroshot-tts | 6.00 | 72.50 | 13.68 |
| 1.2-instruct-tts | 1.70 | 17.67 | 3.21 |
| 2.1-content-edit | 7.00 | 86.02 | 11.19 |
| 3.1-pitch | 5.50 | 52.78 | 9.72 |
| 3.2-speed | 6.86 | 112.82 | 13.72 |
| 3.3-volume | 5.50 | 71.10 | 9.26 |
| 5.1-enhance | 5.00 | 54.96 | 9.31 |
| zh-tts | 3.00 | 19.64 | 5.46 |

Peak process footprint was **17.95 GB** for base and **18.39 GB** for Flash.
These measurements came from a 128 GiB host; they do not constitute a run on
24 GB hardware or acceptance of the maximum 300-second duration. All outputs
were finite mono 24 kHz WAV files. The base volume case had 0.034% of samples
at or above absolute amplitude 0.999; the other cases had none at that threshold.

## Final local validation

- `./scripts/check.sh` passed: 4,505 XCTest cases (340 skipped), 51 Swift Testing
  checks, strict lint, package policy, CLI help, and hygiene checks.
- `pnpm docs:build` passed.
- All 16 WAV files were read back and their hashes matched the receipts. None
  of the generation processes reported swaps.
- After updating the CLI help and experimental-status text, the final rebuilt
  CLI reproduced the saved Flash text-only output **byte for byte** with seed
  42, using managed-model resolution. Model-computation source hashes were
  unchanged from the audio batch.

## Reproduce

Install the Python reference dependencies in a separate environment. Use the
pinned upstream checkout and the model roots installed by `model pull`.
The qualification scripts do not download model weights implicitly.

```bash
.build/debug/mere.run model pull audio-auk-base
.build/debug/mere.run model pull audio-auk-flash

python scripts/qualify-auk-assets.py \
  --base "$BASE" --flash "$FLASH" --thinker "$THINKER" \
  --output "$RESULTS/assets.json"

python scripts/qualify-auk-reference.py \
  --upstream "$UPSTREAM" --model "$BASE" --thinker "$THINKER" \
  --variant base --out "$RESULTS/base-parity"

AUK_QUALIFICATION_ROOT="$RESULTS/base-parity" \
AUK_MODEL_ROOT="$BASE" AUK_THINKER_ROOT="$THINKER" \
MERERUN_TEST_MLX_DEVICE=gpu \
  swift test --filter 'AuKQualificationTests|AuKFrontendQualificationTests'

python scripts/qualify-auk-torch.py \
  --upstream "$UPSTREAM" --model "$BASE" \
  --variant base --receipts "$RESULTS/base-parity"

python scripts/qualify-auk-native.py \
  --binary "$PWD/.build/debug/mere.run" --upstream "$UPSTREAM" \
  --model "$BASE" --thinker "$THINKER" \
  --variant base --out "$RESULTS/base-audio"

python scripts/qualify-auk-audio-metrics.py \
  --upstream "$UPSTREAM" --receipts "$RESULTS/base-audio/receipts.json"
```

Repeat with `--variant flash`, `AUK_VARIANT=flash`, and the Flash model root.
Native parity uses identical initial noise and input tensors. The default audio
batch covers eight cases with seed 42 and writes WAV hashes, CLI results, local
ASR transcripts, timing, and process memory measurements. ASR uses already
installed native Parakeet and Qwen backends. It does not call a cloud service.

## Scope

This is a bounded FP32 Apple Silicon qualification. It does not establish CUDA
parity, quantized inference quality, subjective listening scores, speaker
identity preservation, or reliability across the full distribution of speech
editing tasks. Timing from a debug build is not a release-build benchmark.
