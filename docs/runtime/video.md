# Video Runtime

This page covers the native video-generation path exposed through `mere.run video`.

## Public surface

- `mere.run video generate`
- `mere.run video export-latents`

## Model family

- `video-ltx23-av-mlx`: **default.** LTX 2.3 MLX split checkpoint — recommended for
  both the distilled draft lane and the high-quality `--variant unified-av` lane.
- `video-ltx-av`: legacy merged LTX root, superseded by LTX 2.3. Only still required
  by `video export-latents`; not recommended for `video generate`.
- `video-lingbot-dense-1.3b`: LingBot-Video Dense 1.3B Diffusers checkpoint.
  Runs text-to-video through the native Swift/MLX LingBot pipeline.
- `video-lingbot-moe-30b-a3b`: source LingBot-Video 30B-A3B MoE Diffusers
  checkpoint with refiner; pull it before native conversion.
- `video-lingbot-moe-30b-a3b-4bit`: converted native MLX MoE model. Runs the
  base transformer directly and the released second-stage refiner on demand.

## Typical workflows

### Fast visual draft

The default `distilled` lane is the speed path. It generates video-only MP4s
and is the right first pass for prompt, camera, subject, and composition checks.

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant distilled \
  --model video-ltx23-av-mlx \
  --num-frames 65 \
  --output ./clip.mp4
```

### Directed image-to-video

Use `--image` to anchor the first latent frame. Add `--end-image` when the clip
should move toward a specific final keyframe.

```bash
swift run mere.run video generate \
  "a car drives from a bright morning street into a warm sunset road, smooth forward motion" \
  --variant distilled \
  --model video-ltx23-av-mlx \
  --image ./car-start.png \
  --image-strength 0.9 \
  --end-image ./car-end.png \
  --end-image-strength 0.85 \
  --num-frames 65 \
  --output ./car-start-to-end.mp4
```

`--end-image` requires `--image`; the start conditioning wins if very short
clips make the start and end latent conditioning windows overlap.

### Synchronized AV quality render

For LTX 2.3 audio/video, pull the managed model id and let it install its Gemma
3 companion:

```bash
swift run mere.run model pull video-ltx23-av-mlx
swift run mere.run video generate \
  "dialogue with clean background music" \
  --variant unified-av \
  --model video-ltx23-av-mlx \
  --duration 15 \
  --fps 24 \
  --output ./ltx23.mp4
```

Use `--duration` rather than hand-pairing `--num-frames` and `--fps` for
representative unified AV tests. LTX 2.3 expects 24 fps timing; for example,
15 seconds resolves to 361 frames at 24 fps because LTX frame counts must
satisfy `8n+1`.

`video-ltx23-av-mlx --variant unified-av` is the default and the current quality path
when dialogue, score, and SFX matter. The older `video-ltx-av` merged root is legacy
(retained only for `video export-latents`).

### Export latents

```bash
swift run mere.run video export-latents \
  --prompt "storm clouds over the ocean" \
  --output ./latents.npz
```

`video export-latents` still targets the distilled video-only latent path.

### LingBot-Video Dense and MoE

Pull the Dense checkpoint, then start with the upstream smoke resolution and
frame count before increasing either:

```bash
swift run mere.run model pull video-lingbot-dense-1.3b
swift run mere.run video generate \
  "a dexterous robot folds a blue towel on a workbench" \
  --model video-lingbot-dense-1.3b \
  --width 320 \
  --height 192 \
  --num-frames 9 \
  --steps 40 \
  --guidance-scale 3 \
  --shift 3 \
  --output ./lingbot-smoke.mp4

swift run mere.run model pull video-lingbot-moe-30b-a3b --allow-unsupported
swift run mere.run model quantize video-lingbot-moe-30b-a3b
swift run mere.run video generate \
  "a dexterous robot folds a blue towel on a workbench" \
  --model video-lingbot-moe-30b-a3b-4bit \
  --width 448 \
  --height 256 \
  --num-frames 9 \
  --steps 40 \
  --refiner \
  --refiner-width 960 \
  --refiner-height 544 \
  --output ./lingbot-moe-refined.mp4
```

Selecting `video-lingbot-dense-1.3b` chooses the LingBot pipeline automatically;
use `--variant lingbot` only with an explicit local model root. LingBot sizes
are snapped to multiples of 16 and frame counts to `4n+1`. The first native
release is text-to-video only; `--image` and `--end-image` are rejected for
LingBot rather than being routed through LTX.

LingBot follows the released runner's validation boundary: dimensions are
multiples of 16 and video frame counts are `4n+1`. It does not impose a separate
spatial-to-temporal ratio. Low-resolution long clips can still be poor model
operating points, so use `--temporal-probe` before a costly full run.

`--temporal-probe` runs the normal trajectory only through
`--temporal-probe-step` (default 4), decodes the predicted clean sample, scores
adjacent-frame luma deltas, writes the probe MP4, and stops before the refiner.
Preflight reports the resolved video-token count and emits a large global-
attention warning before model loading. The released 832x480x121 base shape has
48,360 video tokens before text conditioning. LingBot uses global attention, so
attention work grows roughly with the square of this count; CFG above 1 also
runs two transformer branches per denoising step.

The native transformer uses MLX fused attention, normalization, and projection
paths. `--batch-cfg` runs positive and negative conditioning in one masked batch,
matching the upstream option, but it does not reduce arithmetic and can increase
peak memory. Keep serial CFG unless a benchmark on the target machine shows a
gain. `--refiner-batch-cfg` provides the same opt-in for the refiner. The
upstream CUDA runner can instead distribute context and CFG branches across
multiple GPUs; a single Apple GPU cannot reproduce that distributed throughput.
For released-shape parity, provide the structured caption JSON and the
832x480 base dimensions; a 5-second caption resolves to 121 frames:

```bash
swift run mere.run video generate \
  --prompt-json ./prompt.json \
  --negative-prompt-json ./negative.json \
  --model video-lingbot-dense-1.3b \
  --width 832 \
  --height 480 \
  --temporal-probe \
  --temporal-probe-step 1 \
  --output ./lingbot-reference-probe.mp4
```

`model quantize` processes one transformer shard at a time and resumes shards
that already contain the complete output key set. It converts only the routed
expert matrices to 4-bit affine MLX weights; attention, routers, shared experts,
the text encoder, and VAE remain in their released precision. By default the
refiner shards are converted and retained in the output root. Generation stays
base-only unless `--refiner` is passed. The refiner defaults match the release:
8 requested steps, guidance 3, shift 3, threshold 0.85, two low-noise tail
sigmas, and a 1920x1088 output size. Override these with the `--refiner-*`
options.
Use `model quantize --skip-refiner` with an explicit output only when the
smaller base-only artifact is intentional; the canonical 4-bit model ID always
represents a complete transformer-plus-refiner conversion.

The upstream quality workflow recommends structured JSON captions, its separate
Qwen3.6-27B rewriter LoRA, and Auto Negative. The native runtime owns the Qwen3-VL
conditioning template and default universal negative prompt, but it does not
serve that separate rewriter model. Pass its artifacts through `--prompt-json`
and `--negative-prompt-json`; JSON member order and compact serialization are
preserved to match the Python runner's conditioning tokens.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VideoCommand.swift`

### Runtime

- `Sources/MereRunCore/LTX/LTXDistilledLatentGenerator.swift`
- `Sources/MereRunCore/LTX/LTXGemmaTextEncoder.swift`
- `Sources/MereRunCore/LTX/LTXVideoMP4Writer.swift`
- `Sources/MereRunCore/LingBotVideo/LingBotVideoPipeline.swift`

## Source Reading Notes

The LTX runtime has more low-level model, media, and checkpoint-layout code
than most other runtime families in this repo. Start from the public generation
flow before reading the lower-level model definitions.

The best reading order is:

1. public generation types and `LTXDistilledLatentGenerator`
2. request normalization and generation flow
3. denoise and latent-conditioning helpers
4. decoding and media assembly code
5. lower-level model definitions

If you are new to the repo, use [Architecture Reading Map](../architecture.md)
before diving directly into the LTX implementation.
