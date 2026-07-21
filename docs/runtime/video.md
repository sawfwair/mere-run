# Video Runtime

This page covers the native video-generation path exposed through `mere.run video`.

## Public surface

- `mere.run video generate`
- `mere.run video session`
- `mere.run video export-latents`

## Model family

- `video-ltx23-av-mlx`: standalone distilled LTX 2.3 MLX checkpoint for fast
  video-only drafts.
- `video-ltx23-full-mlx`: LTX 2.3 dev checkpoint, official distilled LoRA,
  vocoder, VAEs, and x2 upscaler. This is the shared quality bundle for both
  `--variant unified-av` and source-audio A2Vid.
- `video-ltx23-a2vid-mlx`: compatibility ID for existing A2Vid installs. New
  installs should use `video-ltx23-full-mlx`.
- `video-ltx-av`: legacy merged LTX root, superseded by LTX 2.3. Only still required
  by `video export-latents`; not recommended for `video generate`.

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
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "dialogue with clean background music" \
  --variant unified-av \
  --model video-ltx23-full-mlx \
  --duration 15 \
  --fps 24 \
  --output ./ltx23.mp4
```

Use `--duration` rather than hand-pairing `--num-frames` and `--fps` for
representative unified AV tests. LTX 2.3 expects 24 fps timing; for example,
15 seconds resolves to 361 frames at 24 fps because LTX frame counts must
satisfy `8n+1`.

`video-ltx23-full-mlx --variant unified-av` runs the official two-stage quality
path: guided dev denoising for both audio and video at half resolution, followed
by x2 latent upscaling and four-step refinement after fusing the distilled LoRA.
The older `video-ltx-av` merged root is retained only for `video export-latents`.

Add `--timings` to print phase timings for native LTX 2.3 unified-AV or A2Vid
generation. `--timings-output <path>` writes the same typed report as JSON,
including model-component loading, text encoding, each denoising stage, LoRA
fusion where applicable, upsampling, video/audio decode, MP4 writing, unload,
and total wall time.

### Resident LTX 2.3 generation

`video session` amortizes checkpoint loading across serial synchronized-AV
generations on either the standalone distilled model or the full dev model. It
reads typed snake-case JSONL requests from stdin and writes one typed JSON
result or error to stdout for each input line.

```bash
printf '%s\n' \
  '{"id":"draft-1","prompt":"a fox runs across snow","output":"./draft-1.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  '{"id":"draft-2","prompt":"a fox runs across snow","output":"./draft-2.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  | swift run mere.run video session --model video-ltx23-av-mlx
```

Use `--model video-ltx23-full-mlx` for the two-stage quality lane. That session
keeps the dev transformer and official distilled LoRA resident, leaves the dev
weights unchanged for Stage 1, and activates the adapter only during Stage 2.
This avoids permanent BF16 weight fusion and permits repeat requests without
checkpoint reload or accumulated weight drift.

The first result includes checkpoint-load time. Later results set
`resident_model_reused` to `true` and report zero load time. Full-model requests
still reload the text encoder after the first request so the large transformer,
video decoder, and text encoder do not all remain live during denoising.

### Native source-audio-to-video

Supplying `--audio` selects A2Vid automatically; no LTX variant flag is needed.
The requested segment directly conditions video motion, while that same decoded
source segment becomes the output soundtrack.

```bash
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "the singer performs beneath sweeping blue spotlights" \
  --audio ./song.wav \
  --audio-start-time 42 \
  --duration 6 \
  --image ./artist.png \
  --image-strength 0.9 \
  --output ./shot.mp4
```

The resolved clip duration is the legal `8n+1` frame count divided by `--fps`.
The input must contain at least that much audio after `--audio-start-time`; the
runtime fails rather than padding or silently falling back to text-to-video plus
soundtrack layback. Mono input is duplicated to stereo without auto-gain.

Stage one runs 30 guided full/dev steps by default with frozen audio latents.
Stage two upsamples the video, fuses the official distilled LoRA one tensor pair
at a time, and runs the four upstream distilled sigmas. `--a2v-guidance-scale`,
`--video-cfg-guidance-scale`, `--audio-cfg-guidance-scale`,
`--v2a-guidance-scale`, `--a2v-steps`, and `--negative-prompt` expose the
advanced controls. The `--a2v-guidance-scale` value is also the video stream's
audio-to-video modality guidance in full unified AV.

### Export latents

```bash
swift run mere.run video export-latents \
  "storm clouds over the ocean" \
  --output ./latents.safetensors
```

`video export-latents` still targets the distilled video-only latent path.

Long or high-resolution LTX decodes automatically split the VAE work into
overlapping temporal/spatial tiles. Tile pixels and trapezoidal weights remain
in MLX for device-side accumulation, normalization, and UInt8 conversion; the
runtime no longer reads every float tile back to Swift. Override the automatic
decode budget with `MERERUN_VIDEO_LTX_VAE_DECODE_BUDGET_GB` when validating a
specific memory envelope.

Final MP4 encoding is incremental. The writer transfers one BGRA frame at a
time to FFmpeg stdin or AVAssetWriter and overlaps the next device-to-host frame
transfer with encoder work, avoiding a monolithic host frame buffer and a raw
video spool. Unified-AV audio is likewise transferred and written in aligned
chunks. The decoded MLX frame tensor still exists on device, so this removes
duplicated host-side staging rather than making the generation path
zero-memory. MP4 formats and backend selection are unchanged.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VideoCommand.swift`
- `Sources/MereRunCLI/Commands/VideoSessionCommand.swift`

### Runtime

- `Sources/MereRunCore/LTX/LTXDistilledLatentGenerator.swift`
- `Sources/MereRunCore/LTX/LTXInferenceTimings.swift`
- `Sources/MereRunCore/LTX/LTXGemmaTextEncoder.swift`
- `Sources/MereRunCore/LTX/LTXVideoMP4Writer.swift`

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
