# Video Runtime

This page covers the native video-generation path exposed through `mere.run video`.

## Public surface

- `mere.run video generate`
- `mere.run video export-latents`

## Model family

- `video-ltx23-av-mlx`: **default.** LTX 2.3 MLX split checkpoint — recommended for
  both the distilled draft lane and the high-quality `--variant unified-av` lane.
- `video-ltx23-a2vid-mlx`: LTX 2.3 full/dev checkpoint plus the official
  distilled LoRA for native source-audio-conditioned video.
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
swift run mere.run model pull video-ltx23-av-mlx --accept-model-license
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

### Native source-audio-to-video

Supplying `--audio` selects A2Vid automatically; no LTX variant flag is needed.
The requested segment directly conditions video motion, while that same decoded
source segment becomes the output soundtrack.

```bash
swift run mere.run model pull video-ltx23-a2vid-mlx --accept-model-license
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
`--video-cfg-guidance-scale`, `--a2v-steps`, and `--negative-prompt` expose the
advanced controls.

### Export latents

```bash
swift run mere.run video export-latents \
  --prompt "storm clouds over the ocean" \
  --output ./latents.npz
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

### Runtime

- `Sources/MereRunCore/LTX/LTXDistilledLatentGenerator.swift`
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
