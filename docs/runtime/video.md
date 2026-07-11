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

Long or high-resolution LTX decodes automatically split the VAE work into
overlapping temporal/spatial tiles. Tile pixels and trapezoidal weights remain
in MLX for device-side accumulation, normalization, and UInt8 conversion; the
runtime no longer reads every float tile back to Swift. Override the automatic
decode budget with `MERERUN_VIDEO_LTX_VAE_DECODE_BUDGET_GB` when validating a
specific memory envelope.

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
