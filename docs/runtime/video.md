# Video Runtime

This page covers the native video-generation path exposed through `mere.run video`.

## Public surface

- `mere.run video generate`
- `mere.run video export-latents`

## Model family

- `video-ltx-av`: default LTX root for the faster distilled lane and the legacy
  unified AV lane.
- `video-ltx23-av-mlx`: LTX 2.3 MLX split checkpoint for the high-quality
  `--variant unified-av` lane.

## Typical workflows

### Fast visual draft

The default `distilled` lane is the speed path. It generates video-only MP4s
and is the right first pass for prompt, camera, subject, and composition checks.

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant distilled \
  --model video-ltx-av \
  --num-frames 65 \
  --output ./clip.mp4
```

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

The older `video-ltx-av --variant unified-av` path still exists for compatibility,
but `video-ltx23-av-mlx --variant unified-av` is the current quality path when
dialogue, score, and SFX matter.

### Export latents

```bash
swift run mere.run video export-latents \
  --prompt "storm clouds over the ocean" \
  --output ./latents.npz
```

`video export-latents` still targets the distilled video-only latent path.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VideoGenerateCommand.swift`
- `Sources/MereRunCLI/Commands/VideoExportLatentsCommand.swift`
- `Sources/MereRunCLI/Commands/VideoCommand.swift`

### Runtime

- `Sources/MereRunCore/LTX/LTXDistilledLatentGenerator.swift`

## Important note on code shape

The LTX runtime is still the largest major runtime file in the repo. It is not
an architecture mess anymore, but it still contains more low-level video-model
logic than the other family entrypoints.

The best reading order is:

1. public generation types and `LTXDistilledLatentGenerator`
2. request normalization and generation flow
3. denoise and latent-conditioning helpers
4. decoding and media assembly code
5. lower-level model definitions

If you are new to the repo, use [Architecture Reading Map](../architecture.md)
before diving directly into the LTX implementation.
