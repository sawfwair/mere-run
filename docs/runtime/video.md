# Video Runtime

This page covers the native video-generation path exposed through `mere.run video`.

## Public surface

- `mere.run video generate`
- `mere.run video export-latents`

## Model family

- `video-ltx-av`

## Typical workflows

### Generate video

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant unified-av \
  --model-root ~/Library/Application\ Support/MereRun/models/video-ltx-av \
  --output ./clip.mp4
```

### Export latents

```bash
swift run mere.run video export-latents \
  --prompt "storm clouds over the ocean" \
  --output ./latents.npz
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VideoGenerateCommand.swift`
- `Sources/MereRunCLI/Commands/VideoExportLatentsCommand.swift`

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
