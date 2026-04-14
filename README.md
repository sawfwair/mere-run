# mere.run

mere.run is a Swift package and CLI for local-first inference on Apple Silicon. This OSS repo contains the core inference libraries plus the public `mere.run` executable for image, text, speech, vision, music, video, model management, and local API serving.

## What works today

The public OSS repo currently supports:

- local image generation and deterministic image validation
- local text chat, code generation, and embeddings
- local speech synthesis, transcription, and voice-profile management
- local vision captioning, inspection, segmentation, tracking, and OCR
- native music and video generation
- managed model installs into a shared local model store
- a local API surface for supported engines

## What’s here

- `Sources/MereRunCore`: shared model resolution, manifests, generation primitives, and MLX-backed inference code
- `Sources/AudioCore`, `Sources/AudioCodecs`, `Sources/AudioSTT`, `Sources/AudioTTS`: audio generation and transcription support
- `Sources`: core libraries plus the CLI target that builds the public `mere.run` executable
- `Tests`: SwiftPM test coverage for the core and CLI surfaces
- `vendor/llama.xcframework`: vendored `llama.cpp` runtime used by `mere.run text code` and `mere.run api serve`

## Build

```bash
swift build
swift test
swift run mere.run --help
```

## Quick start

```bash
# See the public command tree
swift run mere.run --help

# List managed models
swift run mere.run model list

# Configure a model source, then pull a managed model
export MERERUN_MODEL_SOURCE_BASE_URL=https://your-host.example/models/
swift run mere.run model pull image-zimage-max

# Generate an image
swift run mere.run image generate \
  --prompt "a ceramic coffee mug in soft morning light" \
  --output ./mug.png

# Run local chat
swift run mere.run text chat \
  --prompt "Summarize diffusion models in one paragraph."

# Generate speech
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --output ./hello.wav

# Inspect an image
swift run mere.run vision inspect ./image.png "Describe this image."

# Ground objects in an image
swift run mere.run model pull vision-ground-falcon-perception
swift run mere.run vision ground ./image.png --query "a person"

# Segment an image
swift run mere.run model pull vision-segment-sam31
swift run mere.run vision segment ./image.png --prompt "a person"

# Track prompted objects through a video
swift run mere.run vision track ./clip.mp4 --prompt "a person"

# Record a short camera session and track it
swift run mere.run vision track-live --output ./live.mp4 --prompt "a person"

# Generate music
swift run mere.run music generate \
  "upbeat electronic groove" \
  --output ./track.wav

# Generate video
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --variant unified-av \
  --model-root ~/Library/Application\\ Support/MereRun/models/video-ltx-av \
  --output ./clip.mp4
```

## Command tree

The public CLI is modality-first:

- `mere.run image generate`
- `mere.run image validate`
- `mere.run text chat`
- `mere.run text code`
- `mere.run text embed`
- `mere.run speech synthesize`
- `mere.run speech transcribe`
- `mere.run speech profile { list, create, delete }`
- `mere.run vision caption`
- `mere.run vision inspect`
- `mere.run vision ground`
- `mere.run vision segment`
- `mere.run vision track`
- `mere.run vision track-live`
- `mere.run vision ocr`
- `mere.run music generate`
- `mere.run video generate`
- `mere.run video export-latents`
- `mere.run model { list, info, pull, remove, repair-manifests }`
- `mere.run api serve`

## Vision notes

- `vision-segment-sam31` is the single managed SAM 3.1 package for segmentation and tracking
- `mere.run vision segment` supports text prompts plus box and point prompting
- `mere.run vision track` seeds objects on the init frame, then propagates them through later frames
- `mere.run vision track-live` currently records a camera clip first and then runs tracking over that recording

## Model store

By default, mere.run uses:

```text
~/Library/Application Support/MereRun/models
```

Override that with:

```bash
export MERERUN_MODELS_DIR=/path/to/models
swift run mere.run --models-root /path/to/models model list
```

If you already have a pre-rename model store from an older mere.run CLI build, migrate it once:

```bash
./scripts/migrate_model_store.sh
```

That script renames legacy model directories to the canonical OSS names used by this repo.

## Model source configuration

Managed model downloads are explicit in the OSS repo. `mere.run model pull` does not fall back to a committed hosted archive URL.

Configure one of these before pulling models:

- `MERERUN_MODEL_SOURCE_BASE_URL=https://your-host.example/models/` for unsigned public archives
- `MERERUN_R2_SIGNED_URL_ENDPOINT` for a signed download service
- direct R2 credentials with `MERERUN_R2_ACCOUNT_ID`, `MERERUN_R2_ACCESS_KEY_ID`, and `MERERUN_R2_SECRET_ACCESS_KEY`

If you only want to run against local paths, you do not need any model-source configuration.

## Validation

```bash
./scripts/check.sh
```

Optional real-world smoke runs:

```bash
MERERUN_RUN_E2E=core ./scripts/check.sh
./scripts/migrate_model_store.sh
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

## Docs

Start with the docs home:

- [`docs/README.md`](./docs/README.md): navigation hub for the full docs set
- local docs site: `pnpm install && pnpm docs:dev`
- production docs build: `pnpm docs:build`
- GitHub Pages deploy: [`.github/workflows/docs.yml`](./.github/workflows/docs.yml)

Core guides:

- [`docs/getting-started.md`](./docs/getting-started.md): build, first commands, first models
- [`docs/cli.md`](./docs/cli.md): full CLI guide and command reference
- [`docs/repository-tour.md`](./docs/repository-tour.md): top-level layout and module ownership
- [`docs/development-workflow.md`](./docs/development-workflow.md): how to work in the repo day to day
- [`docs/testing.md`](./docs/testing.md): validation layers, smoke runs, and troubleshooting
- [`docs/runtime/vision.md`](./docs/runtime/vision.md): native SAM 3.1 segmentation and tracking details

Configuration and model management:

- [`docs/configuration.md`](./docs/configuration.md): runtime environment variables and supported debug toggles
- [`docs/model-sources.md`](./docs/model-sources.md): managed model IDs, explicit archive configuration, and model-store behavior
- [`docs/runtime/model-management.md`](./docs/runtime/model-management.md): model store, manifests, and model commands
- [`docs/migration.md`](./docs/migration.md): hard-cut rename map from the older CLI/model vocabulary

Implementation reading guides:

- [`docs/architecture.md`](./docs/architecture.md): contributor reading order for the runtime families
- [`docs/internals/cli-and-runtime.md`](./docs/internals/cli-and-runtime.md): how the CLI maps onto the runtime
- [`docs/internals/source-layout.md`](./docs/internals/source-layout.md): source tree reference

## Repository layout

```text
mere-run/
  Package.swift
  Sources/
    ...
  Tests/
    ...
  scripts/
  docs/
  vendor/
```
