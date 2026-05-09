<p align="center">
  <a href="https://mere.run"><img src="https://mere.run/showcase/banner.png" alt="mere.run — Create anything. Locally." /></a>
</p>

<p align="center">
  <a href="https://mere.run">mere.run</a> ·
  <a href="https://sawfwair.github.io/mere-run/">Docs</a> ·
  <a href="https://public.stereovoid.com/mere-run-releases/mere-run.dmg">Download DMG</a> ·
  <code>swift build</code>
</p>

# mere.run

mere.run is a Swift package, CLI, and optional macOS studio for local-first inference on Apple Silicon. This OSS repo contains the core inference libraries, the public `mere.run` executable, and a SwiftUI app that keeps the user-facing experience prompt-first while still running the public CLI underneath.

## What works today

The public OSS repo currently supports:

- local image generation and deterministic image validation
- local text chat, code generation, embeddings, and PII anonymization
- local speech synthesis, transcription, and voice-profile management
- local vision captioning, inspection, segmentation, tracking, and OCR
- native music and video generation
- managed model installs into a shared local model store
- a local API surface for supported engines
- an optional macOS studio that wraps the public CLI instead of reimplementing runtime logic

## What’s included in this repo

- `Sources/MereRunCore`: shared model resolution, manifests, generation primitives, and MLX-backed inference code
- `Sources/AudioCore`, `Sources/AudioCodecs`, `Sources/AudioSTT`, `Sources/AudioTTS`: audio generation and transcription support
- `Sources/MereRunCLI`: the target that builds the public `mere.run` executable
- `Sources/MereRunApp`: a SwiftUI `mere.run.app` target with a user-facing studio and advanced CLI details
- `Tests`: SwiftPM test coverage for the core and CLI surfaces
- `vendor/llama.xcframework`: vendored `llama.cpp` runtime used by `mere.run text code` and `mere.run api serve`
- `vendor/mlx-swift_Cmlx.bundle`: vendored Metal shader resources needed by MLX-backed runtime paths
- [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md): redistribution, provenance, and license notes for bundled third-party artifacts

## Platform expectations

- the public CLI and local runtime are developed and validated on Apple Silicon macOS
- `swift build` and `swift test` are the supported first-run validation path for contributors
- some vendored binaries include additional Apple platform slices for package consumers, but the public quickstart is macOS-first

## Build

Install SwiftLint and ripgrep once if you plan to run the contributor
validation script:

```bash
brew install swiftlint ripgrep
```

Install Node.js and pnpm if you plan to run the docs site locally, and Gitleaks
if you want to mirror the repository security scan before publishing changes:

```bash
brew install node pnpm gitleaks
```

```bash
swift build
swift test
swift run mere.run --help
app_path="$(./scripts/build_mere_run_app.sh debug)"
open "$app_path"
```

## Quick start

```bash
# See the public command tree
swift run mere.run --help

# Launch the optional macOS studio
app_path="$(./scripts/build_mere_run_app.sh debug)"
open "$app_path"

# List managed models
swift run mere.run model list

# See what this Mac can run before pulling large models
swift run mere.run model capabilities

# Choose guided, bring-your-own-agent, or manual setup
swift run mere.run setup

# Pull a managed model into the local model store
swift run mere.run model pull image-zimage-nano

# Generate an image
swift run mere.run image generate \
  --prompt "a ceramic coffee mug in soft morning light" \
  --output ./mug.png

# Run local chat
swift run mere.run text chat \
  --stream \
  --prompt "Summarize diffusion models in one paragraph."

# Serve the OpenAI-compatible local API on loopback
swift run mere.run api serve --engine text-chat-gemma4

# Expose the API beyond loopback only with an explicit key
export MERERUN_API_KEY=change-me
swift run mere.run api serve \
  --host 0.0.0.0 \
  --port 11434 \
  --api-key "$MERERUN_API_KEY" \
  --rate-limit-per-minute 120

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
- `mere.run text anonymize`
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
- `mere.run model { list, capabilities, info, pull, remove, repair-manifests }`
- `mere.run api serve`
- `mere.run setup`
- `mere.run agent { onboard, install-pi, start }`

The optional `mere.run.app` product opens to a user-facing studio for the same
command families and keeps raw command previews, logs, runtime paths, model
management, API serving, and arbitrary arguments in Advanced details.

## Vision notes

- `vision-segment-sam31` is the single managed SAM 3.1 package for segmentation and tracking
- `mere.run vision segment` supports text prompts plus box and point prompting
- `mere.run vision track` seeds objects on the init frame, then propagates them through later frames
- `mere.run vision track-live` currently records a camera clip first, searches a short warm-up window for seed objects, and then runs tracking over that recording

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

### Hugging Face snapshot cache

A second store holds models that resolve through the native Hugging Face snapshot path (e.g. `text-chat-gemma4`). It is a mere.run-local cache, not the default `~/.cache/huggingface/hub` — the resolution chain is:

1. `MERERUN_HUB_CACHE` (explicit override)
2. `MERERUN_MODEL_CACHE_HOME/hub` (shared cache root)
3. `~/Library/Application Support/MereRun/hub` (default)
4. `~/Library/Caches/MereRun/hub`, then a temp dir as last-resort fallbacks

If you already pull from Hugging Face elsewhere and want to share cached weights, point `MERERUN_HUB_CACHE` at your existing `huggingface/hub` directory.

## Security defaults

The public OSS build keeps local-first behavior by default and requires explicit opt-in for higher-risk modes:

- `mere.run api serve` can bind to loopback without auth, but non-loopback hosts require `--api-key` or `MERERUN_API_KEY`
- the OpenAI-compatible chat route requires `Content-Type: application/json`, supports `--rate-limit-per-minute` for basic abuse control, and rejects out-of-range generation parameters
- API LoRA adapters are operator-controlled with `--lora`; per-request LoRA paths are rejected
- tool-loop execution in `mere.run text chat` requires interactive approval unless `--auto-approve-tools` is passed for non-shell tools
- `shell_exec` is disabled unless `--allow-shell-exec` is set, and still requires interactive approval when enabled
- `write_file` stays inside the sandbox unless `--allow-absolute-tool-paths` is set
- remote model and LoRA downloads reject plaintext HTTP except for loopback and local-dev cases

These flags are intentionally explicit because they weaken the default safety posture:

- `--allow-shell-exec`
- `--allow-absolute-tool-paths`
- `--auto-approve-tools`
- non-loopback `api serve` binds

## Validation

```bash
./scripts/check.sh
```

Optional real-world smoke runs:

```bash
MERERUN_RUN_E2E=core ./scripts/check.sh
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

## Docs

Start with the docs home:

- Published docs: [`sawfwair.github.io/mere-run`](https://sawfwair.github.io/mere-run/)
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
- [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md): vendored artifact provenance and license notices
- [`CHANGELOG.md`](./CHANGELOG.md): public release notes and OSS-facing changes

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

## Acknowledgements

mere.run exists because the Python MLX community proved that local-first inference on Apple Silicon could feel fast, practical, and joyful. This Swift package is not a replacement for that work; it is a port of those ideas into a public Swift runtime and CLI. The shape of mere.run — what to expose, how to manage models, how to keep inference paths Metal-native — was directly informed by these projects:

- [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) — language model inference on MLX; the reference for `mere.run text` engine surfaces and chat / code paths.
- [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) — vision-language models on MLX; informed `mere.run vision` captioning, OCR, and inspection commands.
- [`Blaizzy/mlx-audio`](https://github.com/Blaizzy/mlx-audio) — TTS / STT / audio codecs on MLX; shaped `mere.run speech` (synthesis, transcription, voice profiles).
- [`filipstrand/mflux`](https://github.com/filipstrand/mflux) — FLUX image generation on MLX; shaped how `mere.run image` exposes engines and validates output.

Where these projects ship runtime artifacts that mere.run actually links against, attribution and license terms live in [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md). The credits above are for the architectural debt: the design conversations, reference implementations, and hard-won model bring-up work these repos held in public before mere.run wrote its first line of Swift.
