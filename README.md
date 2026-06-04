<p align="center">
  <a href="https://mere.run"><img src="https://mere.run/showcase/banner.png" alt="mere.run — Create anything. Locally." /></a>
</p>

<p align="center">
  <a href="https://mere.run">mere.run</a> ·
  <a href="https://docs.mere.run/">Docs</a> ·
  <a href="https://docs.mere.run/linux-quickstart">Linux QuickStart</a> ·
  <a href="https://mere.run/releases/mere-run.dmg">Download DMG</a> ·
  <code>swift build</code>
</p>

# mere.run

mere.run is a Swift package, CLI, and optional macOS studio for local-first inference on Apple Silicon. This OSS repo contains the core inference libraries, the public `mere.run` executable, and a SwiftUI app that keeps the user-facing experience prompt-first while still running the public CLI underneath.

## What works today

The public OSS repo currently supports:

- local image generation across Klein, ZImage, and HiDream O1 families,
  including image-to-image, HiDream reference images, LoRA input, and
  deterministic image validation
- local text chat, code generation, embeddings, and PII anonymization
- local speech synthesis, transcription, and voice-profile management
- local vision captioning, inspection, grounding, segmentation, tracking, live camera tracking, and OCR
- native ACEStep music generation and LTX video generation
- Hugging Face-backed model pulls that resolve into a shared local model store
- offline command cookbooks through `mere.run guide`
- a local OpenAI-compatible API surface with a native Swift runtime model pool
- a quick status snapshot for the local server, loaded API models, active requests, runtime capabilities, and installed model store
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

- the public CLI and local runtime are developed and validated on Apple Silicon macOS 15 or newer
- the optional SwiftUI studio, app bundle, installer, and DMG are macOS-only; Linux compatibility work is for the headless `mere.run` CLI
- `Package.swift` uses Swift tools 6.0 and declares macOS 15 / iOS 18 package platforms
- `swift build` and `swift test` are the supported first-run validation path for macOS contributors
- Linux CLI compatibility work expects a Swift 6.x toolchain, `clang`, `cmake`, `ninja`, `pkg-config`, `gfortran`, curl/zlib/OpenBLAS/LAPACK development headers, `ffmpeg`, `ffprobe`, `gzip`, `unzip`, and `zip`
- media I/O should discover `ffmpeg` and `ffprobe` on `PATH`, with `MERERUN_FFMPEG` and `MERERUN_FFPROBE` reserved for absolute executable overrides
- hosted Linux CI should stay CPU MLX-oriented and fixture-sized; Linux arm64 release packages must use a real CUDA lane
- Linux CUDA validation is limited to the exact hosts that have run the CUDA package and smoke path
- some vendored binaries include additional Apple platform slices for package consumers, while Linux release artifacts stay headless CLI-only

## Install the latest release

The signed and notarized macOS DMG is published at:

```bash
curl -L https://mere.run/releases/mere-run.dmg -o mere-run.dmg
open mere-run.dmg
```

The DMG contains `MereRun.app`, a bundled CLI payload, an optional Codex skill,
runtime assets, notices, and a terminal installer. Drag `MereRun.app` to
Applications for the studio. The app uses its bundled CLI internally; use
Settings to install the `mere.run` terminal command and optional `use-mere-run` skill
when you want them. For terminal-only installs, open the mounted DMG and run:

```bash
cd /Volumes/mere.run/.mere-run
./install.sh
```

The installer copies `mere.run` and its colocated runtime assets to
`/usr/local/bin/mere.run`, using `sudo` only when the destination requires it.

Linux release artifacts are headless CLI-only. The default release workflow
publishes portable tarballs and Debian packages for x86_64/amd64 Ubuntu-style
hosts. Linux arm64 is CUDA-only for release packaging and requires a real arm64
CUDA host or self-hosted runner; CPU arm64 packages are just local smoke-test
artifacts. See the dedicated [Linux QuickStart](./docs/linux-quickstart.md) for
the current validation boundary, CUDA notes, and first commands:

```bash
tag=v0.10.0
version="${tag#v}"

# Portable tarball
curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/mere-run-${tag}-linux-x86_64.tar.gz" -o mere-run-linux.tar.gz
tar -xzf mere-run-linux.tar.gz
cd "mere-run-${tag}-linux-x86_64"
./install.sh

# Debian package
curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/mere-run_${version}_amd64.deb" -o mere-run.deb
sudo apt install ./mere-run.deb
```

Linux packages install the `mere.run` CLI plus colocated runtime assets; they do
not include the macOS SwiftUI studio or DMG layout. On Linux arm64, if a distro
Clang shadows Swift's bundled Clang and cannot compile MLX bf16 headers, the
Linux scripts select a bf16-capable C++ driver or report the `CXX` override to
use. Linux arm64 release packages should be built with CUDA enabled on a host
with the CUDA Toolkit headers, CUDA CCCL headers, cuDNN, and NCCL installed.
CUDA `.deb` artifacts declare the linked CUDA 13 runtime/JIT packages
(`cuda-cccl-13-0`, `cuda-cudart-13-0`, `cuda-nvrtc-13-0`,
`libcublas-13-0`, `libcudnn9-cuda-13`, and `libnccl2`) by default. The
installed launcher also exports the resolved CUDA CCCL include root through
`MERERUN_CUDA_CCCL_INCLUDE_PATH` so MLX CUDA kernels can find `cuda/std/*`
during NVRTC JIT compilation:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/package-linux.sh --version 0.10.0
```

Current CUDA validation should be treated as limited to the exact hosts that
have run the CUDA package/smoke path.

## Build from source

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

For Linux CLI compatibility work, install the platform packages first and keep
the validation headless:

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
export MERERUN_FFMPEG=/usr/bin/ffmpeg
export MERERUN_FFPROBE=/usr/bin/ffprobe
./scripts/check-linux.sh
swift run mere.run --help
```

To build Linux release packages from a Linux x86_64 Swift toolchain host:

```bash
scripts/package-linux.sh --version 0.10.0
ls dist/linux/
```

On Linux arm64, use a CUDA-provisioned host:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/package-linux.sh --version 0.10.0
```

Do not use the app bundle commands on Linux. `mere.run.app`, SwiftUI studio
flows, and DMG packaging stay macOS-only. Linux release packaging is for the
headless CLI tarball and `.deb` only.

## Quick start

```bash
# See the public command tree
swift run mere.run --help

# Read packaged command cookbooks
swift run mere.run guide --list

# Launch the optional macOS studio
app_path="$(./scripts/build_mere_run_app.sh debug)"
open "$app_path"

# List known model IDs and local install status
swift run mere.run model list

# See the local server, served model, model-store path, and installed models
swift run mere.run status

# Set API runtime defaults beside the active model store
swift run mere.run model runtime set text-chat-gemma4 \
  --alias chat-default \
  --pinned \
  --ttl-seconds 3600 \
  --max-tokens 1024

# See what this Mac can run before pulling large models
swift run mere.run model capabilities
swift run mere.run model capabilities --recommended

# Choose guided, bring-your-own-agent, or manual setup
swift run mere.run setup

# Pull a Hugging Face-backed model into the local model store
swift run mere.run model pull image-zimage-nano
swift run mere.run model pull text-chat-lfm25-a1b-8bit

# Generate an image
swift run mere.run image generate \
  --prompt "a ceramic coffee mug in soft morning light" \
  --output ./mug.png

# Pull and run the native Swift Bonsai binary or ternary image model
swift run mere.run model pull image-bonsai-binary
swift run mere.run image generate \
  --model image-bonsai-binary \
  --prompt "a tiny bonsai tree in a sunlit greenhouse" \
  --output ./bonsai.png

# HiDream O1 runs natively for text-only, edit, and multi-reference generation.
swift run mere.run image generate \
  --model image-hidream-o1-dev \
  --prompt "a clean studio product photo of the subject" \
  --ref-image ./subject.png \
  --output ./subject-studio.png

# Run local chat
swift run mere.run text chat \
  --stream \
  --prompt "Summarize diffusion models in one paragraph."

# Run LiquidAI LFM2.5 through the native Swift MLX runtime
swift run mere.run text chat \
  --model text-chat-lfm25-a1b-8bit \
  --prompt "Summarize mixture-of-experts routing in one paragraph."

# Redact PII locally
swift run mere.run text anonymize \
  "My name is Alice Smith and my email is alice@example.com"

# Serve the OpenAI-compatible local API on loopback
swift run mere.run api serve --engine text-chat-gemma4
swift run mere.run api serve --engine text-chat-lfm2

# In another terminal, confirm the server and served model
swift run mere.run status

# Optional Gemma4 prefix KV reuse prototype; status reports cache and timing stats
MERERUN_GEMMA4_PREFIX_KV_CACHE=1 swift run mere.run api serve --engine text-chat-gemma4

# Optional Qwen3.6 text-only prefix KV reuse prototype; vision prompts are excluded
MERERUN_Q35_PREFIX_KV_CACHE=1 swift run mere.run api serve

# Optional decode batching; overlap requires max-active > 1
MERERUN_GEMMA4_CONTINUOUS_BATCHING=1 swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --max-active-requests 2
MERERUN_Q35_CONTINUOUS_BATCHING=1 swift run mere.run api serve \
  --max-active-requests 2

# Experimental Gemma4 packed PolarKV for memory-pressure and long-context decode testing
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --kv-quant-scheme polar \
  --kv-bits 2

# Conservative per-model runtime policy: default KV for short Gemma4 prompts,
# decode-deferred PolarKV at or above 1024 prompt tokens
swift run mere.run model runtime set text-chat-gemma4-turbo --kv-cache-mode auto

# Fixed-token real-checkpoint Gemma4 KV benchmark: default TurboQuant vs decode-deferred PolarKV
swift run mere.run model benchmark gemma4-kv \
  --model text-chat-gemma4-turbo \
  --prompt-repeat-values 32,128,220 \
  --decode-token-values 32,128 \
  --json

# Tiny synthetic VLM eval: Gemma4 12B vision chat vs existing Qwen3-VL inspect backend
swift run mere.run model benchmark vlm --json

# Existing-dataset VLM eval via lmms-eval; dry-run prints the exact command first
swift run mere.run model benchmark vlm \
  --dataset mathvista-testmini \
  --limit 16 \
  --lmms-eval-root ~/src/lmms-eval \
  --dry-run \
  --json

# Expose the API beyond loopback only with an explicit key
export MERERUN_API_KEY=change-me
swift run mere.run api serve \
  --host 0.0.0.0 \
  --port 11434 \
  --api-key "$MERERUN_API_KEY" \
  --rate-limit-per-minute 120 \
  --max-active-requests 1

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
  --model-root "$HOME/Library/Application Support/MereRun/models/video-ltx-av" \
  --output ./clip.mp4
```

## Command tree

The public CLI is modality-first:

- `mere.run guide`
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
- `mere.run model { list, capabilities, info, pull, remove, runtime, repair-manifests }`
- `mere.run status`
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

`mere.run model pull` and runtime auto-download paths use the native Hugging
Face snapshot cache before linking or resolving prepared models into the local
model store. It is a mere.run-local cache, not the default
`~/.cache/huggingface/hub` — the resolution chain is:

1. `MERERUN_HUB_CACHE` (explicit override)
2. `MERERUN_MODEL_CACHE_HOME/hub` (shared cache root)
3. `~/Library/Application Support/MereRun/hub` (default)
4. `~/Library/Caches/MereRun/hub`, then a temp dir as last-resort fallbacks

If you already pull from Hugging Face elsewhere and want to share cached weights,
point `MERERUN_HUB_CACHE` at your existing `huggingface/hub` directory.

## Security defaults

The public OSS build keeps local-first behavior by default and requires explicit opt-in for higher-risk modes:

- `mere.run api serve` can bind to loopback without auth, but non-loopback hosts require `--api-key` or `MERERUN_API_KEY`
- the OpenAI-compatible chat route requires `Content-Type: application/json`, supports `--rate-limit-per-minute` for basic abuse control, decodes the common Chat Completions request shape, and rejects unsupported high-impact fields before generation
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

- Published docs: [`docs.mere.run`](https://docs.mere.run/)
- [`docs/README.md`](./docs/README.md): navigation hub for the full docs set
- local docs site: `pnpm install && pnpm docs:dev`
- production docs build: `pnpm docs:build`
- GitHub Pages deploy: [`.github/workflows/docs.yml`](./.github/workflows/docs.yml)

Core guides:

- [`docs/getting-started.md`](./docs/getting-started.md): build, first commands, first models
- [`docs/linux-quickstart.md`](./docs/linux-quickstart.md): Linux CLI package install, first commands, and validation boundaries
- [`docs/cli.md`](./docs/cli.md): full CLI guide and command reference
- [`docs/repository-tour.md`](./docs/repository-tour.md): top-level layout and module ownership
- [`docs/development-workflow.md`](./docs/development-workflow.md): how to work in the repo day to day
- [`docs/testing.md`](./docs/testing.md): validation layers, smoke runs, and troubleshooting
- [`docs/runtime/vision.md`](./docs/runtime/vision.md): native SAM 3.1 segmentation and tracking details

Configuration and model management:

- [`docs/configuration.md`](./docs/configuration.md): runtime environment variables and supported debug toggles
- [`docs/model-sources.md`](./docs/model-sources.md): managed model IDs, Hugging Face sources, and model-store behavior
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
- [`filipstrand/mflux`](https://github.com/filipstrand/mflux) — MLX
  image-generation reference work for Z-Image and FLUX-family behavior; shaped
  how `mere.run image` loads components, schedules denoising, decodes VAE
  output, exposes engines, and validates generated images.

Where these projects ship runtime artifacts that mere.run actually links against, attribution and license terms live in [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md). The credits above are for the architectural debt: the design conversations, reference implementations, and hard-won model bring-up work these repos held in public before mere.run wrote its first line of Swift.
