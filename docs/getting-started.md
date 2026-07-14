# Getting Started

This guide gets a fresh clone to the point where you can build the package,
inspect the command tree, configure model sources, and run a first local
workflow.

## What this repo contains

`mere-run` is a Swift package with:

- reusable inference libraries in `Sources/MereRunCore`, `Sources/AudioCore`,
  `Sources/AudioCodecs`, `Sources/AudioSTT`, and `Sources/AudioTTS`
- a public executable product named `mere.run`
- an optional macOS SwiftUI studio named `mere.run.app` that wraps the public
  CLI
- tests and smoke harnesses for the package and CLI surfaces

It does not include hosted-service, billing, or private-deployment surfaces.

## Prerequisites

For the supported macOS developer path:

- Apple Silicon Mac
- macOS 15 or newer
- Xcode command line tools
- SwiftLint and ripgrep for `./scripts/check.sh`
  (`brew install swiftlint ripgrep`)
- enough disk space for model installs in
  `~/Library/Application Support/MereRun/models`

For Linux CLI compatibility work:

- Swift 6.x toolchain
- `clang`, `cmake`, `ninja`, `pkg-config`, `gfortran`, curl/zlib/OpenBLAS/LAPACK development headers
- `ffmpeg` and `ffprobe` for media probing and conversion
- `gzip`, `unzip`, and `zip` for portable LoRA checkpoint archives
- enough disk space for a headless model store

On Ubuntu-style runners, the system package layer is:

```bash
sudo apt-get update
sudo apt-get install -y clang cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
```

If the media tools are not on `PATH`, point the CLI at explicit binaries:

```bash
export MERERUN_FFMPEG=/opt/ffmpeg/bin/ffmpeg
export MERERUN_FFPROBE=/opt/ffmpeg/bin/ffprobe
```

## Build the package

From the repo root:

```bash
swift build
swift test
swift run mere.run --help
app_path="$(./scripts/build_mere_run_app.sh debug)"
open "$app_path"
```

That confirms the package graph, CLI product, optional app product, app bundle,
and basic command parsing are all working.

On Linux compatibility branches, keep the first pass headless and stop at the
CLI surface:

```bash
swift run mere.run --help
```

The macOS app product is intentionally outside the Linux target.

## Launch the macOS studio

The app is a user-facing local studio backed by the public CLI. It opens to a
unified canvas and prompt bar, keeps generated outputs in a local library, and
keeps command previews and logs in Advanced details. Launch it from a checkout
with:

```bash
app_path="$(./scripts/build_mere_run_app.sh debug)"
open "$app_path"
```

For contributor smoke tests, `swift run mere.run.app` still builds the executable
product, but the bundle script is the recommended local launch path for normal
macOS window behavior. The app auto-detects a bundled CLI first, then nearby
SwiftPM build products, common install locations, and finally the current
package checkout. It does not silently install the terminal command on launch;
open Settings and choose `Install CLI` or `Install Skill` when you want the
bundled command or `use-mere-run` Codex skill copied into user-visible
locations.

The studio is macOS-only. Linux users and Linux CI should exercise the CLI and
local API surfaces directly rather than trying to build or launch
`mere.run.app`.

## Build Linux package artifacts

Linux package artifacts are headless CLI-only. They install the `mere.run` CLI
plus colocated runtime assets; they do not include `mere.run.app`, SwiftUI
studio flows, or the macOS DMG layout. For a Linux-only setup path, first
commands, package checks, and CUDA validation limits, see
[Linux QuickStart](./linux-quickstart.md).

```bash
scripts/package-linux.sh --version 0.21.0
ls dist/linux/
```

CUDA packages must be built and smoke-tested on matching CUDA hardware before
being treated as supported. Linux arm64 packages are CUDA-only and should be
built on a real arm64 CUDA host with `MERERUN_LINUX_ACCEL=cuda`; CPU arm64
packages are local smoke artifacts, not useful release targets.

## Understand the command tree

The public CLI is modality-first:

- `mere.run guide ...`
- `mere.run image ...`
- `mere.run text ...`
- `mere.run speech ...`
- `mere.run vision ...`
- `mere.run music ...`
- `mere.run sfx ...`
- `mere.run video ...`
- `mere.run model ...`
- `mere.run status`
- `mere.run api ...`
- `mere.run setup ...`
- `mere.run agent ...`

For the full reference, see [CLI Reference](./cli.md).

## Choose a model store location

By default, models live in:

```text
~/Library/Application Support/MereRun/models
```

Override that for a session with either:

```bash
export MERERUN_MODELS_DIR=/path/to/models
```

or:

```bash
swift run mere.run --models-root /path/to/models model list
```

## Check local status

Use `status` whenever you want a quick snapshot of this machine's mere.run
state:

```bash
swift run mere.run status
```

It reports whether the local API server is reachable, which model the server
currently exposes through `/v1/models`, the active model store, and the managed
models installed there. Use JSON output when scripting:

```bash
swift run mere.run status --json
```

## Pull a model

Managed downloads use cataloged Hugging Face repos. No private model-source host
or R2 credentials are required.

Example:

```bash
swift run mere.run model capabilities
swift run mere.run model pull image-zimage-nano
# Optional compact FLUX.2 Klein path:
swift run mere.run model pull image-bonsai-binary
```

Set `MERERUN_HUB_CACHE` when you want the Hugging Face cache on another disk.
See [Model Sources](./model-sources.md) for the full matrix.

For guided onboarding, run:

```bash
swift run mere.run setup
```

The setup command offers a local Mere agent powered by Pi, a bring-your-own-agent
handoff prompt for Claude/Codex, or manual commands. Use
`--mode agent --agent-model small` to select the Qwen3.5 9B GGUF setup agent
explicitly. On 96 GB+ Apple Silicon Macs, the hardware-tier and premier agent
path selects DeepSeek V4 Flash as the preferred setup agent. On Linux, install
or provide Pi separately with `--pi-path` or PATH before using `--start`.

## Run a first workflow

### Image generation

```bash
swift run mere.run image generate \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png

swift run mere.run image generate \
  --model image-bonsai-binary \
  --prompt "a tiny bonsai tree in a sunlit greenhouse" \
  --output ./bonsai.png

swift run mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a cinematic product photo of a translucent portable speaker, crisp reflections" \
  --steps 8 \
  --output ./speaker.png
```

### Text chat

```bash
swift run mere.run text chat \
  --stream \
  --prompt "Explain classifier-free guidance in one paragraph."
```

### Speech synthesis

```bash
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --output ./hello.wav
```

### Vision inspect

```bash
swift run mere.run vision inspect ./image.png "Describe this image."
```

### Vision ground

```bash
swift run mere.run model pull vision-ground-falcon-perception
swift run mere.run vision ground ./image.png --query "a person"
```

### Vision segment

```bash
swift run mere.run model pull vision-segment-sam31
swift run mere.run vision segment ./image.png --prompt "a person"
swift run mere.run vision track ./clip.mp4 --prompt "a person"
swift run mere.run vision track-live --output ./live.mp4 --prompt "a person"
```

## Validate your local environment

Run the repo validation script:

```bash
./scripts/check.sh
```

Optional end-to-end smoke coverage:

```bash
MERERUN_RUN_E2E=core ./scripts/check.sh
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

## What to read next

- [CLI Reference](./cli.md) if you want command details
- [Configuration](./configuration.md) if you need to tune paths or runtime
  behavior
- [Repository Tour](./repository-tour.md) if you want to work on the code
- [Testing Guide](./testing.md) if you plan to contribute
