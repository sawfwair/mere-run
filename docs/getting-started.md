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

- Apple Silicon Mac
- macOS 15 or newer
- Xcode command line tools
- SwiftLint and ripgrep for `./scripts/check.sh`
  (`brew install swiftlint ripgrep`)
- enough disk space for model installs in
  `~/Library/Application Support/MereRun/models`

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
package checkout.

## Understand the command tree

The public CLI is modality-first:

- `mere.run image ...`
- `mere.run text ...`
- `mere.run speech ...`
- `mere.run vision ...`
- `mere.run music ...`
- `mere.run video ...`
- `mere.run model ...`
- `mere.run api ...`

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

## Configure model downloads

The OSS repo does not assume a built-in hosted download endpoint. Before
running `mere.run model pull`, configure one of these:

- `MERERUN_MODEL_SOURCE_BASE_URL` for unsigned archives
- `MERERUN_R2_SIGNED_URL_ENDPOINT` for a signed download service
- direct R2 credentials through
  `MERERUN_R2_ACCOUNT_ID`, `MERERUN_R2_ACCESS_KEY_ID`, and
  `MERERUN_R2_SECRET_ACCESS_KEY`

Example:

```bash
export MERERUN_MODEL_SOURCE_BASE_URL=https://your-host.example/models/
swift run mere.run model capabilities
swift run mere.run model pull image-zimage-nano
```

See [Model Sources](./model-sources.md) for the full matrix.

For guided onboarding, run:

```bash
swift run mere.run setup
```

The setup command offers a local Mere agent powered by Pi, a bring-your-own-agent
handoff prompt for Claude/Codex, or manual commands. Use
`--mode agent --agent-model small` to select the Qwen3.5 9B GGUF setup agent
explicitly.

## Run a first workflow

### Image generation

```bash
swift run mere.run image generate \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png
```

### Text chat

```bash
swift run mere.run text chat \
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
