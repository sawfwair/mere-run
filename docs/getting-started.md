# Getting Started

This guide gets a fresh clone to the point where you can build the package,
inspect the command tree, configure model sources, and run a first local
workflow.

## What this repo contains

`mere-run` is a Swift package with:

- reusable inference libraries in `Sources/MereRunCore`, `Sources/AudioCore`,
  `Sources/AudioCodecs`, `Sources/AudioSTT`, and `Sources/AudioTTS`
- a public executable product named `mere.run`
- tests and smoke harnesses for the package and CLI surfaces

It does not include the older app, relay, billing, or hosted-service layers.

## Prerequisites

- Apple Silicon Mac
- macOS 15 or newer
- Xcode command line tools
- enough disk space for model installs in
  `~/Library/Application Support/MereRun/models`

## Build the package

From the repo root:

```bash
swift build
swift test
swift run mere.run --help
```

That confirms the package graph, CLI product, and basic command parsing are all
working.

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

If you are migrating from an older mere.run CLI build, run:

```bash
./scripts/migrate_model_store.sh
```

That renames pre-OSS model directories to the canonical public IDs used by this
repo.

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
swift run mere.run model pull image-zimage-max
```

See [Model Sources](./model-sources.md) for the full matrix.

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
./scripts/migrate_model_store.sh
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

## What to read next

- [CLI Reference](./cli.md) if you want command details
- [Configuration](./configuration.md) if you need to tune paths or runtime
  behavior
- [Repository Tour](./repository-tour.md) if you want to work on the code
- [Testing Guide](./testing.md) if you plan to contribute
