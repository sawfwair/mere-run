# Getting started

This page walks you through building the package from source, pulling a
model, and generating a first image, paragraph, and spoken line. All inference
runs on your machine.

If you only want to use the app, install a signed build from
[mere.run releases](https://mere.run/releases) and go to
[Pull a model](#pull-a-model). The remaining sections describe the from-source
path.

## What you are building

`mere-run` is one Swift package that produces:

- The public `mere.run` executable, which provides the command-line interface
  (CLI) and performs the work
- `mere.run.app`, an optional macOS studio that uses the same CLI
- Reusable inference libraries in `Sources/MereRunCore`, `Sources/AudioCore`,
  `Sources/AudioCodecs`, `Sources/AudioSTT`, and `Sources/AudioTTS`
- Tests and smoke harnesses for both interfaces

No hosted inference service runs behind it. Inference stays local. The CLI
and app use the network only for explicit operations such as model downloads,
installation and update checks, and Relay.

## Prerequisites

For the supported macOS developer path:

- An Apple Silicon Mac
- macOS 15 or later
- Xcode command-line tools
- SwiftLint and ripgrep for `./scripts/check.sh`
  (`brew install swiftlint ripgrep`)
- Enough disk space for model installs in
  `~/Library/Application Support/MereRun/models`

For Linux CLI compatibility work:

- Swift 6.1 or later
- `clang`, `cmake`, `ninja`, `pkg-config`, `gfortran`, and the curl, zlib,
  OpenBLAS, and LAPACK development headers
- `ffmpeg` and `ffprobe` for media probing and conversion
- `gzip`, `unzip`, and `zip` for portable low-rank adaptation (LoRA)
  checkpoint archives
- Enough disk space for a headless model store

On Ubuntu-style hosts, install the system packages:

```bash
sudo apt-get update
sudo apt-get install -y clang cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
```

If the media tools aren't on `PATH`, point the CLI at explicit binaries:

```bash
export MERERUN_FFMPEG=/opt/ffmpeg/bin/ffmpeg
export MERERUN_FFPROBE=/opt/ffmpeg/bin/ffprobe
```

## Build the package

From the repository root:

```bash
swift build
swift test
swift run mere.run --help
app_path="$(./scripts/build_mere_run_app.sh debug)"
open "$app_path"
```

If all five commands succeed, the package resolves, the CLI builds and parses
its arguments, and the Mac app bundles and opens.

On Linux, build only the headless CLI:

```bash
swift run mere.run --help
```

The Linux build doesn't include the macOS app product.

## Launch the macOS studio

The Studio is a prompt-first interface for the CLI you built. The sidebar
lists fifteen domains, each domain has its tasks in a control at the top of the
page, and a prompt task gives you one composer, a feed of your runs, and a local
library of generated artifacts. Press ⌥⌘C to see the exact command behind a
generation, or open the Command Console (Window ▸ Command Console, ⇧⌘C) to run any
`mere.run` command. Launch the Studio from a checkout:

```bash
app_path="$(./scripts/build_mere_run_app.sh debug)"
open "$app_path"
```

For contributor smoke tests, `swift run mere.run.app` also builds the
executable product. Use the bundle script for normal macOS window behavior.

The app looks for a CLI in this order: the bundled CLI, nearby Swift Package
Manager build products, common install locations, and the active package
checkout. It doesn't install the terminal command on launch. To copy the
bundled command or the `use-mere-run` Codex skill into user-visible locations,
open **Settings** and click **Install CLI** or **Install Skill**.

The Studio is macOS-only. On Linux, use the CLI and the local API server
directly.

MereRun 0.27.1 and later checks the stable signed update feed once per day.
To check manually, click **MereRun > Check for Updates…**. Each update replaces
the whole signed app bundle atomically, so the Studio and its embedded CLI stay
on the same version. Releases earlier than 0.27.1 don't include the updater
and need one manual disk image install before they can update in the app.

## Build Linux package artifacts

Linux package artifacts contain only the headless `mere.run` CLI and its
colocated runtime assets. They don't include `mere.run.app`, the SwiftUI
Studio, or the macOS disk image layout. To build packages for version 0.67.1,
run:

```bash
scripts/package-linux.sh --version 0.67.1
ls dist/linux/
```

Published Linux packages target NVIDIA's Compute Unified Device Architecture
(CUDA). Build and smoke-test each CUDA package on matching CUDA hardware before
you treat it as supported. CPU packages are local test artifacts, not release
targets. For setup commands, package checks, and CUDA validation limits, see
the [Linux quickstart](./linux-quickstart.md).

## Understand the command tree

Commands are grouped by what you want to make or analyze, such as `image`,
`text`, `speech`, `vision`, `music`, `sfx`, `video`, and `world`. Separate
command groups run graphs, manage models, and serve APIs. The following table
is generated from the CLI:

<!-- BEGIN GENERATED: CLI TOP LEVEL -->
| Command | Purpose |
| --- | --- |
| [`mere.run guide`](/cookbooks) | Read offline mere.run command cookbooks. |
| [`mere.run catalog`](/cli) | Inspect the machine-readable command capability contract. |
| [`mere.run image`](/runtime/image) | Generate and validate image models. |
| [`mere.run text`](/runtime/text) | Run local chat, code, embedding, anonymization, classification, and decision workflows. |
| [`mere.run speech`](/runtime/speech) | Synthesize, transcribe, diarize, and manage voice profiles. |
| [`mere.run vision`](/runtime/vision) | Embed, caption, inspect, face-analyze, segment, track, pose, depth, geometry, optical flow, and OCR visual media. |
| [`mere.run geo`](/runtime/geo) | Run native geospatial inference models on local Earth-observation data. |
| [`mere.run audio`](/runtime/audio) | Generate, edit, and enhance audio locally. |
| [`mere.run music`](/runtime/music) | Generate, analyze, transcribe, and separate music locally. |
| [`mere.run sfx`](/runtime/sfx) | Generate sound effects locally. |
| [`mere.run video`](/runtime/video) | Generate and understand video with native Swift/MLX pipelines. |
| [`mere.run world`](/runtime/world) | Run persistent local conditioned-video world sessions. |
| [`mere.run graph`](/workflows) | Validate, materialize, run, and submit portable workflow graphs. |
| [`mere.run executor`](/workflows#executor-profiles) | Manage local, SSH, and relay workflow executors. |
| [`mere.run relay`](/workflows#direct-relay-relay-serve) | Host the relay API surface directly on this machine. |
| [`mere.run run`](/workflows#run-directories) | Inspect durable mere.run workflow reports and run directories. |
| [`mere.run eval`](/evaluation-packs) | Run reproducible evaluations from external, content-addressed packs. |
| [`mere.run model`](/runtime/model-management) | List, pull, locate, remove, inspect, optimize, and clean up models. |
| [`mere.run adapter`](/runtime/model-management) | List and pull verified LoRA adapters. |
| [`mere.run status`](/runtime/model-management) | Show local server, loaded model, and installed model status. |
| [`mere.run gate`](/gate) | Run the end-to-end quality gate against installed models. |
| [`mere.run config`](/configuration) | Get and set persisted mere.run configuration (e.g. Hugging Face token). |
| [`mere.run api`](/runtime/api-server) | Serve local models through API surfaces. |
| [`mere.run open-webui`](/runtime/api-server#open-webui-companion) | Start the optional Open WebUI companion against a local mere.run API. |
| [`mere.run plugin`](/plugins) | Discover and install official mere.run companion plugins. |
| [`mere.run setup`](/getting-started) | Choose a guided, BYOA, or manual mere.run setup path. |
| [`mere.run agent`](/getting-started) | Install and start the optional guided local setup agent. |
<!-- END GENERATED: CLI TOP LEVEL -->

For all commands and options, see the [CLI reference](./cli.md).

## Choose a model store location

By default, models live in:

```text
~/Library/Application Support/MereRun/models
```

To use a different store for a shell session, set `MERERUN_MODELS_DIR`:

```bash
export MERERUN_MODELS_DIR=/path/to/models
```

To use a different store for one command, pass `--models-root`:

```bash
swift run mere.run --models-root /path/to/models model list
```

## Check local status

To see a snapshot of this machine's mere.run state, run `status`:

```bash
swift run mere.run status
```

The command reports whether the local API server is reachable, which model
the server exposes through `/v1/models`, the active model store, and the
managed models installed there. For scripts, request JSON output:

```bash
swift run mere.run status --json
```

## Pull a model

Models download from cataloged Hugging Face repositories into a local store
that you own. mere.run has no private model host and no inference credential to
buy. An upstream repository can still require you to accept its terms and
provide a Hugging Face token.

Before you download several gigabytes, check which models this machine can
run, and then pull one:

```bash
swift run mere.run model capabilities
swift run mere.run model pull image-zimage-nano
# Optional compact FLUX.2 Klein path:
swift run mere.run model pull image-bonsai-binary
```

To keep every Hugging Face cache operation on another disk, set
`MERERUN_HUB_CACHE`. For a single pull, pass `model pull --cache-dir PATH`. A
model installed through an external cache is unavailable while that volume is
disconnected. For every model ID and its source, see
[model sources](./model-sources.md).

For guided onboarding, run:

```bash
swift run mere.run setup
```

The setup command offers a local Mere agent powered by Pi, a
bring-your-own-agent handoff prompt for Claude or Codex, or manual commands. To
select the tool-capable Ornith 1.0 9B setup agent, pass
`--mode agent --agent-model small`; it needs at least 16 GB of unified memory.
On Apple Silicon Macs with at least 96 GB of unified memory, the default `tier`
choice and the `premier` choice select DeepSeek V4 Flash. To preview the steps
without installing anything, add `--dry-run`. On Linux, install Pi or provide
it with `--pi-path` or `PATH` before you use `--start`.

### Agent commands

`mere.run agent` runs the guided setup agent directly. Its default
subcommand is `onboard`:

- `mere.run agent onboard`: summarizes this machine's model capabilities and
  prepares the optional Pi agent.
- `mere.run agent status`: inspects agent, Pi, provider, and model readiness.
- `mere.run agent install-pi`: installs the most recent published Pi
  coding-agent release.
- `mere.run agent start`: starts Pi against a local mere.run setup-agent API
  server.

To pull the recommended setup model and start the agent, run:

```bash
swift run mere.run agent onboard --pull-recommended --accept-model-license
swift run mere.run agent start
```

`agent onboard` also accepts `--install-pi`, `--configure-pi`, `--model`,
`--host`, and `--port`, which configure the Pi provider extension. By default,
`agent start` downloads a missing managed model and installs Pi. To refuse
both, pass `--no-bootstrap`. Other `start` options include `--model`,
`--prompt`, `--inline`, `--skip-server`, `--allow-unsupported`, and
`--pi-path`.

## Make something

Each of the following commands runs independently, so you can start with any
of them. Text chat, Qwen3 speech synthesis, and vision inspection download
their default models on first use. Other examples include a `model pull` step
or use a model from [Pull a model](#pull-a-model). If a command reports a
missing model, run `mere.run model pull MODEL_ID` with the ID it names.

### Image generation

```bash
swift run mere.run image generate \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png

swift run mere.run image generate \
  --model image-bonsai-binary \
  --prompt "a tiny bonsai tree in a sunlit greenhouse" \
  --output ./bonsai.png

# Review the Krea 2 community license before accepting it.
swift run mere.run model pull image-krea2-turbo --accept-model-license
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

### Speech transcription

Transcription uses Parakeet by default. Whistle and Nemotron 3.5 Transcribe
are also available through `--backend`; for details, see the
[speech guide](./runtime/speech.md).

```bash
swift run mere.run model pull speech-asr-parakeet
swift run mere.run speech transcribe ./hello.wav
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

### Face analysis

```bash
swift run mere.run model pull vision-face-buffalo-l --accept-model-license
swift run mere.run vision face detect ./group.jpg --json
swift run mere.run vision face compare ./reference.jpg ./candidate.jpg --json
```

### Vision segment

```bash
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision segment ./image.png --prompt "a person"
swift run mere.run vision track ./clip.mp4 --prompt "a person"
swift run mere.run vision track-live --output ./live.mp4 --prompt "a person"
```

## Validate your local environment

Before you open a pull request, run the same check script that CI runs:

```bash
./scripts/check.sh
```

To add end-to-end smoke tests, set `MERERUN_RUN_E2E`:

```bash
MERERUN_RUN_E2E=core ./scripts/check.sh
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

## What to read next

- [CLI reference](./cli.md): every command and option
- [Offline cookbooks](./cookbooks.md): model and command guides from
  `mere.run guide`
- [Configuration](./configuration.md): paths and runtime behavior
- [Repository tour](./repository-tour.md): where the code lives
- [Testing guide](./testing.md): validation before you contribute
