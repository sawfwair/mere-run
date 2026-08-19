# Getting Started

By the end of this page you will have built the package from source, pulled a
model, and generated your first image, paragraph, and spoken line — all on your
own machine, with nothing leaving it.

If you only want to use the app, install a signed build from
[mere.run/releases](https://mere.run/releases) instead and jump straight to
[Pull a model](#pull-a-model). Everything below is the from-source path.

## What you are building

`mere-run` is one Swift package that produces:

- the public `mere.run` executable — the CLI that does all the actual work
- `mere.run.app`, an optional macOS studio that drives that same CLI rather
  than a separate backend
- reusable inference libraries in `Sources/MereRunCore`, `Sources/AudioCore`,
  `Sources/AudioCodecs`, `Sources/AudioSTT`, and `Sources/AudioTTS`
- tests and smoke harnesses for both surfaces

There is no hosted inference service behind it. Inference stays local; network
access happens only for explicit operations such as model downloads,
installation and update checks, or Relay.

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

If all five succeed, the package resolves, the CLI builds and parses, and the
Mac app bundles and opens. That is the whole toolchain proven in one pass.

On Linux compatibility branches, keep the first pass headless and stop at the
CLI surface:

```bash
swift run mere.run --help
```

The macOS app product is intentionally outside the Linux target.

## Launch the macOS studio

The studio is a prompt-first face on the same CLI you just built — one canvas,
one prompt bar, and a local library of everything you have made. The exact
command behind each generation is always one click away under Advanced. Launch
it from a checkout with:

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

MereRun 0.27.1 and newer checks the stable signed update feed once per day and
also exposes **MereRun > Check for Updates…**. Updates replace the whole signed
app bundle atomically, so the Studio and its embedded CLI stay on the same
version. Releases older than 0.27.1 do not contain the updater and need one
final manual DMG install before future updates can arrive in-app.

## Build Linux package artifacts

Linux package artifacts are headless CLI-only. They install the `mere.run` CLI
plus colocated runtime assets; they do not include `mere.run.app`, SwiftUI
studio flows, or the macOS DMG layout. For a Linux-only setup path, first
commands, package checks, and CUDA validation limits, see
[Linux QuickStart](./linux-quickstart.md).

```bash
scripts/package-linux.sh --version 0.23.0
ls dist/linux/
```

CUDA packages must be built and smoke-tested on matching CUDA hardware before
being treated as supported. Linux arm64 packages are CUDA-only and should be
built on a real arm64 CUDA host with `MERERUN_LINUX_ACCEL=cuda`; CPU arm64
packages are local smoke artifacts, not useful release targets.

## Understand the command tree

Commands are organized by what you want to make — `image`, `text`, `speech`,
`vision`, `music`, `sfx`, `video`, `world` — with separate families for running
graphs, managing models, and serving. This table is generated from the CLI
itself, so it always matches the binary you just built:

<!-- BEGIN GENERATED: CLI TOP LEVEL -->
| Command | Purpose |
| --- | --- |
| [`mere.run guide`](/cookbooks) | Read offline mere.run command cookbooks. |
| [`mere.run catalog`](/cli) | Inspect the machine-readable command capability contract. |
| [`mere.run image`](/runtime/image) | Generate and validate image models. |
| [`mere.run text`](/runtime/text) | Run local chat, code, embedding, and anonymization workflows. |
| [`mere.run speech`](/runtime/speech) | Synthesize, transcribe, diarize, and manage voice profiles. |
| [`mere.run vision`](/runtime/vision) | Caption, inspect, face-analyze, segment, track, pose, depth, geometry, optical flow, and OCR visual media. |
| [`mere.run geo`](/runtime/geo) | Run native geospatial inference models on local Earth-observation data. |
| [`mere.run audio`](/runtime/audio) | Enhance general audio locally. |
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

Models come from cataloged Hugging Face repos into a local store you own. There
is no private mere.run model host or inference credential to buy. An upstream
repository can still require terms acceptance and a Hugging Face token.

Ask the machine what it can actually run before spending gigabytes on a
download:

```bash
swift run mere.run model capabilities
swift run mere.run model pull image-zimage-nano
# Optional compact FLUX.2 Klein path:
swift run mere.run model pull image-bonsai-binary
```

Set `MERERUN_HUB_CACHE` when you want every Hugging Face cache operation on
another disk, or pass `model pull --cache-dir PATH` for one explicit pull. A
model installed through an external cache is unavailable while that volume is
disconnected.
See [Model Sources](./model-sources.md) for the full matrix.

For guided onboarding, run:

```bash
swift run mere.run setup
```

The setup command offers a local Mere agent powered by Pi, a bring-your-own-agent
handoff prompt for Claude/Codex, or manual commands. Use
`--mode agent --agent-model small` to select the tool-capable native Ornith 9B
setup agent explicitly. On 96 GB+ Apple Silicon Macs, the hardware-tier and premier agent
path selects DeepSeek V4 Flash as the preferred setup agent. On Linux, install
or provide Pi separately with `--pi-path` or PATH before using `--start`.

### Agent commands

`mere.run agent` exposes the guided setup agent directly, with `onboard` as
the default subcommand:

- `mere.run agent onboard` — summarize this machine's model capabilities and
  prepare the optional Pi agent
- `mere.run agent install-pi` — install the latest Pi coding-agent release
- `mere.run agent start` — start Pi against a local mere.run setup-agent API
  server

```bash
swift run mere.run agent onboard --pull-recommended --accept-model-license
swift run mere.run agent start
```

`agent onboard` also takes `--install-pi`, `--configure-pi`, `--model`, and
`--host`/`--port` to write the Pi provider extension. `agent start`
bootstraps by default — it auto-pulls the missing managed model from Hugging
Face and auto-installs Pi; pass `--no-bootstrap` to refuse both. Other
`start` flags include `--model`, `--prompt`, `--skip-server`,
`--allow-unsupported`, and `--pi-path`.

## Make something

Each of these writes a real file to disk, and none of them depend on the
others. Start wherever you are curious.

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

Before you open a pull request, run the same script CI does:

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
