<p align="center">
  <a href="https://mere.run"><img src="https://mere.run/showcase/banner.png" alt="mere.run — Create anything. Locally." /></a>
</p>

<p align="center">
  <a href="https://mere.run">mere.run</a> ·
  <a href="https://docs.mere.run/">Docs</a> ·
  <a href="https://mere.run/releases">Downloads</a> ·
  <a href="https://relay.mere.run">Relay + Nodes</a> ·
  <a href="https://plugins.mere.run">Plugins</a> ·
  <code>swift build</code>
</p>

# mere.run

mere.run is a local-first inference runtime for Apple Silicon and headless Linux.
One public CLI covers image, text, speech, vision, music, sound, video, 3D,
worlds, training, and local API serving. The optional macOS Studio uses that
same CLI, model store, and run history rather than a separate backend.

<p align="center">
  <img src="docs/media/demo-inline.gif" alt="One terminal session: local chat, image generation, grounding, image-to-3D, SFX, music, TTS and ASR, video, an OpenAI-compatible API, and a workflow graph — rendered inline, all on one machine." />
</p>
<p align="center"><sub>One real terminal session on an M4 Max — chat, image, grounding, image→3D, SFX, music, speech round-trip, video, API serving, and a typed workflow graph. Everything local.</sub></p>

## Start here

Install a signed release from [mere.run/downloads](https://mere.run/releases),
then let the CLI inspect the machine before pulling a large model:

```bash
mere.run setup
mere.run model capabilities --recommended
mere.run guide --list
```

For a small first image workflow:

```bash
mere.run model pull image-zimage-nano
mere.run image generate \
  --model image-zimage-nano \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png
```

Use `mere.run guide <command path>` for the packaged offline cookbook behind a
creative or model-specific command, and `mere.run <group> --help` for the exact
current flags.

## What works today

| Area | Public commands | Current surface |
| --- | --- | --- |
| Images and LoRAs | `image generate`, `image train-lora`, `adapter list`, `adapter pull` | Klein, ZImage, HiDream O1, Krea 2, Ideogram 4, and Bonsai; text-to-image, edits, multiple references, structured prompts, local Krea/Klein training, and checksum-pinned public adapters |
| Text, code, and agents | `text chat`, `text code`, `text embed`, `text anonymize`, `text train-lora`, `agent` | Local chat and tool use, including Bonsai 27B binary/ternary vision chat; code generation, embeddings, PII redaction, text LoRA training, and guided local-agent setup |
| Vision understanding | `vision caption`, `inspect`, `face`, `ground`, `segment`, `track`, `track-live`, `pose`, `flow`, `ocr` | Captioning and VQA, local face detection/identity embeddings, LightOn/GLM/Infinity OCR, Falcon grounding, SAM 3.1 segmentation and tracking, body/hand/face landmarks, and dense optical flow |
| Depth, geometry, and 3D | `vision depth-video`, `geometry`, `geometry-multiview`, `image-to-3d*`; `image reconstruct-3d*` | Video Depth Anything, MoGe-2, Depth Anything 3, TripoSR, InstantMesh, and TRELLIS.2; depth/confidence EXRs, cameras, point clouds, 3DGS initialization, OBJ, PLY, GLB, and PBR voxel artifacts |
| Video and worlds | `video generate`, `video cosmos3`, `video prepare-masks`, `video animate`, `video session`, `video export-latents`, `world serve` | LTX video, synchronized LTX 2.3 audio/video, resident distilled and full-dev LTX workers, Wan 2.2 TI2V, native Cosmos3-Edge generation/reasoning/action dynamics, native SAM 3.1 mask preparation, native SCAIL-2 subject animation/replacement, and warm DreamX or Cosmos3 world sessions |
| Music and sound | `music analyze`, `generate`, `realtime`, `transcribe`; `sfx generate`, `sfx video generate` | ACE-Step generation, analysis, and covers; Magenta RT2 realtime MIDI performance; MuScriptor full-mix MIDI transcription; Woosh and native MMAudio text/video-conditioned sound |
| Speech | `speech synthesize`, `speech transcribe`, `speech diarize`, `speech listen`, `speech profile` | Qwen3 TTS, saved voice profiles, Qwen3 live ASR, Parakeet transcription, and native MLX Sortformer speaker diarization |
| Serving and operations | `api serve`, `open-webui quickstart`, `status`, `run`, `model runtime`, `gate` | OpenAI-compatible chat, embeddings, images, TTS, and STT; resident model pooling, TTL/pinning, memory guards, durable run inspection, and installed-model quality gates |
| Automation | `--preflight --json`, `--progress-json`, `image run-plan`, `guide` | Typed preflight actions, machine-readable progress, replayable plans, durable run directories, checksums, and offline command cookbooks |
| Portable workflows | `graph catalog`, `graph preflight`, `graph submit`, `executor`, `run watch`, `run fetch` | Immutable typed graphs and content-addressed bundles that run locally or through configured SSH and relay executors with the same events, diagnostics, and run directory |

The [showcase](https://mere.run/#showcase) publishes real outputs across these
surfaces. Model availability, memory fit, download size, and licensing still
vary by command, so check `mere.run model capabilities` before a large pull and
[`docs/model-sources.md`](./docs/model-sources.md) before redistributing model
artifacts.

## Relay, Nodes, and plugins

The core runtime stays local, but it does not have to stay on one machine.

- [Relay](https://relay.mere.run) pools Macs and Linux hosts approved under the
  same mere.world account. The separate `mere.run node` desktop app advertises
  each machine's models and availability, then accepts work through an outbound
  connection. Current node downloads cover macOS, Linux x86_64, and Linux arm64.
- [Official plugins](https://plugins.mere.run) are companion executables for
  VFX, realtime performance, production tracking, private document workflows,
  automation, and user-owned GPU training. They keep plans, manifests,
  resumability, artifacts, and cleanup outside the core inference process.

```bash
mere.run plugin list
mere.run plugin info mere-vfx-tools
mere.run plugin install mere-vfx-tools --yes
mere.run plugin doctor mere-vfx-tools
```

Relay/Node code lives in `sawfwair/relay-mere-run`; plugin contracts and source
live in [`sawfwair/mere-run-plugins`](https://github.com/sawfwair/mere-run-plugins).
Neither adds a hosted inference backend to this repository.

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

The latest signed macOS build is mirrored at:

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

Linux package builds are headless CLI-only. They install the `mere.run` CLI plus
colocated runtime assets; they do not include the macOS SwiftUI studio or DMG
layout. See the dedicated [Linux QuickStart](./docs/linux-quickstart.md) for the
current validation boundary, CUDA notes, and first commands. On Linux arm64, if a distro Clang shadows Swift's
bundled Clang and cannot compile MLX bf16 headers, the Linux scripts select a
bf16-capable C++ driver or report the `CXX` override to use. Linux arm64 release
packages should be built with CUDA enabled on a host with the CUDA Toolkit
headers, CUDA CCCL headers, cuDNN, and NCCL installed. CUDA `.deb` artifacts
derive their runtime/JIT dependencies from the linked `libcudart` major. CUDA
12 packages target the 12.8 NVIDIA packages with Lambda Stack alternatives;
CUDA 13 packages target the 13.0 NVIDIA packages. The
installed launcher also exports the resolved CUDA CCCL include root through
`MERERUN_CUDA_CCCL_INCLUDE_PATH` so MLX CUDA kernels can find `cuda/std/*`
during NVRTC JIT compilation:

```bash
MERERUN_LINUX_ACCEL=cuda MERERUN_SKIP_MLX_CUDA_EXAMPLE=1 \
  scripts/package-linux.sh --version 0.23.0 --artifact-suffix cuda
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

To build the x86_64 CUDA release package on a CUDA development host or a
GPU-less builder with CUDA development packages:

```bash
MERERUN_LINUX_ACCEL=cuda MERERUN_SKIP_MLX_CUDA_EXAMPLE=1 \
  scripts/package-linux.sh --version 0.23.0 --artifact-suffix cuda
ls dist/linux/
```

On Linux arm64, use a CUDA-provisioned host:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/package-linux.sh --version 0.23.0
```

Do not use the app bundle commands on Linux. `mere.run.app`, SwiftUI studio
flows, and DMG packaging stay macOS-only. Linux release packaging is for the
headless CUDA CLI tarball and `.deb` only; no CPU-only Linux artifact is
published.

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
# Runtime TTLs unload idle loaded models during pool operations.
# Memory-guard tiers drive pressure LRU and admission pauses.
# Pinned models are kept out of automatic eviction.

# See what this Mac can run before pulling large models
swift run mere.run model capabilities
swift run mere.run model capabilities --recommended

# Chat winners by RAM band
# 16-23 GB: text-chat-gemma4-12b-4bit
# 24-95 GB: text-chat-gemma4-12b-4bit
# 96+ GB: text-agent-deepseek-v4-flash for agent/API chat; Gemma 12B 4-bit for normal local chat
# Coding-agent comparison: text-code-north-mini, text-agent-ornith-35b-mlx, and text-agent-ornith-35b

# Choose guided, bring-your-own-agent, or manual setup
swift run mere.run setup

# Pull a Hugging Face-backed model into the local model store
swift run mere.run model pull image-zimage-nano
swift run mere.run model pull image-zimage-nano --preflight --json
swift run mere.run model pull text-chat-lfm25-a1b-8bit --accept-model-license
swift run mere.run model pull text-chat-laguna-s-2-1
swift run mere.run model pull text-chat-laguna-xs-2-1
swift run mere.run model pull text-chat-inkling-small
swift run mere.run model pull text-chat-bonsai-27b-1bit
swift run mere.run model pull text-chat-bonsai-27b-2bit
swift run mere.run model pull text-code-north-mini
swift run mere.run model pull text-agent-ornith-35b
# text-agent-ornith-35b-mlx is local-only until a converted MLX snapshot is published

# Run Prism ML's binary model for lower residency/faster decode, or substitute
# text-chat-bonsai-27b-2bit for the larger ternary checkpoint.
swift run mere.run text chat \
  --model text-chat-bonsai-27b-1bit \
  --prompt "Explain unified memory for local inference."

# Inkling-Small is an explicit mixed-precision native MLX pull for 128 GB machines.
swift run mere.run text chat \
  --model text-chat-inkling-small \
  --context-size 32768 \
  --prompt "Plan a recovery-safe repository migration."

# Pull and apply the promoted Mere Platform Assistant adapter
swift run mere.run model pull text-chat-gemma4-12b-4bit
swift run mere.run adapter pull mere-platform-assistant
swift run mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora mere-platform-assistant \
  --prompt "What can you help me with in Mere?"

# Generate an image
swift run mere.run image generate \
  --prompt "a ceramic coffee mug in soft morning light" \
  --output ./mug.png

# Preflight the same request as JSON before loading the image model
swift run mere.run image generate \
  --prompt "a ceramic coffee mug in soft morning light" \
  --output ./mug.png \
  --preflight \
  --json

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

# Krea 2 Turbo runs natively for 8-step text-to-image generation.
swift run mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a cinematic product photo of a translucent portable speaker, crisp reflections" \
  --steps 8 \
  --output ./speaker.png

# Train Krea 2 LoRAs on Raw, then run them on Turbo.
swift run mere.run image train-lora \
  --model image-krea2-raw \
  --data ./style-dataset \
  --output ./style-krea2.safetensors \
  --recipe krea-cinematic-style \
  --quiet

# Train a practical local Klein style LoRA with the fast 9B recipe.
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --visualize \
  --quiet

# Apply a Klein LoRA to a reference image.
swift run mere.run image generate \
  --model image-klein-9b \
  --ref-image ./reference-pose.png \
  --strength 0.55 \
  --prompt "TRIGGER_TOKEN two dancers in a rainy city street, natural human anatomy, no extra limbs" \
  --lora ./style-adapter.safetensors \
  --lora-scale 1.5 \
  --width 1024 --height 768 \
  --steps 16 \
  --seed 525252 \
  --output ./style-reference.png

# Reopen the local run dashboard later from the run directory.
swift run mere.run image visualize-run .

# Discover and inspect run artifacts headlessly.
swift run mere.run run list --root ./runs --json
swift run mere.run run inspect ./runs/render --json
swift run mere.run run inspect ./render.plan.json --json

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
swift run mere.run api serve --engine text-chat-laguna

# Inspect the serving plan before starting the server or loading a model
swift run mere.run api serve --engine text-chat-gemma4 --preflight --json

# In another terminal, confirm the server and served model
swift run mere.run status

# Native embeddings for local RAG clients
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  --data '{"model":"text-embed-qwen3-0.6b","input":"mere.run native embeddings"}'

# Native image generation and audio through the same /v1 API
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  --data '{"model":"image-zimage-nano","prompt":"a compact local AI workstation","size":"1024x1024"}'
curl http://127.0.0.1:8080/v1/images/edits \
  -F model=qwen-image-edit \
  -F prompt="make the workstation dusk-lit while preserving the layout" \
  -F image=@input.png
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output speech.wav \
  --data '{"model":"speech-tts-qwen3-nano","input":"mere.run is online","voice":"nova","response_format":"wav"}'
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=speech-asr-parakeet \
  -F file=@speech.wav

# Optional Open WebUI companion smoke
swift run mere.run open-webui quickstart --dry-run
swift run mere.run open-webui quickstart --pull --accept-model-license
swift run mere.run guide open-webui
scripts/smoke-open-webui.sh print-env
scripts/smoke-open-webui.sh live-smoke

# Gemma4 prefix KV reuse is on by default for api serve; set 0 for a baseline
MERERUN_GEMMA4_PREFIX_KV_CACHE=0 swift run mere.run api serve --engine text-chat-gemma4

# Qwen3.6 text-only prefix KV reuse is also on by default; set 0 for a baseline
MERERUN_Q35_PREFIX_KV_CACHE=0 swift run mere.run api serve

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

# Fixed-token real-checkpoint Gemma4 MTP benchmark: serial decode vs verified MTP
swift run mere.run model benchmark gemma4-mtp \
  --model text-chat-gemma4-12b-4bit \
  --prompt-repeat-values 128,220 \
  --decode-token-values 48,128 \
  --json

# Requested-token real-checkpoint Qwen3.6 MTP benchmark: baseline vs adaptive vs forced
swift run mere.run model benchmark q36-mtp \
  --prompt-repeat-values 8,80,150 \
  --temperature-values 0,0.7 \
  --decode-tokens 32 \
  --json

# Real serving-path workload: streaming chat TTFT, throughput, cache, and batching counters
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 1
swift run mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --json

# Small real coding-eval slice: Ornith vs North Mini vs Qwen3-Coder
swift run mere.run model benchmark code \
  --allow-code-execution \
  --json

# Small grounded-chat eval: local email/workspace evidence, abstention, and format checks
swift run mere.run model benchmark chat --json

# Small tool-call eval: synthetic Mere tools, parsed tool names/arguments only
swift run mere.run model benchmark tool-calls --json

# Gemma4 tool-result continuation: one- and two-tool chains against a real checkpoint
swift run mere.run model benchmark tool-continuations --log-responses --json

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

# Detect and compare faces locally
swift run mere.run model pull vision-face-buffalo-l --accept-model-license
swift run mere.run vision face detect ./group.jpg --json
swift run mere.run vision face compare ./reference.jpg ./candidate.jpg --json

# Segment an image
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision segment ./image.png --prompt "a person"

# Track prompted objects through a video
swift run mere.run vision track ./clip.mp4 --prompt "a person"
swift run mere.run vision track ./clip.mp4 --prompt "a person" --preflight --json

# Record a short camera session and track it
swift run mere.run vision track-live --output ./live.mp4 --prompt "a person"
swift run mere.run vision pose ./person.png --json-output ./person-pose.json
swift run mere.run vision flow ./frame-001.png ./frame-002.png --output ./motion.flo

# Generate music
swift run mere.run model pull music-acestep
swift run mere.run music generate \
  "upbeat electronic groove" \
  --output ./track.wav

# Generate with ACE-Step XL Turbo on larger Macs
swift run mere.run model pull music-acestep-xl-turbo
swift run mere.run music generate \
  "cinematic synth pop with bright vocal harmonies" \
  --model music-acestep-xl-turbo \
  --output ./xl-track.wav

# Generate an ACE-Step cover from a source song
swift run mere.run music generate \
  "dream-pop cover with soft vocals" \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
  --audio-cover-strength 1.0 \
  --output ./cover.wav

# Analyze a source song before cover/remix work
swift run mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-turbo-lm4b \
  --lm-subdirectory acestep-5Hz-lm-4B \
  > ./song-analysis.json

# Transcribe a full mix into instrument-separated MIDI with MuScriptor
swift run mere.run model pull music-muscriptor-medium --accept-model-license
swift run mere.run music transcribe ./song.mp3 \
  --output ./song.mid \
  --context-output ./song-context.json

# Generate a style-transfer cover from a source song
swift run mere.run music generate \
  "modern reggaeton dance club remix, 96 bpm dembow rhythm, syncopated kick-snare groove, punchy 808 sub bass, bright Latin percussion" \
  --model music-acestep-xl-turbo \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
  --audio-cover-strength 0.20 \
  --cover-noise-strength 0.0 \
  --output ./reggaeton-cover.wav

# Stream and optionally capture Magenta RT2 music on Apple Silicon macOS
swift run mere.run music realtime \
  "ambient modular synths with brushed drums" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./live.wav \
  --no-play

# Steer realtime Magenta RT2 from a CoreMIDI controller such as OP-1
swift run mere.run music realtime --list-midi-inputs
swift run mere.run music realtime \
  --midi-monitor \
  --midi-input "OP-1 Bluetooth" \
  --midi-log-raw \
  --duration 30
swift run mere.run music realtime \
  "minimal synth pop, dry drums, tape-warped bass" \
  --model music-magenta-rt2-small \
  --duration 120 \
  --midi-input "OP-1 Bluetooth" \
  --midi-channel all \
  --midi-log-events \
  --midi-cc 1=temp:0.2:1.4 \
  --midi-cc 2=drums:0:2

# Generate a Foley / sound-effect WAV with Woosh or MMAudio
swift run mere.run model pull sfx-woosh-dflow --accept-model-license
swift run mere.run model pull sfx-woosh-flow --accept-model-license
swift run mere.run sfx generate \
  "metal wrench dropping onto concrete, bright clang and brief ring" \
  --model sfx-woosh-dflow \
  --duration 5 \
  --output ./wrench-clang.wav
swift run mere.run sfx ae encode ./wrench-clang.wav -o ./wrench-clang-latents.npy
swift run mere.run sfx ae decode ./wrench-clang-latents.npy -o ./wrench-clang-roundtrip.wav
swift run mere.run sfx clap score \
  "metal wrench dropping onto concrete" \
  ./wrench-clang.wav
swift run mere.run model pull sfx-woosh-dvflow-8s --accept-model-license
swift run mere.run model pull sfx-woosh-synchformer --accept-model-license
swift run mere.run sfx video generate \
  "footsteps echoing in a hallway" \
  ./silent-hallway.mp4 \
  --model sfx-woosh-dvflow-8s \
  --output ./hallway-footsteps.wav
swift run mere.run sfx video generate \
  "footsteps echoing in a hallway" \
  ./silent-hallway.mp4 \
  --model sfx-woosh-dvflow-8s \
  --output ./hallway-footsteps.wav \
  --preflight \
  --json
swift run mere.run model pull sfx-mmaudio-large-44k-v2 --accept-model-license
swift run mere.run sfx generate \
  "ocean waves striking a stone breakwater" \
  --negative-prompt "speech, music" \
  --model sfx-mmaudio-large-44k-v2 \
  --duration 8 \
  --steps 25 \
  --output ./breakwater.wav

# MMAudio weights are non-commercial; its Apple CLIP component is research-only.
# See docs/model-sources.md for the component-level licenses.

# Generate a fast video-only draft
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --num-frames 65 \
  --output ./clip.mp4

# Inspect the same render before loading MLX or writing an MP4
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --num-frames 65 \
  --output ./clip.mp4 \
  --preflight \
  --json

# Anchor the first and last keyframes for directed image-to-video
swift run mere.run video generate \
  "a car drives from a bright morning street into a warm sunset road, smooth forward motion" \
  --image ./car-start.png \
  --end-image ./car-end.png \
  --num-frames 65 \
  --output ./clip-directed.mp4

# Generate final-quality video without an audio stream
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "a red fox runs across a snowy clearing, detailed winter fur, natural motion" \
  --quality final \
  --duration 4 \
  --output ./clip-final.mp4

# Generate synchronized final-quality LTX 2.3 audio/video
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "dialogue with clean background music and subtle city ambience" \
  --quality final \
  --output-mode audio-video \
  --duration 15 \
  --fps 24 \
  --output ./clip-av.mp4

# Condition video on a selected source-audio segment and preserve that soundtrack
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "a kinetic live performance, camera orbiting the vocalist" \
  --audio ./song.wav \
  --audio-start-time 30 \
  --duration 5 \
  --image ./performer.png \
  --output ./performance.mp4

# Animate a masked reference subject from a driving video with native SCAIL-2
swift run mere.run model pull video-scail2-14b-mlx
swift run mere.run adapter pull scail2-lightx2v-4step
swift run mere.run video animate \
  "a dancer in a red silk dress" \
  --reference ./reference-dancer-prepared.png \
  --reference-mask ./reference-dancer-mask.png \
  --driving-video ./pose.mp4 \
  --driving-mask ./pose-mask.mp4 \
  --tail-policy pad-trim \
  --audio-source driving \
  --output ./animated.mp4
```

## Command tree

The public CLI is modality-first:

- `mere.run guide`
- `mere.run image { dataset, generate, reconstruct-3d, reconstruct-3d-trellis2, reconstruct-3d-multiview, run-plan, train-lora, visualize-run, validate }`
- `mere.run text { chat, code, embed, anonymize, train-lora }`
- `mere.run speech { synthesize, transcribe }`
- `mere.run speech profile { list, create, delete }`
- `mere.run vision { caption, inspect, ground, segment, track, track-live, pose, flow, depth-video, geometry, geometry-multiview, image-to-3d, image-to-3d-trellis2, image-to-3d-multiview, ocr }`
- `mere.run music { analyze, generate, realtime, transcribe }`
- `mere.run sfx { ae, clap, condition, generate, video }`
- `mere.run video { prepare-masks, animate, cosmos3, generate, session, export-latents }`
- `mere.run world serve`
- `mere.run run { list, inspect }`
- `mere.run model { list, pull, remove, info, capabilities, runtime, benchmark, repair-manifests }`
- `mere.run adapter { list, pull }`
- `mere.run status`
- `mere.run gate`
- `mere.run config { set, get, unset, list, path }`
- `mere.run api serve`
- `mere.run open-webui quickstart`
- `mere.run plugin { list, info, install, doctor }`
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

Inspect physical usage and sharing before deleting anything:

```bash
mere.run model storage
mere.run model gc          # read-only plan
mere.run model gc --force  # recompute under lock, then delete
```

`model remove <id>` reclaims backing payloads only when no other current or
legacy model link uses them; `--keep-cache` retains those bytes intentionally.
New pulls are revision-addressed and hard-link matching content blobs, while
existing legacy cache directories remain compatible and can be adopted without
copying payload bytes.

## Security defaults

The public OSS build keeps local-first behavior by default and requires explicit opt-in for higher-risk modes:

- `mere.run api serve --preflight --json` reports host/port, auth, model,
  runtime, and redacted follow-up actions before starting a server or loading a
  model
- `mere.run api serve` can bind to loopback without auth, but non-loopback hosts require `--api-key` or `MERERUN_API_KEY`
- the OpenAI-compatible chat and embedding routes require `Content-Type: application/json`, support `--rate-limit-per-minute` for basic abuse control, decode the common OpenAI request shapes, and reject unsupported high-impact fields before generation
- API LoRA adapters are operator-controlled with `--lora`; it accepts a
  verified installed adapter catalog id or a local path, while per-request LoRA
  paths are rejected
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
- [`docs/workflows.md`](./docs/workflows.md): portable graphs, job bundles, SSH,
  relay, worker protocol, and remote run lifecycle
- [`docs/repository-tour.md`](./docs/repository-tour.md): top-level layout and module ownership
- [`docs/development-workflow.md`](./docs/development-workflow.md): how to work in the repo day to day
- [`docs/testing.md`](./docs/testing.md): validation layers, smoke runs, and troubleshooting
- [`docs/runtime/vision.md`](./docs/runtime/vision.md): native SAM 3.1 segmentation and tracking details
- [`docs/runtime/sfx.md`](./docs/runtime/sfx.md): native Woosh and MMAudio SFX generation, CLAP scoring, and video-conditioned generation details

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
