# mere.run CLI

`mere.run` is the public command-line interface for the OSS `mere.run`
package. It exposes a modality-first command tree for image, text, speech,
vision, geometry, 3D, music, SFX, video, persistent worlds, model and adapter
management, local status snapshots, and local API serving.

If you are looking for the broader docs set, start at [mere.run Documentation](/).

## Overview

```bash
swift run mere.run --help
```

Public tree:

<!-- BEGIN GENERATED: CLI TREE -->
- [`mere.run guide`](/cookbooks) — Read offline mere.run command cookbooks.
- [`mere.run catalog`](/cli) — Inspect the machine-readable command capability contract.
- [`mere.run image`](/runtime/image) — Generate and validate image models.
  - `mere.run image dataset` — Inspect image training datasets.
    - `mere.run image dataset discover` — Find image-caption dataset candidates under a root directory.
  - `mere.run image generate` — Generate images with local image models.
  - `mere.run image reconstruct-3d` — Reconstruct a colored object mesh from one image with native TripoSR.
  - `mere.run image reconstruct-3d-trellis2` — Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2.
  - `mere.run image reconstruct-3d-multiview` — Reconstruct a colored mesh from 4 or 6 user-supplied views with native InstantMesh.
  - `mere.run image run-plan` — Run a saved image workflow plan.
  - `mere.run image train-lora` — Train a local image LoRA adapter.
  - `mere.run image visualize-run` — Open a local LoRA training run viewer.
  - `mere.run image validate` — Run advanced deterministic validation for local image model families.
- [`mere.run text`](/runtime/text) — Run local chat, code, embedding, and anonymization workflows.
  - `mere.run text chat` — Run local chat with text chat models.
  - `mere.run text code` — Run local code generation with GGUF models via llama.cpp.
  - `mere.run text embed` — Generate text embeddings using native Qwen3-Embedding-0.6B.
  - `mere.run text anonymize` — Detect and redact PII using OpenAI Privacy Filter.
  - `mere.run text train-lora` — Train a native text LoRA adapter from chat-style SFT JSONL.
- [`mere.run speech`](/runtime/speech) — Synthesize, transcribe, and manage voice profiles.
  - `mere.run speech synthesize` — Generate speech from text using Qwen3-TTS.
  - `mere.run speech transcribe` — Transcribe or translate speech to text using native ASR backends.
  - `mere.run speech listen` — Transcribe a macOS microphone with live Qwen ASR.
  - `mere.run speech profile` — Manage saved voice clone profiles.
    - `mere.run speech profile list` — List saved speech voice profiles.
    - `mere.run speech profile create` — Create a speech profile from reference audio.
    - `mere.run speech profile delete` — Delete a speech profile by id.
- [`mere.run vision`](/runtime/vision) — Caption, inspect, face-analyze, segment, track, pose, depth, geometry, optical flow, and OCR visual media.
  - `mere.run vision caption` — Generate training-friendly captions for images.
  - `mere.run vision inspect` — Describe or answer questions about an image using Qwen3-VL.
  - `mere.run vision face` — Detect faces, create identity embeddings, and compare faces locally.
    - `mere.run vision face detect` — Detect faces and five-point landmarks in an image.
    - `mere.run vision face embed` — Create a normalized ArcFace embedding for one face in an image.
    - `mere.run vision face compare` — Compare one face from each of two images with cosine similarity.
    - `mere.run vision face batch` — Analyze many images with one warm detector/recognizer session and emit JSONL.
  - `mere.run vision ground` — Ground text queries in an image with the native Falcon Perception runtime.
  - `mere.run vision segment` — Segment prompted objects in an image with the native SAM 3.1 runtime.
  - `mere.run vision track` — Track prompted objects through a video with the native SAM 3.1 runtime.
  - `mere.run vision track-live` — Capture from a camera and track text-prompted objects with the native SAM 3.1 runtime.
  - `mere.run vision pose` — Detect body, hand, and face landmarks in an image with the native platform runtime.
  - `mere.run vision flow` — Generate dense optical flow between two equal-size images.
  - `mere.run vision depth-video` — Generate temporally consistent relative or metric video depth with native VDA-S.
  - `mere.run vision geometry` — Generate metric depth, normals, camera intrinsics, and a point cloud with native MoGe-2.
  - `mere.run vision geometry-multiview` — Solve native DA3-Small multi-view relative geometry, confidence, and cameras.
  - `mere.run vision image-to-3d` — Reconstruct a colored object mesh from one image with native TripoSR.
  - `mere.run vision image-to-3d-trellis2` — Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2.
  - `mere.run vision image-to-3d-multiview` — VFX alias for native 4/6-view InstantMesh reconstruction.
  - `mere.run vision ocr` — Extract text from images using LightOnOCR, GLM-OCR, or Infinity-Parser2.
- [`mere.run music`](/runtime/music) — Generate music locally.
  - `mere.run music analyze` — Analyze source audio with ACE-Step audio understanding.
  - `mere.run music generate` — Generate audio from a music prompt.
  - `mere.run music realtime` — Run Magenta RealTime 2 music generation.
  - `mere.run music serve` — Start a warm resident ACE-Step music generation API.
  - `mere.run music train-adapter` — Train a native ACE-Step LoRA or LoKr adapter.
  - `mere.run music transcribe` — Transcribe a full music mix into instrument-separated MIDI with MuScriptor.
- [`mere.run sfx`](/runtime/sfx) — Generate sound effects locally.
  - `mere.run sfx ae` — Encode or decode Woosh audio autoencoder latents.
    - `mere.run sfx ae encode` — Encode audio into normalized Woosh-AE latents.
    - `mere.run sfx ae decode` — Decode normalized Woosh-AE latents into audio.
  - `mere.run sfx clap` — Embed and score sound effects with Woosh-CLAP.
    - `mere.run sfx clap score` — Score a text prompt against an audio file.
  - `mere.run sfx condition` — Export Woosh conditioning tensors.
    - `mere.run sfx condition text` — Encode a prompt with the Woosh text conditioner.
  - `mere.run sfx generate` — Generate a sound effect from a text prompt.
  - `mere.run sfx video` — Generate sound effects from video conditioning.
    - `mere.run sfx video generate` — Generate an 8-second sound effect from a video or Synchformer features.
- [`mere.run video`](/runtime/video) — Generate and understand video with native Swift/MLX pipelines.
  - `mere.run video animate` — Animate or replace a masked subject with native Swift/MLX SCAIL-2.
  - `mere.run video cosmos3` — Run native NVIDIA Cosmos3-Edge generation and action modes.
  - `mere.run video export-latents` — Run native Swift/MLX distilled LTX denoising and export final latents.
  - `mere.run video generate` — Generate MP4 video with native Swift/MLX video models.
  - `mere.run video prepare-masks` — Prepare reviewable, palette-safe SCAIL-2 masks with native SAM 3.1.
  - `mere.run video session` — Keep an LTX 2.3 runtime resident for JSONL generation requests.
- [`mere.run world`](/runtime/world) — Run persistent local conditioned-video world sessions.
  - `mere.run world serve` — Serve one warm native world-model session over HTTP.
- [`mere.run graph`](/workflows) — Validate, materialize, run, and submit portable workflow graphs.
  - `mere.run graph catalog` — List registered workflow node contracts.
  - `mere.run graph dataset` — Discover graph-ready datasets without loading model runtimes.
    - `mere.run graph dataset discover` — Find and rank image-caption dataset directories under a root.
  - `mere.run graph validate` — Validate a typed workflow graph without executing it.
  - `mere.run graph preflight` — Preflight a workflow graph against an executor.
  - `mere.run graph materialize` — Create an immutable workflow job bundle in a run directory.
  - `mere.run graph export-job` — Export an immutable workflow job bundle for any executor.
  - `mere.run graph run` — Run a workflow graph locally.
  - `mere.run graph run-job` — Run an existing immutable workflow job bundle locally.
  - `mere.run graph submit` — Submit a portable workflow job to an SSH or relay executor.
  - `mere.run graph submit-job` — Submit an existing immutable workflow job bundle to SSH or Relay.
  - `mere.run graph worker` — Run the portable workflow worker protocol.
    - `mere.run graph worker probe` — Report worker capabilities.
    - `mere.run graph worker execute` — Execute a materialized job bundle.
    - `mere.run graph worker inspect` — Inspect a worker run manifest.
    - `mere.run graph worker cancel` — Request cooperative worker cancellation.
- [`mere.run executor`](/workflows#executor-profiles) — Manage local, SSH, and relay workflow executors.
  - `mere.run executor add` — Add an executor profile.
    - `mere.run executor add ssh` — Add an SSH executor.
    - `mere.run executor add relay` — Add a relay executor.
  - `mere.run executor list` — List executor profiles.
  - `mere.run executor inspect` — Inspect an executor profile.
  - `mere.run executor probe` — Probe executor capabilities.
  - `mere.run executor login` — Sign in to a relay executor through its advertised device authorization flow.
  - `mere.run executor auth-status` — Inspect relay authentication without printing credentials.
  - `mere.run executor logout` — Remove the saved credential for a relay executor.
  - `mere.run executor fleet` — Inspect relay node identity, eligibility, inventory, and recent work.
  - `mere.run executor node-refresh` — Ask a connected relay node to rescan its capabilities and installed models.
  - `mere.run executor node-configure` — Update relay scheduling policy for a fleet node.
  - `mere.run executor remove` — Remove an executor profile.
- [`mere.run run`](/workflows#run-directories) — Inspect durable mere.run workflow reports and run directories.
  - `mere.run run list` — Find durable run directories, structured reports, and run plans under a root.
  - `mere.run run inspect` — Inspect a run directory, structured report, or run plan.
  - `mere.run run watch` — Watch the worker event stream for an SSH or relay graph job.
  - `mere.run run fetch` — Fetch a remote graph run into the standard local run-directory format.
  - `mere.run run cancel` — Cancel a graph run.
  - `mere.run run retry` — Retry an immutable relay graph job.
- [`mere.run model`](/runtime/model-management) — List, pull, remove, inspect, and clean up models.
  - `mere.run model list` — List all known models with install status.
  - `mere.run model pull` — Download a managed model into the local model store.
  - `mere.run model remove` — Remove a model from the local model store.
  - `mere.run model storage` — Inspect physical model storage, sharing, and reclaimable space.
  - `mere.run model gc` — Find or delete unreferenced model payloads and partial downloads.
  - `mere.run model info` — Print a model's manifest, validation status, and resolved component paths.
  - `mere.run model capabilities` — Show which managed models this machine can run.
  - `mere.run model runtime` — Read and update per-model API runtime settings.
    - `mere.run model runtime get` — Print typed API runtime settings for a managed model.
    - `mere.run model runtime set` — Update typed API runtime settings for a managed model.
  - `mere.run model benchmark` — Run focused local model benchmarks.
    - `mere.run model benchmark chat` — Run a small grounded-chat eval slice against local assistant models.
    - `mere.run model benchmark tool-calls` — Run a small tool-call selection eval against local chat models.
    - `mere.run model benchmark tool-continuations` — Evaluate Gemma 4 continuation after completed tool calls.
    - `mere.run model benchmark code` — Run a real coding-eval slice against local coding models.
    - `mere.run model benchmark gemma4-kv` — Compare Gemma4 default KV cache decode against packed PolarKV.
    - `mere.run model benchmark gemma4-mtp` — Compare Gemma4 serial decode against verified MTP speculative decode.
    - `mere.run model benchmark q36-mtp` — Compare Qwen3.6 serial decode against adaptive and forced MTP speculative decode.
    - `mere.run model benchmark laguna-dflash` — Measure Laguna target-only and DFlash decode in one resident process.
    - `mere.run model benchmark api-workload` — Replay a chat workload against a running API server and measure runtime cache counters.
    - `mere.run model benchmark vlm` — Compare vision-language chat models on synthetic or lmms-eval datasets.
  - `mere.run model repair-manifests` — Write missing mererun_model.json for known models in the local mere.run model store.
- [`mere.run adapter`](/runtime/model-management) — List and pull verified LoRA adapters.
  - `mere.run adapter list` — List cataloged LoRA adapters and their install state.
  - `mere.run adapter pull` — Download and verify a cataloged LoRA adapter.
- [`mere.run status`](/runtime/model-management) — Show local server, loaded model, and installed model status.
- [`mere.run gate`](/gate) — Run the end-to-end quality gate against installed models.
- [`mere.run config`](/configuration) — Get and set persisted mere.run configuration (e.g. Hugging Face token).
  - `mere.run config set` — Set a config value.
  - `mere.run config get` — Print a config value (secrets masked).
  - `mere.run config unset` — Remove a config value.
  - `mere.run config list` — Show all config values (secrets masked).
  - `mere.run config path` — Print the config file path.
- [`mere.run api`](/runtime/api-server) — Serve local models through API surfaces.
  - `mere.run api serve` — Start an OpenAI-compatible API server for local chat and embedding models.
- [`mere.run open-webui`](/runtime/api-server#open-webui-companion) — Start the optional Open WebUI companion against a local mere.run API.
  - `mere.run open-webui quickstart` — Start mere.run, run the official Open WebUI Docker image, and configure the connection.
- [`mere.run plugin`](/plugins) — Discover and install official mere.run companion plugins.
  - `mere.run plugin list` — List official plugins from the live catalog.
  - `mere.run plugin info` — Show one plugin's catalog entry and install command.
  - `mere.run plugin install` — Install one official plugin using its catalog install command.
  - `mere.run plugin doctor` — Run an installed plugin's doctor command.
- [`mere.run setup`](/getting-started) — Choose a guided, BYOA, or manual mere.run setup path.
- [`mere.run agent`](/getting-started) — Install and start the optional guided local setup agent.
  - `mere.run agent onboard` — Summarize this machine's model capabilities and prepare the optional Pi agent.
  - `mere.run agent install-pi` — Install the latest Pi coding-agent release for use with mere.run.
  - `mere.run agent start` — Start Pi against a local mere.run setup-agent API server.
<!-- END GENERATED: CLI TREE -->

## Global model-store override

The CLI honors the shared models root override:

```bash
swift run mere.run --models-root /Volumes/FastSSD/mererun-models model list
```

That is equivalent to setting:

```bash
export MERERUN_MODELS_DIR=/Volumes/FastSSD/mererun-models
```

## Canonical managed model IDs

See [`model-sources.md`](./model-sources.md) for the full source story,
including which IDs are pullable from Hugging Face. The most common managed IDs
are:

- Images: `image-klein-nano`, `image-klein-base`, `image-klein-base-9b`, `image-klein-max`,
  `image-bonsai-binary`, `image-bonsai-ternary`, `image-zimage-nano`, `image-zimage-base`, `image-zimage-max`,
  `image-hidream-o1`, `image-hidream-o1-dev`, `image-krea2-raw`,
  `image-krea2-turbo`,
  `image-ideogram4-sdnq-uint4`
- Text chat: `text-chat-gemma4`, `text-chat-mebot`, `text-chat-psi-agent`, `text-chat-q36-nano`, `text-chat-lfm25-a1b-8bit`
- Text code / agents: `text-agent-qwen35-9b`, `text-agent-ornith-9b`, `text-agent-ornith-35b-mlx`, `text-agent-ornith-35b`, `text-code-north-mini`, `text-code-qwen3`
- Text embed: `text-embed-qwen3-0.6b`
- Text anonymize: `text-anonymize-privacy-filter`
- Speech TTS: `speech-tts-qwen3-nano`, `speech-tts-qwen3-customvoice`
- Speech ASR: `speech-asr-qwen3`, `speech-asr-parakeet`
- Vision OCR: `vision-ocr-lighton`, `vision-ocr-infinity-flash`,
  `vision-ocr-infinity-pro-int8`, `vision-ocr-infinity-pro`
- Vision segmentation / tracking: `vision-segment-sam31`
- Vision grounding: `vision-ground-falcon-perception`
- Face detection and identity embeddings: `vision-face-buffalo-l`
- Music: `music-acestep`, `music-acestep-xl-turbo`, `music-acestep-xl-turbo-lm4b`, `music-acestep-xl-sft`, `music-acestep-xl-base`, `music-magenta-rt2-small`, `music-magenta-rt2-base`
- SFX: `sfx-woosh-dflow`, `sfx-woosh-flow`
- Video: `video-ltx-av`, `video-ltx23-av-mlx`, `video-ltx23-full-mlx`, `video-ltx23-a2vid-mlx`, `video-wan22-ti2v-5b-mlx`, `video-scail2-14b-mlx`

For subsystem-specific implementation guides, see:

- [Image Runtime](./runtime/image.md)
- [Text Runtime](./runtime/text.md)
- [Speech Runtime](./runtime/speech.md)
- [Vision Runtime](./runtime/vision.md)
- [Music Runtime](./runtime/music.md)
- [SFX Runtime](./runtime/sfx.md)
- [Video Runtime](./runtime/video.md)

## Common workflows

### Pull and inspect models

```bash
swift run mere.run model list
swift run mere.run status
swift run mere.run model capabilities
swift run mere.run model runtime get text-chat-gemma4
swift run mere.run model pull image-zimage-nano --accept-model-license
swift run mere.run model info image-zimage-nano
```

### Generate an image

```bash
swift run mere.run image generate \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png
```

### Chat locally

```bash
swift run mere.run text chat \
  --prompt "Explain classifier-free guidance."
```

### Generate speech and transcribe it back

```bash
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --output ./hello.wav

swift run mere.run speech transcribe ./hello.wav --backend auto
```

For live microphone transcription on macOS:

```bash
swift run mere.run speech listen --list-devices
swift run mere.run speech listen --device <core-audio-uid>
```

### Inspect, segment, track, and OCR

```bash
swift run mere.run vision inspect ./diagram.png "What does this diagram show?"
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
swift run mere.run vision track ./clip.mp4 --prompt "a cat"
swift run mere.run vision ocr ./page.png --backend lighton
```

### Generate music

```bash
swift run mere.run model pull music-acestep
swift run mere.run music generate \
  "upbeat electronic groove" \
  --output ./track.wav

swift run mere.run model pull music-acestep-xl-turbo
swift run mere.run music generate \
  "upbeat electronic groove" \
  --model music-acestep-xl-turbo \
  --output ./xl-track.wav

swift run mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-turbo-lm4b \
  --lm-subdirectory acestep-5Hz-lm-4B

swift run mere.run model pull music-magenta-rt2-small
swift run mere.run music realtime \
  "ambient modular synths with brushed drums" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./live.wav \
  --no-play
```

### Generate sound effects

```bash
swift run mere.run model pull sfx-woosh-dflow --accept-model-license
swift run mere.run sfx generate \
  "metal wrench dropping onto concrete, bright clang and brief ring" \
  --model sfx-woosh-dflow \
  --duration 5 \
  --output ./wrench-clang.wav
```

### Generate video

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --quality final \
  --duration 5 \
  --output ./clip.mp4
```

### Run a portable workflow

```bash
mere.run graph validate workflow.json --inputs-json inputs.json --json
mere.run graph preflight workflow.json --inputs-json inputs.json --executor local --json
mere.run graph run workflow.json --inputs-json inputs.json --run-dir ./runs/local
mere.run run inspect ./runs/local --json
```

The same immutable job bundle can run through configured SSH or relay
executors. See [Portable Workflows](./workflows.md).

### Start a persistent world session

```bash
mere.run world serve \
  --base-model video-wan22-ti2v-5b-mlx \
  --model video-dreamx-world-5b-ar-mlx \
  --prepare
```

See [Persistent World Runtime](./runtime/world.md) for the HTTP lifecycle,
camera controls, state artifacts, and authentication boundary.

### Serve a local API

```bash
swift run mere.run api serve --engine text-chat-gemma4
```

### Install a companion plugin

Official companion plugins are distributed outside the Swift package. The CLI
reads the live catalog from `sawfwair/mere-run-plugins`, prints the exact install
command by default, and only executes it when `--yes` is present.

```bash
swift run mere.run plugin list
swift run mere.run plugin info mere-runpod
swift run mere.run plugin install mere-runpod
swift run mere.run plugin install mere-runpod --yes
swift run mere.run plugin doctor mere-runpod
```

See [Companion Plugins](./plugins.md) for installation verification and the
typed graph-provider boundary.

## Command reference

Model installation in the OSS repo is explicit. `mere.run model pull` uses cataloged Hugging Face snapshots only; local-path-only models must be supplied with command-specific `--model` or `--model-root` options. See [`configuration.md`](./configuration.md) and [`model-sources.md`](./model-sources.md).

Public LoRA releases use the separate checksum-pinned adapter catalog:

```bash
mere.run adapter list
mere.run adapter pull mere-platform-assistant
mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora mere-platform-assistant \
  --prompt "Summarize the current Mere workspace."
```

### `mere.run plugin`

Discover and install official companion plugins from the live plugin catalog.
Plugins are separate executables, not code loaded into the `mere.run` process.

```bash
swift run mere.run plugin list
swift run mere.run plugin info mere-runpod
swift run mere.run plugin install mere-runpod --yes
swift run mere.run plugin doctor mere-runpod
```

Key options:

- `--catalog-url`: plugin catalog URL or local JSON path
- `--json`: emit catalog or plugin metadata as JSON
- `--channel`: install channel, defaulting to the catalog default
- `--yes`: execute the install command; omitted means dry-run preview
- `--force`: pass `--force` to `pipx install`

### `mere.run image generate`

Generate a PNG with a local image model.

```bash
swift run mere.run image generate --prompt "<text>" [options]
```

Key options:

- `--prompt`: required text prompt
- `--model`: canonical model id or local model path
- `--output`: output PNG path
- `--width`, `--height`
- `--steps`: override the model-specific step default
- `--cfg`: override the model-specific CFG default
- `--input`: image-to-image source. For FLUX.2 Klein, this is treated as a
  single reference image.
- `--ref-image`: repeatable FLUX.2 Klein or HiDream O1 reference image
- `--keep-original-aspect`: preserve one HiDream reference image's aspect ratio
- `--strength`: image-to-image/reference change strength
- `--structured-prompt`, `--json-prompt`: expand the prompt into a structured JSON caption with a local text chat model before image generation
- `--structured-prompt-model`: text chat model id for the adapter; defaults to `text-chat-gemma4-12b-4bit`
- `--structured-prompt-output`: write the generated structured JSON caption to a file
- `--lora`, `--lora-scale`
- `--krea-base-quantization-bits 4|8`: quantize the frozen Krea transformer and
  load the text encoder, transformer, and VAE sequentially to reduce peak memory
- `--preflight`: inspect the generation request without loading the model or writing an image
- `--json`: with `--preflight`, emit a structured JSON report
- `--quiet`
- `--progress-json`: stream progress to stderr as JSON lines instead of
  human-readable text, one object per event, e.g.
  `{"event":"progress","stage":"denoising","step":2,"total_steps":4}`.
  `step` is the generator's 0-based step index; each stage emits a final event
  with `step == total_steps`. Takes precedence over `--quiet` for progress
  output, so wrappers can keep stdout limited to the output path while still
  observing per-step progress.

Unless `--quiet` is set, progress diagnostics on stderr include the native image
backend, for example `native MLX/Metal (default device: gpu)` on Apple Silicon.
Use `--preflight --json` when scripts or wrappers need model/input/output
checks and declarative actions before starting generation. The report includes
structured diagnostics plus actions such as `start-generation`, `pull-model`,
and `open-output-directory`; hard blockers exit nonzero after printing JSON.
It also includes `result.run_plan`, which can be saved and replayed without
reconstructing CLI arguments. Save that object to a file and use
`image run-plan --materialize` when a wrapper needs a durable run directory
before generation starts.

Examples:

```bash
swift run mere.run image generate --prompt "a black cat on a red sofa"
swift run mere.run image generate --prompt "a black cat on a red sofa" --preflight --json
swift run mere.run image generate --model image-zimage-nano --prompt "retro robot illustration" --output ./robot.png
swift run mere.run image generate --model image-bonsai-binary --prompt "sunlit greenhouse bonsai" --output ./bonsai.png
swift run mere.run image generate --model image-krea2-turbo --prompt "translucent portable speaker product photo" --steps 8 --krea-base-quantization-bits 4 --output ./speaker.png
swift run mere.run image generate --prompt "turn this into a pencil sketch" --input ./photo.png --strength 0.6
swift run mere.run image generate --model image-ideogram4-sdnq-uint4 --prompt "a knight and a white horse in a sunny meadow" --structured-prompt --structured-prompt-output ./knight-prompt.json --output ./knight.png
swift run mere.run image generate \
  --model image-hidream-o1-dev \
  --prompt "put this subject in a studio portrait" \
  --ref-image ./subject.png \
  --output ./portrait.png
swift run mere.run image generate \
  --model image-klein-base \
  --prompt "a trading card character with the same sticker layout" \
  --ref-image ./card-reference.png \
  --output ./card.png

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
```

### `mere.run image train-lora`

Train a local text-to-image LoRA adapter. Krea 2 LoRAs are trained on
`image-krea2-raw` and can be used with `image-krea2-turbo` for fast inference.
FLUX.2 Klein LoRAs are trained on a Klein base model selected with `--model`
and can be used with Klein generation via `image generate --lora`.

```bash
swift run mere.run model pull image-krea2-raw --accept-model-license
swift run mere.run model pull image-krea2-turbo --accept-model-license
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-krea2.safetensors \
  --recipe krea-cinematic-style \
  --quiet
swift run mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a studio portrait in the trained style" \
  --lora ./style-krea2.safetensors \
  --output ./style-preview.png
```

For quick Krea smoke runs, `krea-fast-style` trains on `image-krea2-raw` for
100 steps with LR `0.0005`, 10-step warmup/cosine decay, 768 square, rank 32,
and alpha 32. Treat it as a proof pass and inspect images before trusting the
adapter. For stronger widescreen style datasets, `krea-cinematic-style` uses
200 steps, LR `0.0001`, 20-step warmup/cosine decay, 768x416, rank 32, alpha
32, and compiled-step disablement; override `--width`/`--height` for other
source aspects.

For practical Klein Base 9B style training, the `klein-fast-style` recipe uses
the undistilled BF16 `image-klein-base-9b` model, 1000 steps, LR `0.00005`,
max side `512`, the fast Klein target surface, disk-backed latent caching,
compiled-step disablement, and 250-step checkpoints:

```bash
swift run mere.run model pull image-klein-base-9b --accept-model-license
swift run mere.run model pull image-klein-9b --accept-model-license
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --visualize \
  --quiet
swift run mere.run image generate \
  --model image-klein-9b \
  --prompt "TRIGGER_TOKEN a studio portrait in the trained style" \
  --lora ./style-klein.safetensors \
  --lora-scale 2.0 \
  --output ./style-preview.png
```

Dataset folders use image files with matching `.txt` captions:

```text
style-dataset/
  001.png
  001.txt
  002.jpg
  002.txt
```

If you have a project data root with many nested artifacts, discover trainable
dataset leaves before preflighting a training run:

```bash
swift run mere.run image dataset discover \
  --root ./project-data \
  --json

swift run mere.run image dataset discover \
  --root ./project-data \
  --training-output-root ./lora-output \
  --training-recipe klein-fast-style \
  --json
```

The discovery report scans child directories, reports image/caption counts,
marks trainable candidates, and returns a `choose-dataset` action for wrappers
that want to turn a root folder into a concrete dataset selection. Each
candidate includes patches for `request.data` and `run_plan.arguments.data`.
With `--training-output-root`, candidates also include an `image train-lora
--preflight --json` command using a deterministic `.safetensors` path under
that output root. Add `--training-model` or `--training-recipe` to include those
options in the emitted per-candidate commands.

Before starting a real LoRA run, use preflight JSON to catch cheap blockers
without loading the model or allocating the training runtime:

```bash
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --preflight \
  --json
```

The preflight report writes one structured JSON document to stdout. It includes
dataset counts, model readiness, output-path checks, diagnostics, and
declarative actions such as `start-training` or `pull-model`. It also includes
`result.run_plan`, a normalized executable plan that wrappers can save and hand
back to the CLI later. Hard blockers exit nonzero after printing the JSON report
so scripts and agents can inspect the same payload humans do.

To replay a saved plan, write `result.run_plan` to disk and run:

```bash
swift run mere.run image run-plan ./style.plan.json --preflight --json
swift run mere.run image run-plan ./style.plan.json
```

To create a durable run directory before training, materialize the plan:

```bash
swift run mere.run image run-plan ./style.plan.json \
  --materialize ./runs/style \
  --json
```

Materialization writes `plan.json`, `actions.json`, `run.json`, and an initial
events file into the run directory. The copied `plan.json` relocates the output
adapter path into that directory so later training artifacts land beside it.
Running the materialized `plan.json` appends `run_started`, progress, and final
status events to the same event stream.

Key options:

- `--data`: dataset directory with image + caption pairs
- `--output`: output `.safetensors` adapter path
- `--model`: Raw/base model id or local model path; defaults to `image-krea2-raw`
- `--width`, `--height`: fixed training resolution; must be divisible by 16
- `--training-steps`, `--steps`
- `--batch-size`
- `--learning-rate`, `--lr`
- `--rank`, `--alpha`
- `--max-text-length`
- `--scheduler-steps`
- `--caption-dropout`
- `--recipe`: named training recipe: `krea-fast-style`, `krea-cinematic-style`,
  or `klein-fast-style`
- `--lr-warmup-steps`, `--no-cosine-scheduler`, `--lr-min-factor`: Krea/Klein
  LR scheduler controls
- `--lite`: train only attention Q/V layers to reduce memory
- `--base-quantization-bits 4|8`: quantize the frozen Krea base transformer while training
- `--exclude-preview-images`
- `--checkpoint-interval`: save intermediate Klein LoRA adapters every N steps
- `--max-resolution`: preserve source aspect ratio up to a maximum side length
- `--low-ram`: use the Klein disk-backed latent cache
- `--no-compile`: disable compiled train-step graphs
- `--lora-target-preset`: exact Klein target preset, including `fal-klein-fast`
- `--lora-target-mode`: for FLUX.2 Klein, `suffix` or `transformer-linear-walk`
- `--preflight`: inspect the training request without running training
- `--json`: with `--preflight`, emit a structured JSON report
- `--visualize`: start a loopback LoRA training dashboard for the run
- `--visualize-port`: port for the local dashboard; defaults to `8787`
- `--quiet`

### `mere.run image run-plan`

Run or preflight a saved image workflow plan. Supported plan kinds are
`image.generate`, emitted as `result.run_plan` from
`image generate --preflight --json`, and `image.train_lora`, emitted as
`result.run_plan` from `image train-lora --preflight --json`.

```bash
swift run mere.run image run-plan ./render.plan.json --preflight --json
swift run mere.run image run-plan ./render.plan.json
swift run mere.run image run-plan ./render.plan.json --materialize ./runs/render --json
swift run mere.run image run-plan ./style.plan.json --preflight --json
swift run mere.run image run-plan ./style.plan.json --materialize ./runs/style --json
swift run mere.run image run-plan ./style.plan.json
```

`--materialize` writes `plan.json`, `actions.json`, `run.json`, an initial
events file, and standard artifact directories before execution. For
`image.generate`, the copied plan relocates the output image and optional
structured-prompt sidecar into the run directory, and execution appends
`run_started`, `run_finished`, or `run_failed` events. For `image.train_lora`,
the same directory also carries checkpoints, samples, loss metrics, and adapter
artifacts.

### `mere.run run list`

Discover durable run directories, structured report JSON files, and saved
run-plan JSON files under a workspace root. This is the first headless browse
step for wrappers that do not already know the exact run/report/plan path.

```bash
swift run mere.run run list --root ./runs --json
swift run mere.run run list --root . --max-depth 5 --json
swift run mere.run run list --root ./render.plan.json --json
```

Each entry includes its kind, relative path, inspection status, run state where
available, created/updated timestamps where known, command/format details,
diagnostic counts, and declarative actions for `run inspect --json`. The
top-level report also includes an
`inspect-selected` action for UI pickers. Legacy/plugin run folders are listed
as warning-level entries when their `run.json` can be identified but is not a
native mere.run training manifest.

### `mere.run run inspect`

Inspect a durable run directory, structured report JSON file, or saved run-plan
JSON file. This is the headless readback path for wrappers that need to reopen
work after `run list`, `--preflight --json`, or `image run-plan --materialize`.

```bash
swift run mere.run run list --root ./runs --json
swift run mere.run run inspect ./runs/render --json
swift run mere.run run inspect ./runs/render
swift run mere.run run inspect ./render.plan.json --json
swift run mere.run run inspect ./preflight-report.json --json
```

For run directories, the JSON report includes `run.json` manifest details,
`actions.json` summaries, `*.events.jsonl` status, known artifact paths, and
follow-up actions such as opening the run directory or preflighting the saved
plan. For report files, it summarizes the original command, mode, status,
diagnostic count, and action count. For plan files, it reports the plan kind,
command, creation timestamp, and captured working directory. Legacy/plugin run
manifests are kept warning-level and expose `legacy_manifest` details such as
provider, GPU, dataset path/count, recipe id, command, and run status. Run
directories also include a compact `metrics` object with loss CSV summary,
latest loss, step range, sample image count, checkpoint count, and adapter
count.

### `mere.run image visualize-run`

Open the same local LoRA training dashboard for an existing run directory.
The viewer reads `run.json`, `*.loss.csv`, `*.events.jsonl`, samples,
checkpoints, and adapter artifacts from disk.

```bash
swift run mere.run image visualize-run ./runs/my-style --port 8787
```

### `mere.run image validate`

Run advanced deterministic validation for the local image families.

```bash
swift run mere.run image validate --family zimage --test all
swift run mere.run image validate --family klein --test vae --output ./validation_output
swift run mere.run image validate --save-reference
swift run mere.run image validate --compare --reference-dir ./validation_output
```

Key options:

- `--family`: `zimage` or `klein`
- `--test`: `vae`, `encoder`, `transformer`, `pipeline`, or `all`
- `--output`
- `--save-reference`
- `--compare`
- `--reference-dir`

### `mere.run text chat`

Run local text chat with the Gemma 4, Qwen3.6/Bonsai, LFM2, or Psi family.

```bash
swift run mere.run text chat --prompt "<text>" [options]
```

Key options:

- `--prompt`
- `--system`
- `--model`: canonical model id
- `--model-root`: explicit local model root
- `--max-tokens`
- `--context-size`: maximum prompt plus generation context. Bonsai 27B uses
  its published 262,144-token limit by default.
- `--temperature`: defaults to 0.7, or the model's published value where one
  exists (Bonsai: 0.7; Ornith lanes: 1.0)
- `--top-p`: defaults to 0.9, or the model's published value (Bonsai/Ornith: 0.95)
- `--top-k`: defaults to no cutoff, or the model's published value (Bonsai/Ornith: 20)
- `--min-p`: relative probability floor from 0 through 1; `0` disables it.
  For example, `0.05` removes tokens below 5% of the leading token's
  probability. It does not change greedy generation.
- `--kv-bits`: native Qwen-family models accept affine 4-bit or 8-bit resident
  KV caches. Gemma4 also supports its model-specific cache schemes.
- `--response-format text|json_object`: require a complete JSON object from a
  native MLX Gemma or Qwen-family model. JSON mode forces thinking off and
  validates each token before streaming.
- `--thinking` / `--no-thinking`: show reasoning output / disable reasoning
  generation. R1-style lanes (`text-agent-ornith-*`) generate with thinking
  enabled by default even though the reasoning stays hidden. Bonsai 27B also
  defaults to thinking-enabled generation.
- `--stream`
- `--stats`: includes Gemma4 MTP state and accept/draft counts when a Gemma4 model is used
- `--quiet`

Unless `--quiet` is set, diagnostics on stderr include the selected text
backend, for example native MLX/Metal for MLX models or llama.cpp/GGUF for GGUF
models.

Examples:

```bash
swift run mere.run text chat --prompt "What is classifier-free guidance?"
swift run mere.run text chat --model text-chat-bonsai-27b-1bit --context-size 262144 --kv-bits 4 --prompt "Plan a long-context repository review."
swift run mere.run text chat --model text-chat-bonsai-27b-2bit --context-size 262144 --kv-bits 4 --prompt "Compare two repository migration plans."
swift run mere.run text chat --model text-chat-q36-nano --prompt "Explain speculative decoding."
swift run mere.run text chat --model text-chat-q36-nano --response-format json_object --prompt 'Return an object with a name and an array of tags.'
swift run mere.run text chat --model text-agent-ornith-9b --prompt "Write a compact Swift slugify helper."
swift run mere.run text chat --model text-chat-lfm25-a1b-8bit --prompt "Summarize LFM2 in one paragraph."
swift run mere.run text chat --stream --prompt "Write a short welcome message."
swift run mere.run text chat --thinking --stats --prompt "How would you design a tokenizer?"
```

`json_object` supports nested objects and arrays, Unicode and escaped strings,
numbers, booleans, and null. It is not JSON Schema or strict structured output.
The llama.cpp/GGUF Q36 lane used by Linux packages does not yet have a JSON
grammar and rejects `--response-format json_object`; use native MLX Q36 on
Apple Silicon for this release.

### `mere.run text code`

Run local code generation with GGUF models through the vendored `llama.cpp` runtime.

```bash
swift run mere.run text code --prompt "<text>" [options]
```

Key options:

- `--prompt`
- `--model`: GGUF file or canonical code model id if your local setup resolves it
- `--stream`
- `--stats`
- `--temperature`
- `--top-p`
- `--min-p`
- `--max-tokens`

Examples:

```bash
swift run mere.run text code --prompt "Write a Swift function to reverse a string"
swift run mere.run text code --model ./Qwen3-Coder-Next-Q4_K_M.gguf --stream --prompt "Implement a trie in Rust"
```

North Mini Code is managed as `text-code-north-mini` for local coding-agent
comparisons. It pulls the Unsloth GGUF quant and runs through the same native
Swift/llama.cpp code path as Qwen:

```bash
swift run mere.run model pull text-code-north-mini
swift run mere.run text code --model text-code-north-mini --prompt "Sketch a small Swift Result helper."
```

Ornith 35B is managed as `text-agent-ornith-35b` for larger local coding-agent
comparisons. It pulls DeepReinforce's Q4_K_M GGUF quant and runs through the
same native Swift/llama.cpp code path:

```bash
swift run mere.run model pull text-agent-ornith-35b
swift run mere.run text code --model text-agent-ornith-35b --prompt "Sketch a small Swift Result helper."
```

`text-agent-ornith-35b-mlx` is the native Swift/MLX lane for a locally converted
Ornith 1.0 35B Q4 directory. It is intentionally local-only until a converted
MLX snapshot is published:

```bash
swift run mere.run text chat --model text-agent-ornith-35b-mlx --prompt "Sketch a small Swift Result helper."
```

### `mere.run text embed`

Generate embeddings with the native Qwen3 embedding model.

```bash
swift run mere.run text embed "semantic search query"
swift run mere.run text embed "foo" "bar" --output embeddings.json --pretty
```

Key options:

- positional text arguments
- `--model`
- `--max-tokens`
- `--output`
- `--pretty`

### `mere.run text anonymize`

Detect and redact PII with the native OpenAI Privacy Filter model.

```bash
swift run mere.run text anonymize "My name is Alice Smith and my email is alice@example.com"
swift run mere.run text anonymize --json --pretty "Phone: 555-1234"
cat notes.txt | swift run mere.run text anonymize --output redacted.txt
```

Key options:

- positional text arguments, or stdin when omitted
- `--model`
- `--max-tokens`
- `--replacement`: template supporting `{label}` and `{index}`
- `--json`
- `--output`
- `--pretty`

### `mere.run speech synthesize`

Generate speech from text with Qwen3-TTS.

```bash
swift run mere.run speech synthesize "<text>" --output ./speech.wav [options]
```

Key options:

- `--output`: required
- `--model`: canonical speech TTS id or local model path
- `--voice`
- `--mode`: `style` or `clone`
- `--profile`
- `--ref-audio`
- `--ref-text`
- `--language`
- `--save-profile`
- `--temperature`
- `--stream`
- `--stream-chunk-tokens`
- `--quiet`

Examples:

```bash
swift run mere.run speech synthesize "Hello from mere.run" --output ./hello.wav
swift run mere.run speech synthesize "Welcome aboard" --voice "A calm British male voice" --output ./welcome.wav
swift run mere.run speech synthesize "Read this in my cloned voice" --mode clone --profile my-voice --output ./clone.wav
```

### `mere.run speech transcribe`

Transcribe or translate local audio with the speech backends.

```bash
swift run mere.run speech transcribe <audio.wav> [options]
```

Key options:

- positional audio path
- `--backend`: `auto`, `qwen`, or `parakeet`
- `--task`: `transcribe` or `translate`
- `--model`
- `--language`
- `--max-tokens`
- `--stream`
- `--stream-chunk-ms`
- `--stream-decode-ms`
- `--input-format pcm-s16le` (raw stdin only)
- `--sample-rate 16000` (raw stdin only)
- `--jsonl` (raw stdin only)
- `--no-timestamps`
- `--output`
- `--quiet`

Examples:

```bash
swift run mere.run speech transcribe ./audio.wav
swift run mere.run speech transcribe ./audio.wav --task translate --backend qwen
swift run mere.run speech transcribe ./audio.wav --stream --output ./transcript.txt
producer | swift run mere.run speech transcribe - \
  --stream --input-format pcm-s16le --sample-rate 16000 --jsonl
```

Raw streaming stdin is protocol v1: mono signed 16-bit little-endian PCM at
16 kHz. JSONL stdout begins with `ready`, followed by versioned `partial`,
`commit`, and `stats` events, and ends with exactly one terminal `final` on
graceful EOF. Diagnostics remain on stderr.

### `mere.run speech listen`

Capture a macOS input device and transcribe it in real time with Qwen.

```bash
swift run mere.run speech listen [options]
```

Use `--list-devices` to print stable CoreAudio UIDs and `--device <uid>` to
select one. The system input is the default. `--decode-ms` defaults to 2000,
`--silence-ms` defaults to 900, and `--jsonl` emits the same protocol-v1 event
stream as raw stdin. Ctrl-C gracefully flushes the active utterance.

### `mere.run speech profile`

Manage reusable voice clone profiles.

Subcommands:

- `mere.run speech profile list`
- `mere.run speech profile create`
- `mere.run speech profile delete`

Examples:

```bash
swift run mere.run speech profile list
swift run mere.run speech profile create \
  --name narrator \
  --audio ./ref.wav \
  --text "reference transcript"
swift run mere.run speech profile delete --id <uuid>
```

### `mere.run vision caption`

Generate captions for one or more images.

```bash
swift run mere.run vision caption ./images/*.png
swift run mere.run vision caption ./images/*.png --output-dir ./captions
swift run mere.run vision caption ./cards/*.jpg \
  --output-dir ./captions \
  --prompt-file ./card-caption-prompt.txt \
  --focus "full card border" "printed title text" \
  --trigger-token cardstyle
```

- `--prompt`: caption instruction
- `--prompt-file`: read reusable caption instructions from a UTF-8 text file
- `--focus`: visible details the captioner should prioritize
- `--trigger-token`: prefix each saved caption with an exact LoRA trigger token

### `mere.run vision inspect`

Ask a direct question about an image.

```bash
swift run mere.run vision inspect ./diagram.png "What does this diagram show?"
```

### `mere.run vision segment`

Segment prompted objects in an image using the native SAM 3.1 runtime.

```bash
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
```

Key options:

- `--prompt`: one or more text object prompts
- `--box`: one or more `x1,y1,x2,y2[,label]` geometry prompts
- `--point`: one or more `x,y,positive[,label]` or `x,y,negative[,label]` geometry prompts
- `--model`: managed model id or local SAM 3.1 model root
- `--output`: annotated image path
- `--json-output`: metadata path
- `--mask-output-dir`: optional per-object mask export directory
- `--threshold`: score cutoff, default `0.05`
- `--resolution`
- `--show-boxes`
- `--multimask`: emit up to three candidates per geometry-prompted object

Defaults:

- annotated image: `<image-stem>_segmented.<ext>`
- JSON metadata: `<image-stem>_segmented.json`

Notes:

- still-image runs accept text, box, and point prompts in the same invocation
- `--mask-output-dir` writes one PNG mask per exported detection candidate
- empty detection sets still produce annotated output plus JSON metadata

Examples:

```bash
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
swift run mere.run vision segment ./photo.jpg --prompt "a person" "a phone" --show-boxes
swift run mere.run vision segment ./photo.jpg --box "120,80,420,760,person" --mask-output-dir ./masks
swift run mere.run vision segment ./photo.jpg --point "512,384,positive,person" --point "700,200,negative,person"
swift run mere.run vision segment ./photo.jpg --prompt "a dog" --output ./photo-segmented.png --json-output ./photo-segmented.json
```

### `mere.run vision track`

Track prompted objects through a video with the native SAM 3.1 runtime.

```bash
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision track ./clip.mp4 --prompt "a dog"
```

Key options:

- `--prompt`: one or more text prompts used to seed objects on the init frame
- `--box`: one or more `x1,y1,x2,y2[,label]` geometry prompts
- `--point`: one or more `x,y,positive[,label]` or `x,y,negative[,label]` geometry prompts
- `--init-frame`: starting frame index for seeding
- `--end-frame`: optional inclusive final frame index
- `--output`: annotated video path
- `--json-output`: tracking metadata path
- `--mask-output-dir`: optional per-frame mask export directory
- `--threshold`: score cutoff, default `0.05`
- `--preflight`
- `--json`: only with `--preflight`
- `--show-boxes`
- `--show-labels`

Defaults:

- annotated video: `<video-stem>_tracked.mp4`
- JSON metadata: `<video-stem>_tracked.json`

Notes:

- text prompts seed objects on `--init-frame`, then the native tracker reuses geometry prompts for later frames
- box and point prompts seed explicit tracked objects directly on the init frame
- `--mask-output-dir` writes per-frame mask PNGs under frame-named subdirectories
- prompt sets must include at least one text, box, or point prompt
- `--preflight --json` prints a structured report without loading SAM,
  extracting frames, writing the annotated video, or writing tracking JSON. The
  report includes video path state, prompt parse status, model availability,
  output/json/mask destinations, init/end frame options, threshold, resolution,
  diagnostics, and declarative actions.

Examples:

```bash
swift run mere.run vision track ./clip.mp4 --prompt "a dog" --init-frame 12
swift run mere.run vision track ./clip.mp4 --prompt "a dog" --init-frame 12 --preflight --json
swift run mere.run vision track ./clip.mp4 --box "40,50,120,180,dog" --box "200,80,320,260,person" --show-boxes
```

### `mere.run vision track-live`

Capture a camera clip and run native SAM 3.1 tracking over the recorded session.

```bash
swift run mere.run vision track-live --output ./live.mp4 --prompt "a person"
```

Key options:

- `--prompt`: one or more text prompts used to seed objects from the init frame
- `--camera`: camera device index
- `--duration-seconds`
- `--init-frame`: initial frame index used to seed tracking
- `--seed-search-frames`: additional frames to search when the init frame finds no objects
- `--output`: annotated video path
- `--json-output`: tracking metadata path
- `--threshold`: score cutoff, default `0.05`
- `--show-boxes`
- `--show-labels`

Notes:

- `track-live` currently records a camera clip first, then runs tracking over the recorded media
- live tracking searches a short warm-up window after the init frame so startup exposure or motion blur does not silently produce an unsegmented output
- live mode accepts text prompts only in the current implementation
- `--output` is required; `--json-output` is optional

### `mere.run vision face`

Run local Buffalo-L face detection and ArcFace identity embeddings.

```bash
swift run mere.run model pull vision-face-buffalo-l --accept-model-license
swift run mere.run vision face detect ./group.jpg --json
swift run mere.run vision face detect ./group.jpg --include-embeddings --json-output ./faces.json
swift run mere.run vision face embed ./reference.jpg --face-index 0 --json
swift run mere.run vision face compare ./reference.jpg ./candidate.jpg --json
swift run mere.run vision face batch --input-list ./photos.txt --include-embeddings --jsonl-output ./faces.jsonl
```

Common options include `--model`, `--score-threshold`, and
`--execution-provider auto|cpu|coreml`. `embed` and `compare` select the largest
face unless an explicit zero-based face index is supplied.
`batch` keeps one detector/recognizer session warm across a path list and writes
one success-or-error record per image, making it the high-throughput library
indexing surface.

Buffalo-L weights are restricted to non-commercial research use by their
upstream provider. `model pull` requires `--accept-model-license`, prints the
license URL before download, and stores the weights outside the application.

### `mere.run vision pose`

Detect body, hand, and face landmarks in a local image through the native
platform runtime. Coordinates are normalized with a bottom-left origin and the
JSON payload records image dimensions, subject kinds, point names, and
confidence values.

```bash
swift run mere.run vision pose ./person.png
swift run mere.run vision pose ./person.png --no-face --minimum-confidence 0.25 --json
```

Key options:

- `--json-output`: output path; defaults to `<image>_pose.json`
- `--no-body`, `--no-hands`, `--no-face`: disable individual landmark groups
- `--max-hands`: maximum hand observations
- `--minimum-confidence`: filter weak landmarks
- `--json`: print the complete payload on stdout as well as writing it

On platforms without the Apple Vision framework, the command reports the
unsupported native runtime explicitly.

### `mere.run vision flow`

Generate dense optical flow between two equal-size images through the native
platform runtime. The output is a standard Middlebury `.flo` file containing
one 32-bit horizontal/vertical vector per pixel.

```bash
swift run mere.run vision flow ./frame-001.png ./frame-002.png \
  --output ./frame-001-to-002.flo \
  --json-output ./frame-001-to-002.json \
  --accuracy high
```

Metadata includes dimensions, vector count, accuracy, mean magnitude, and
maximum magnitude. Accuracy values are `low`, `medium`, `high`, and
`very-high`.

### `mere.run vision ocr`

Extract text from one or more images.

```bash
swift run mere.run vision ocr <images...> [options]
```

Key options:

- `--backend`: `lighton`, `glm`, or `infinity`
- `--compare`: compare LightOn against the selected secondary backend; defaults
  to GLM when `--backend lighton`
- `--model`: managed id or path to the LightOn OCR root when using the LightOn backend
- `--glmocr-cli`, `--glm-config`
- `--infinity-runtime`: `native` or `external`; native uses the Swift Q35 runtime
- `--infinity-model`: native managed model id or local path; upstream model or
  server id when `--infinity-runtime external`
- `--infinity-parser-cli`: path to the Infinity-Parser2 `parser` executable for
  external runs
- `--infinity-backend`: external `vllm-server`, `vllm-engine`, or `transformers`
- `--infinity-api-url`, `--infinity-api-key`: external vLLM server settings
- `--infinity-task`: `doc2json`, `doc2md`, or `custom`
- `--infinity-output-format`: `md` or `json`
- `--max-tokens`
- `--quiet`

Examples:

```bash
swift run mere.run model pull vision-ocr-lighton
swift run mere.run model pull vision-ocr-infinity-flash
swift run mere.run model pull vision-ocr-infinity-pro-int8
swift run mere.run vision ocr ./page.png --backend lighton --model ~/Library/Application\ Support/MereRun/models/vision-ocr-lighton
swift run mere.run vision ocr ./page.png --backend glm
swift run mere.run vision ocr ./page.png --backend infinity --infinity-task doc2md
swift run mere.run vision ocr ./page.png --compare --backend infinity
swift run mere.run vision ocr ./page.png --backend infinity --infinity-runtime external --infinity-api-url http://127.0.0.1:8000/v1/chat/completions
```

### `mere.run music analyze`

Analyze an audio file with ACE-Step 5 Hz LM audio understanding and print JSON
metadata to stdout.

```bash
swift run mere.run music analyze "<audio>" [options]
```

Key options:

- `--model`, `-m`
- `--checkpoints-root`
- `--turbo-subdirectory`
- `--vae-subdirectory`
- `--lm-subdirectory`
- `--duration`: analyze the first N seconds instead of the full decoded input
- `--max-new-tokens`
- `--lm-temperature`
- `--lm-top-k`
- `--lm-top-p`
- `--include-raw-lm`
- `--include-audio-codes`
- `--quiet`

Examples:

```bash
swift run mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-turbo-lm4b \
  --lm-subdirectory acestep-5Hz-lm-4B
swift run mere.run music analyze ./song.mp3 --duration 30 > ./song-analysis.json
```

### `mere.run music generate`

Generate music from a caption. The default model uses the native ACE-Step
pipeline with optional lyrics; Magenta RT2 models use the native Apple Silicon
Magenta bridge and ignore ACE-Step-only controls.

```bash
swift run mere.run music generate "<caption>" [options]
```

Key options:

- `--output`
- `--checkpoints-root`
- `--lyrics`
- `--lyrics-file`
- `--source-audio`: source song for ACE-Step cover conditioning; implies cover mode unless `--non-cover` is set
- `--analyze-source-audio`: use ACE-Step 5 Hz LM audio understanding to fill missing cover metadata from `--source-audio`
- `--reference-audio`: optional ACE-Step timbre reference audio file(s)
- `--duration`
- `--steps`
- `--use-lm`
- `--lm-subdirectory` (for example `acestep-5Hz-lm-4B` with `music-acestep-xl-turbo-lm4b`)
- `--text-subdirectory`
- `--seed`
- `--quiet`

Magenta RT2 options:

- `--style-conditioning`: `streaming` keeps the realtime C++ coarse style-token policy; `full` uses all MusicCoCa style tokens like the Python high-level generator
- `--temperature`
- `--top-k`
- `--cfg-musiccoca`
- `--cfg-notes`
- `--cfg-drums`
- `--drumless`
- `--unmask-width`
- `--seed-rotation`
- `--prefill-silence`
- `--prefill-duration`

Environment:

- `MERERUN_MUSIC_ACESTEP_ROOT`

Examples:

```bash
swift run mere.run music generate "upbeat electronic groove" --output ./track.wav
swift run mere.run music generate \
  "ambient piano and soft rain" \
  --lyrics-file ./lyrics.txt \
  --duration 8 \
  --steps 4 \
  --output ./ambient.wav
swift run mere.run music generate \
  "dream-pop cover with soft vocals" \
  --source-audio ./song.mp3 \
  --analyze-source-audio \
  --lyrics-file ./cover-lyrics.txt \
  --audio-cover-strength 0.85 \
  --output ./cover.wav
swift run mere.run music generate \
  "ambient modular synths with brushed drums" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./magenta.wav
```

### `mere.run music transcribe`

Transcribe a full music mix into instrument-separated MIDI with native
MuScriptor inference.

```bash
swift run mere.run music transcribe "<audio>" [options]
```

Key options:

- `--model`: one of `music-muscriptor-small`, `music-muscriptor-medium`, or `music-muscriptor-large`
- `--model-path`: explicit local model root
- `--output`, `-o`: output path, or `-` for stdout
- `--format`, `-f`: `midi`, `json`, or `jsonl`
- `--instruments`: comma-separated expected instrument groups
- `--list-instruments`
- `--sampling` and `--temperature`
- `--max-tokens-per-chunk`
- `--strict-eos`
- `--beam-size`
- `--chunk-batch-size`: upper bound on independent five-second chunks decoded
  together for greedy, sampling, or beam mode (default `4`). The runtime may
  reduce it for current unified-memory headroom and model/beam saturation;
  `1` always selects the single-chunk path.
- `--dtype`: `bfloat16`, `float16`, or `float32`
- `--context-output`: optional JSON path for detected tempo, meter, key,
  confidence scores, and beat positions; use `-` for stdout
- `--no-musical-context`: retain the legacy fixed-120-BPM MIDI conductor track
- `--quiet`

The model repositories are gated and the weights are CC BY-NC 4.0. Accept the
upstream Hugging Face terms before pulling.

```bash
swift run mere.run model pull music-muscriptor-medium --accept-model-license
swift run mere.run music transcribe ./song.mp3 --output ./song.mid
swift run mere.run music transcribe ./song.mp3 \
  --output ./song.mid --context-output ./song-context.json
swift run mere.run music transcribe ./song.wav \
  --instruments voice,drums,electric_bass \
  --format jsonl --output -
```

MIDI output detects musical context by default and writes Standard MIDI File
tempo, time-signature, and key-signature meta events. The detector combines
source-audio accents with MuScriptor note onsets and preserves absolute note
timing. It repeats the meter event and adds a marker at the first detected
downbeat rather than quantizing the notes. Fields without sufficient evidence
are omitted.

### `mere.run music realtime`

Run Magenta RealTime 2 generation. On macOS the command plays to the default
audio device by default; pass `--output` to capture a WAV file. Use `--no-play`
with `--output` and `--duration` for a headless smoke run.

```bash
swift run mere.run music realtime "<prompt>" [options]
```

Key options:

- `--model`: `music-magenta-rt2-small`, `music-magenta-rt2-base`, or a local Magenta RT2 root
- `--duration`
- `--output`
- `--play`, `--no-play`
- `--style-conditioning`: `streaming` keeps the realtime C++ coarse style-token policy; `full` uses all MusicCoCa style tokens like the Python high-level generator
- `--temperature`
- `--top-k`
- `--cfg-musiccoca`
- `--cfg-notes`
- `--cfg-drums`
- `--drumless`
- `--unmask-width`
- `--seed-rotation`
- `--prefill-silence`
- `--prefill-duration`
- `--interactive`: read live steering commands from stdin
- `--list-midi-inputs`: list CoreMIDI input sources and exit
- `--midi-monitor`: monitor a CoreMIDI input without loading Magenta RT2
- `--midi-log-events`: log parsed MIDI note and CC events to stderr
- `--midi-log-raw`: log raw CoreMIDI packet bytes to stderr
- `--midi-input`: CoreMIDI source name or unique ID for live note/control steering
- `--midi-channel`: `all` or `1` through `16`
- `--midi-note-offset`: transpose incoming MIDI notes before sending them to Magenta RT2
- `--midi-cc`: repeatable mapping as `cc=target:min:max`, for example `1=temp:0.2:1.4`
- `--quiet`

Interactive commands:

- `prompt <text>`
- `style streaming|full`
- `temp <value>`, `topk <value>`
- `mc <value>`, `notes <value>`, `drums <value>`
- `noteon <0-131>`, `noteoff <0-131>`, `onset 0|1`
- `drumless on|off`, `unmask <value>`, `seed <value>`
- `reset`, `quit`, `help`

Examples:

```bash
swift run mere.run music realtime \
  "ambient pads with sub bass" \
  --model music-magenta-rt2-small \
  --duration 4 \
  --output ./live.wav

swift run mere.run music realtime \
  "drumless glassy arpeggios" \
  --model music-magenta-rt2-small \
  --duration 2 \
  --output ./smoke.wav \
  --no-play

swift run mere.run music realtime \
  "ambient modular synths" \
  --model music-magenta-rt2-small \
  --duration 30 \
  --interactive

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
```

### `mere.run sfx generate`

Generate a mono WAV sound effect from a text prompt. The default model uses the
native Sony Research Woosh DFlow path; the original Woosh Flow checkpoint is
also available as `sfx-woosh-flow`. `sfx-mmaudio-large-44k-v2` selects the
native 44.1 kHz MMAudio large-v2 runtime.

```bash
swift run mere.run sfx generate "<prompt>" [options]
```

Key options:

- `--model`: `sfx-woosh-dflow`, `sfx-woosh-flow`,
  `sfx-mmaudio-large-44k-v2`, or a matching local model root
- `--negative-prompt`: negative text conditioning for MMAudio
- `--output`
- `--duration`
- `--steps`
- `--cfg`
- `--renoise`
- `--seed`
- `--quiet`

Examples:

```bash
swift run mere.run sfx generate \
  "metal wrench dropping onto concrete, bright clang and brief ring" \
  --model sfx-woosh-dflow \
  --duration 5 \
  --steps 4 \
  --cfg 4.5 \
  --output ./wrench-clang.wav
```

```bash
swift run mere.run sfx generate \
  "ocean waves striking a stone breakwater" \
  --negative-prompt "speech, music" \
  --model sfx-mmaudio-large-44k-v2 \
  --duration 8 \
  --steps 25 \
  --output ./breakwater.wav
```

### `mere.run sfx ae`

Encode audio into normalized Woosh-AE latents or decode those latents back to a
mono WAV.

```bash
swift run mere.run sfx ae encode ./input.wav -o ./input-latents.npy
swift run mere.run sfx ae decode ./input-latents.npy -o ./input-roundtrip.wav
```

### `mere.run sfx condition text`

Export Woosh text-conditioning tensors for a prompt. The output safetensors file
contains `embeddings` and `mask` arrays.

```bash
swift run mere.run sfx condition text "glass breaking" -o ./glass-condition.safetensors
```

### `mere.run sfx clap score`

Score a text prompt against an audio file with the native Woosh-CLAP text and
PaSST audio towers. The command prints JSON to stdout.

```bash
swift run mere.run sfx clap score "glass breaking" ./glass.wav
```

### `mere.run sfx video generate`

Generate a mono WAV sound effect from a raw video file or precomputed
Synchformer video features. `.npy` feature inputs must have shape
`[frames, 768]` or `[1, frames, 768]` for Woosh. MMAudio requires the original
video so it can compute both CLIP and Synchformer conditioning.

```bash
swift run mere.run model pull sfx-woosh-synchformer --accept-model-license
swift run mere.run sfx video generate \
  "footsteps echoing in a hallway" \
  ./silent-hallway.mp4 \
  --model sfx-woosh-dvflow-8s \
  --duration 8 \
  --output ./hallway-footsteps.wav
```

```bash
swift run mere.run sfx video generate \
  "a skateboard rolling over rough pavement" \
  ./skateboard.mp4 \
  --model sfx-mmaudio-large-44k-v2 \
  --negative-prompt "speech, music" \
  --clip-batch-size 4 \
  --sync-batch-size 1 \
  --output ./skateboard.wav
```

Use `--preflight --json` to inspect input mode, output path state, VFlow/DVFlow
model availability, raw-video conditioning requirements, duration, effective
step count, CFG, renoise schedule, and follow-up actions before loading MLX or
generating audio. `.npy` feature inputs do not require Synchformer during
preflight or generation.

### `mere.run video cosmos3`

Run the complete pinned `nvidia/Cosmos3-Edge` checkpoint through native
Swift/MLX:

```bash
mere.run model pull video-cosmos3-edge-mlx --accept-model-license

mere.run video cosmos3 \
  "continue forward through the same corridor" \
  --mode forward-dynamics \
  --image ./vesper.png \
  --action-domain camera_pose \
  --action-file ./camera-actions.json \
  --action-chunk-size 60 \
  --action-resolution 256 \
  --output ./vesper-forward.mp4
```

`--mode` accepts `text-to-image`, `image-to-image`, `text-to-video`,
`image-to-video`, `video-to-video`, `policy`, `forward-dynamics`,
`inverse-dynamics`, and `reasoner`. Action domains cover the published
automotive, camera, hand, Push-T, UMI, Bridge, DROID, RoboMIND, Galbot, AgiBot,
and Fractal layouts. Policy and inverse-dynamics outputs are written as JSON.
Forward-dynamics action files contain normalized model-space values, not meters;
the resident Cosmos3 world server can also accept them as
`model_space_actions`.

The reasoner accepts text alone or one `--image`/`--video` and shares the
checkpoint's understanding transformer with its packed SigLIP2 vision tower:

```bash
mere.run video cosmos3 \
  "Describe the navigable paths and obstacles." \
  --mode reasoner --image ./vesper.png --max-new-tokens 128
```

See `mere.run guide video-cosmos3` for sampling defaults, action JSON, exact
upstream pins, and the resident world-server recipe.

### `mere.run video animate`

Animate or replace a masked reference subject using the native Swift/MLX
SCAIL-2 runtime:

```bash
swift run mere.run video animate "<prompt>" \
  --reference <image> \
  --reference-mask <mask-image> \
  --driving-video <video> \
  --driving-mask <mask-video> \
  [options]
```

The required image/video masks use seven-color SCAIL-2 segmentation. Important
options are `--mode animation|replacement`, `--model-root`, `--width`,
`--height`, `--steps`, `--guidance-scale`, `--shift`, `--fps`,
`--segment-length`, `--segment-overlap`, and paired repeatable
`--additional-reference` / `--additional-reference-mask`. `--tail-policy`
accepts `drop` or `pad-trim`; `--audio-source` accepts `none` or `driving`.
`--profile fast` is the default: 832x480, the separately pulled
`scail2-lightx2v-4step` adapter, and the published no-CFG four-step Euler
recipe with shift 5. `--profile quality` explicitly selects the configurable
40-step UniPC/CFG recipe. The segment defaults remain 81 frames with five
clean-history overlap frames; compatibility defaults are `drop` and `none`.
`pad-trim` preserves legal `1 mod 4` clip lengths and pads only the final
incomplete temporal segment to the next legal length.

`--preflight --json` validates the MLX model root, input pairs, output, and
execution plan without loading MLX or decoding video. The converted checkpoint
is distributed separately from the mere.run source repository; runtime
generation is native Swift/MLX and never launches Python or ComfyUI.

### `mere.run video prepare-masks`

Prepare reference and driving masks from a typed schema-version 1 plan:

```bash
swift run mere.run video prepare-masks \
  --plan ./request.json \
  --output-dir ./artifacts \
  [--preview-frame <frame>] \
  [--preflight --json]
```

Plans contain `mode` (`animation` or `replacement`), exact driver
geometry/FPS/range, one to six stable subjects,
unique legal palette colours, project-materialized reference images,
text/box/positive/negative selectors, and optional point/box/dense painted-PNG
keyframe corrections. Preview mode segments references plus one selected
driving frame. Reference preparation aspect-fits each image and its mask into
the requested canvas; replacement mode mattes non-subject reference pixels to
black. Full mode tracks both directions, records gaps and quality warnings, and
emits prepared reference image/mask pairs, a ProRes 4444 categorical driving
mask, overlay MP4, contact sheet, tracking/quality JSON, normalized driver, and
a canonical hashed manifest.

### `mere.run video generate`

Generate MP4 video with the native LTX and Wan2.2 pipelines.

```bash
swift run mere.run video generate "<prompt>" [options]
```

Key options:

- `--quality`: `draft` selects the fast standalone-distilled checkpoint;
  `final` selects the full dev + distilled-LoRA quality pipeline
- `--output-mode`: `video-only` (default) or synchronized `audio-video`
- `--variant`: compatibility selector; `distilled` defaults to draft
  video-only and `unified-av` defaults to final audio-video; explicit `--model`
  still wins
- `--model-root`
- `--output`
- `--width`, `--height`
- `--num-frames`
- `--duration`
- `--fps`
- `--seed`
- `--steps`, `--guidance-scale`, `--shift`, `--negative-prompt` for Wan2.2
- `--audio`: source audio path; automatically selects native LTX 2.3 A2Vid
- `--audio-start-time`: source segment offset in seconds (default `0`)
- `--a2v-guidance-scale`: audio-modality guidance (default `3`)
- `--video-cfg-guidance-scale`: full/A2Vid video text CFG (default `3`)
- `--audio-cfg-guidance-scale`: full unified-AV audio text CFG (default `7`)
- `--v2a-guidance-scale`: full unified-AV video-to-audio modality guidance
  (default `3`)
- `--a2v-steps`: full/dev stage-one steps (default `30`)
- `--negative-prompt`: also overrides the official A2Vid negative prompt
- `--image`
- `--image-strength`
- `--end-image`
- `--end-image-strength`
- `--preflight`
- `--json`: only with `--preflight`
- `--timings`: native LTX 2.3 split-distilled, unified-AV, or A2Vid phase
  timings on stderr; unavailable for legacy merged distilled roots
- `--timings-output`: write those timings as JSON
- `--quiet`

Environment:

- `MERERUN_VIDEO_LTX_MODEL_ROOT`
- `MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT` for an external
  `mlx-community/gemma-3-12b-it-4bit` checkout used by `video-ltx23-av-mlx`,
  `video-ltx23-full-mlx`, and the legacy `video-ltx23-a2vid-mlx`

For `--output-mode audio-video`, keep `--fps 24` unless you are deliberately making
a retimed clip. LTX 2.3 unified AV is trained around 24 fps; using 8 fps can
make generated motion look slow while audio remains normal. Use `--duration`
for clip length so the CLI can choose the nearest legal `8n+1` frame count.
Use the default `--quality draft` lane for fast iterations. Use `--quality
final` for the current high-quality two-stage checkpoint. Output is a separate
choice: both qualities default to video-only, and `--output-mode audio-video`
adds synchronized generated audio. Suppressing audio on the same checkpoint is
an output contract, not the source of the draft lane's speed advantage.

On `video-ltx23-av-mlx`, distilled generation uses the split-layout LTX 2.3
transformer and preserves its joint audio/video denoising contract because
audio-to-video cross attention contributes to the video result. It skips
loading the audio VAE and vocoder, and the resulting MP4 has no audio stream.

With `--audio`, the command resolves `video-ltx23-full-mlx` automatically.
The full/dev transformer performs guided half-resolution denoising with frozen
source-audio latents; after x2 upsampling, the official distilled LoRA is fused
for the four-step refinement. The original selected audio segment is muxed into
the MP4. Short inputs and incompatible models fail explicitly; there is no
soundtrack-only fallback.

For native Wan2.2 image-to-video, pass
`--model video-wan22-ti2v-5b-mlx --image <frame>`. Wan dimensions are snapped
to multiples of 32 and frame counts follow `4n+1`; 512x320 with 17 frames is a
useful short world-transition baseline. Tiny 128-pixel outputs are structural
smokes, not quality renders.

Preflight mode:

- `--preflight --json` prints a structured plan without loading MLX, loading a
  video model, creating directories, or writing an MP4.
- the report includes model availability, output path state, source/end image
  and audio state, resolved dimensions, resolved frame count/duration, source
  audio offset, seed, input mode, whether audio conditions generation, whether
  the source soundtrack is preserved, diagnostics, and declarative actions.
- blockers such as a missing model root, missing image, invalid frame rate, or
  `--end-image` without `--image` produce JSON and a nonzero exit. Requesting
  phase timings on a legacy merged distilled root or Wan2.2 is also blocked.
- notes such as dimension/frame snapping remain machine-readable so a UI can
  explain the exact render that will run.

Examples:

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --num-frames 65

swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --num-frames 65 \
  --output ./clip.mp4 \
  --preflight \
  --json

swift run mere.run video generate \
  "a red fox runs across a snowy clearing, detailed winter fur, natural motion" \
  --quality final \
  --duration 4 \
  --output ./fox-final.mp4

swift run mere.run video generate \
  "the first-person camera walks straight forward through the same corridor" \
  --model video-wan22-ti2v-5b-mlx \
  --image ./corridor.png \
  --width 512 \
  --height 320 \
  --num-frames 17 \
  --steps 40 \
  --output ./corridor-forward.mp4

swift run mere.run video generate \
  "two actors talking beside a window while a restrained orchestral score and distant city sirens play underneath" \
  --quality final \
  --output-mode audio-video \
  --duration 15 \
  --fps 24 \
  --output ./dialogue-score-sfx.mp4

swift run mere.run video generate \
  "a kinetic live performance, camera orbiting the vocalist" \
  --audio ./song.wav \
  --audio-start-time 30 \
  --duration 5 \
  --image ./performer.png \
  --output ./performance.mp4

swift run mere.run video generate \
  "a car drives from a bright morning street into a warm sunset road, smooth forward motion" \
  --image ./car-start.png \
  --end-image ./car-end.png \
  --num-frames 65 \
  --output ./car-start-to-end.mp4
```

### `mere.run video session`

Keep a standalone distilled or full dev LTX 2.3 runtime resident while
processing serial JSONL generation requests:

```bash
printf '%s\n' \
  '{"id":"draft-1","prompt":"a fox runs across snow","output":"./draft-1.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  '{"id":"draft-2","prompt":"a fox runs across snow","output":"./draft-2.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  | swift run mere.run video session --model video-ltx23-av-mlx
```

Each stdin line produces one typed result or error line on stdout. A request
requires `prompt` and `output`; it can override `width`, `height`, `num_frames`,
`fps`, `seed`, `image`, `image_strength`, `end_image`, and
`end_image_strength`. Successful responses include phase timings and
`resident_model_reused`. Pass `--model video-ltx23-full-mlx` for the two-stage
quality lane. The full session preserves the dev checkpoint for Stage 1 and
activates its resident distilled LoRA only during Stage 2, so repeat requests
do not reload or mutate the base transformer.

### `mere.run video export-latents`

Run native distilled LTX denoising and export the final latent tensor.

```bash
swift run mere.run video export-latents \
  --model-root /path/to/distilled-ltx \
  --output out.safetensors \
  "a cinematic drone flyover at sunrise"
```

### `mere.run model list`

List all managed model IDs, whether they are installed, and their referenced
payload size. Referenced sizes follow symlinks and are not additive when models
share payloads.

```bash
swift run mere.run model list
```

### `mere.run model storage` and `mere.run model gc`

Inspect physical storage, sharing, and per-model removal impact without
double-counting shared files:

```bash
mere.run model storage
mere.run model storage --json
```

Preview unreferenced payload, stale snapshot, blob, revision-reference, and
partial-download cleanup. The default is read-only; mutation requires `--force`:

```bash
mere.run model gc
mere.run model gc --json
mere.run model gc --force
```

Cleanup rechecks the ownership graph under the same lock used by managed pulls.
Existing legacy cache layouts remain readable and are adopted into the
revision-addressed layout without copying matching payload bytes on a later pull.

### `mere.run model remove`

Remove an install and reclaim backing payloads that no remaining managed or
legacy link uses:

```bash
mere.run model remove image-zimage-nano
mere.run model remove image-zimage-nano --force --json
mere.run model remove image-zimage-nano --keep-cache
```

### `mere.run status`

Show a quick local snapshot: whether the API server answers, which model it
reports as loaded through `/v1/models`, the active model-store path/source, and
which managed models are installed in that store.
When the server exposes the native runtime pool, JSON status also includes pool
entries, active request counts, request admission queue depth, the memory
snapshot, runtime capability flags, aggregate cache stats, per-model prefix KV
cache stats, per-model decode batching stats when enabled, aggregate benchmark
stats from completed native chat requests, embedding/image/TTS/ASR sidecar residency and
readiness, and the runtime settings path. `loaded` remains the
backward-compatible resident-object field; additive `ready: false` means a text
model is still preparing or a sidecar's first operation is loading or failed.
The human formatter calls that sidecar state `resident (not ready)` and excludes
it from the `loaded models` summary.

```bash
swift run mere.run status
swift run mere.run status --host 127.0.0.1 --port 11434
swift run mere.run status --json
```

Useful options:

- `--host`: local API host to check, default `127.0.0.1`
- `--port`: local API port to check, default `8080`
- `--api-key`: bearer token for `/v1/models`, also read from `MERERUN_API_KEY`
- `--timeout-seconds`: network probe timeout
- `--json`: emit a structured snapshot for scripts and agents

### `mere.run model runtime get`

Read typed per-model API runtime settings from the active model store.

```bash
swift run mere.run model runtime get text-chat-gemma4
swift run mere.run model runtime get text-chat-gemma4 --json
```

Settings are stored at
`<active model store>/.mere-run/runtime-model-settings.json` and follow
`--models-root` / `MERERUN_MODELS_DIR`.

### `mere.run model runtime set`

Update typed API serving defaults for a managed API-capable model.

```bash
swift run mere.run model runtime set text-chat-gemma4 \
  --alias chat-default \
  --pinned \
  --ttl-seconds 3600 \
  --max-context-tokens 8192 \
  --max-tokens 1024 \
  --temperature 0.6 \
  --top-p 0.9 \
  --min-p 0.05 \
  --kv-cache-mode auto
```

Use the matching `--clear-*` flags, including `--clear-min-p`, to remove
optional values. Engine overrides
are validated against the curated catalog. Gemma4, Qwen-family, and LFM2 accept
the explicit `affine8` resident-cache mode as a memory control relative to
full-precision KV. Qwen-family and LFM2 dequantize the generic cache for
attention. Gemma Turbo already defaults to a smaller 4-bit TurboQuant cache, so
forcing affine 8-bit can increase its KV residency. `default` restores the
engine/model/server default, not necessarily full precision. Gemma additionally
accepts `polar2` and `auto`; non-Gemma4 models reject PolarKV runtime modes.
`--ttl-seconds` unloads idle loaded models during the runtime
pool's opportunistic eviction passes, while `--pinned` protects a model from
automatic TTL/LRU eviction without blocking explicit unload. Memory-pressure LRU
uses the API server's `--memory-guard` tier. The guard derives soft/hard
ceilings from Darwin physical footprint (RSS elsewhere), host memory headroom,
and a tier reserve;
elevated pressure evicts the least-recently-used idle unpinned model, while
critical pressure evicts every idle unpinned model.

Managed embedding, image, TTS, and ASR sidecar models accept only the residency controls
`--pinned`, `--unpinned`, `--ttl-seconds`, and `--clear-ttl`. Their default idle
TTL is 300 seconds. Text-only alias, context, sampling, engine, and KV settings
are rejected for sidecars. The special `qwen-image-edit` repository lane is
resident but is not configurable through `model runtime`, so it uses the
default lifecycle policy.

### `mere.run model pull`

Download a managed Hugging Face snapshot into the local model store. The command checks
the model capability catalog and available disk space before downloading so unsupported
machines do not pull models they cannot run and tight disks fail with a useful cache path.

```bash
swift run mere.run model pull image-zimage-nano --accept-model-license
swift run mere.run model pull image-zimage-nano --accept-model-license --preflight --json
swift run mere.run model pull --all
```

Use `--preflight --json` to check support, install state, download source,
model store path, hub cache path, disk headroom, and next actions before any
download starts. The report includes `pull-model` or `pull-models` actions and
exits nonzero after printing JSON when hard blockers are present.

Use `--allow-unsupported` only when you intentionally accept the runtime risk.

Models with non-commercial, research-only, gated, revenue-limited, or custom
acceptable-use terms require `--accept-model-license` before a new download.
The preflight reports the exact component terms and blocks without
acknowledgement; `--all` skips restricted entries unless the flag is present.
mere.run records the immutable source revisions and acknowledged term URLs in
the installed `mererun_model.json`, but the user remains responsible for
deciding whether their intended use complies. See
[`model-sources.md`](./model-sources.md#restricted-model-downloads).

### `mere.run adapter list` and `mere.run adapter pull`

List the built-in public adapter catalog or install one immutable release:

```bash
mere.run adapter list
mere.run adapter list --json
mere.run adapter pull mere-platform-assistant
mere.run adapter pull scail2-lightx2v-4step
```

The pull verifies the cataloged byte count and SHA-256 before atomically
installing the adapter. Stdout contains only the installed path; progress and
verification diagnostics go to stderr. Use the adapter id directly with
`text chat --lora`, `api serve --lora`, or the matching SCAIL
`video animate --distilled-adapter` option. `video animate --profile fast`
selects `scail2-lightx2v-4step` and its fixed four-step schedule.

For a cross-command decision guide, see [Benchmarking](./benchmarking.md). The
sections below are the command reference for each benchmark lane.

### `mere.run model benchmark gemma4-kv`

Run a fixed-token real-checkpoint Gemma4 KV cache comparison. The command runs
the selected Gemma4 model twice in one process: default Gemma4 KV settings first,
then the runtime `polar2` mode, which uses model-default prefill with packed
`polar` 2-bit KV from token 0 for decode. It
disables EOS stopping so both variants decode exactly `--decode-tokens`, and
reports TTFT, prefill tok/s, KV conversion time, decode tok/s, end-to-end tok/s,
and process resident memory before and after each variant.

```bash
swift run mere.run model benchmark gemma4-kv \
  --model text-chat-gemma4-turbo \
  --decode-tokens 48 \
  --json
```

Use `--prompt`, `--prompt-file`, or `--prompt-repeat` to control prompt length.
Use `--prompt-repeat-values` and `--decode-token-values` with comma-separated
values to run a prompt-size/decode-length matrix for promotion evidence.
The default fixture prompt is deterministic and intended for local A/B
comparisons, not model-quality evaluation.

### `mere.run model benchmark gemma4-mtp`

Run a fixed-token real-checkpoint Gemma4 MTP comparison. The command runs the
selected Gemma4 model twice in one process: `baseline` with
`MERERUN_GEMMA4_MTP=0`, then `mtp` with `MERERUN_GEMMA4_MTP=1`. It defaults to
the practical 4-bit checkpoint and disables EOS stopping so both variants decode
exactly `--decode-tokens`.

```bash
swift run mere.run model benchmark gemma4-mtp \
  --model text-chat-gemma4-12b-4bit \
  --decode-tokens 48 \
  --json
```

Output includes prompt tokens, generated tokens, load time, prefill time, decode
time, TTFT, prefill tok/s, decode tok/s, end-to-end tok/s, process resident
memory, decode speedup, end-to-end speedup, and MTP counters for rounds, drafted
tokens, accepted tokens, rejected tokens, acceptance rate, and accepted tokens
per round.

Use `--prompt`, `--prompt-file`, or `--prompt-repeat` to control prompt length.
Use `--prompt-repeat-values` and `--decode-token-values` with comma-separated
values to run a prompt-size/decode-length matrix. `--mtp-block-size` and
`--mtp-min-prompt-tokens` are optional benchmark overrides for draft block size
and the activation threshold; leaving them unset uses the runtime policy
defaults. The default fixture is deterministic and intended for throughput
comparison, not model-quality evaluation.

### `mere.run model benchmark tool-continuations`

Run two deterministic real-checkpoint Gemma 4 cases that continue after
completed tool results. The cases exercise typed nested and null arguments,
reasoning metadata, tool-call correlation fields, and a repeated two-tool chain.
They require the final answer to use the authoritative results and reject a
spurious additional tool call.

```bash
swift run mere.run model benchmark tool-continuations \
  --model text-chat-gemma4-12b-4bit \
  --log-responses \
  --json
```

Use `--dry-run` to print the plan without loading a checkpoint. `--model-root`
accepts a local converted Gemma 4 directory while leaving the managed install
untouched. `--max-tokens` and `--context-size` override the per-case generation
and context limits.

### `mere.run model benchmark q36-mtp`

Run a requested-token real-checkpoint Qwen-family MTP comparison. The command
supports `text-chat-q36-nano` (default), `text-agent-ornith-9b`, and
`text-agent-ornith-35b-mlx`, and runs the selected model with three policies:

- `baseline`: MTP disabled with `MERERUN_Q35_MTP_SPECULATION=0`.
- `adaptive`: production long-context policy.
- `forced`: MTP enabled with `MERERUN_Q35_MTP_SPECULATION=1` and a configurable
  forced threshold.

```bash
swift run mere.run model benchmark q36-mtp \
  --prompt-repeat-values 8,80,150 \
  --temperature-values 0,0.7 \
  --decode-tokens 32 \
  --json
```

The runtime may still stop on EOS before `--decode-tokens`. Output includes
prompt tokens, generated tokens, load time, prefill time, decode time, TTFT,
prefill tok/s, decode tok/s, end-to-end tok/s, process resident memory, and
adaptive/forced speedups versus baseline. Greedy forced MTP uses the native
block verifier; non-greedy forced MTP stays on the exact probabilistic
speculative path. Use `--mtp-block-size` to test a different greedy draft block
cap and `--forced-mtp-min-prompt-tokens` to adjust the forced policy threshold.

### `mere.run model benchmark api-workload`

Replay streaming OpenAI-compatible chat requests against an already-running
`mere.run api serve` process. This is the serving-path benchmark for request
admission, prefix KV reuse, automatic decode batching on supported engines, and
the eventual SSD KV decision.

```bash
MERERUN_GEMMA4_PREFIX_KV_CACHE=0 \
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 1

swift run mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --json
```

To test the measured-work path, rerun the same workload with default prefix
reuse and `--max-active-requests 4`; supported Gemma4, Qwen-family, and LFM2
engines automatically enable decode batching at concurrency above `1`. Then
compare TTFT, wall-clock throughput, and runtime status deltas:

```bash
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 4

swift run mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --concurrency 4 \
  --json
```

The built-in workload uses one stable system prefix and varied final user turns.
Output reports per-request TTFT, total latency, streamed chunk count, wall-clock
requests/sec, prefix KV hits/misses, reused prefix tokens, decode batched steps,
single decode steps, and whether SSD KV cache is available. Use
`--workload-file` to replay JSONL rows with either `{ "id", "user" }` or
`{ "id", "messages" }`.

### `mere.run model benchmark code`

Run a small real coding-eval slice against installed local coding models. The
default suite is `humaneval-slice`, a three-task HumanEval subset covering
`HumanEval/0`, `HumanEval/3`, and `HumanEval/8`. The default model comparison
uses the supported members of the coding comparison lane for this machine:
`text-agent-ornith-9b` and `text-code-north-mini` on 32 GB Macs, with
`text-code-qwen3` added on 64 GB and larger machines. Pass `--models` to force a
specific explicit comparison.

```bash
swift run mere.run model benchmark code \
  --allow-code-execution \
  --json
```

The command prompts each model once per task, combines the generated Python with
the task tests, and runs that candidate in a sandboxed `python3` subprocess with
a per-candidate timeout. Because scoring executes generated code locally, pass
`--allow-code-execution` for real runs or `--dry-run` to inspect the plan.
The default `--sandbox auto` uses `sandbox-exec` on macOS and `bubblewrap` on
Linux when available. Use `--sandbox none` only for a trusted local smoke where
timeout and temporary-directory hygiene are enough. The default generation cap is
`--max-tokens 1024`, and capped cases are reported as `reachedMaxTokens` in JSON
or `capped=true` in text output. Reasoning blocks are preserved separately as
`reasoningCharacters`/`reasoning_chars` and
`incompleteReasoning`/`reasoning_incomplete`, while only visible code is
executed. `reasoning_reopened=true` flags a second generated reasoning block,
which usually indicates a loop or phase restart. Use `--models` and `--tasks`
to narrow the slice while iterating. Use `--models text-agent-ornith-35b` for
the larger Ornith GGUF eval target.

Pass `--thinking` to let the model reason before answering (matching
thinking-enabled published evals). Reasoning is split from the scored code as
usual; the HumanEval-specific stop sequences are disabled for thinking runs
because they can fire inside the reasoning block, so pair `--thinking` with a
larger `--max-tokens` (for example 3072).

For a larger slice from the official HumanEval data, download and decompress
`HumanEval.jsonl.gz`, then pass the JSONL file with `--humaneval-file`:

```bash
curl -L https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o /tmp/HumanEval.jsonl.gz
gunzip -c /tmp/HumanEval.jsonl.gz > /tmp/HumanEval.jsonl
swift run mere.run model benchmark code \
  --humaneval-file /tmp/HumanEval.jsonl \
  --tasks HumanEval/0,HumanEval/1,HumanEval/2,HumanEval/3,HumanEval/4 \
  --allow-code-execution
```

### `mere.run model benchmark vlm`

Run a tiny synthetic VLM smoke, or use `lmms-eval` to compare an installed
vision-chat model against existing multimodal datasets.

```bash
swift run mere.run model benchmark vlm --json
```

The default synthetic suite compares `vision-chat-gemma4-12b` with the existing
`vision-inspect-qwen3-vl-2b` backend on deterministic color, location, and
counting fixtures.

For existing datasets, install or check out `lmms-eval`, then start with a
dry-run:

```bash
swift run mere.run model benchmark vlm \
  --dataset mathvista-testmini \
  --limit 16 \
  --lmms-eval-root ~/src/lmms-eval \
  --dry-run \
  --json
```

Preset dataset flags map to upstream task names:

| Dataset flag | lmms-eval task |
| --- | --- |
| `mathvista-testmini` | `mathvista_testmini` |
| `mmmu-val` | `mmmu_val` |
| `chartqa` | `chartqa` |
| `docvqa-val` | `docvqa_val` |
| `mme` | `mme` |

Use `--lmms-tasks` for raw upstream task names, `--external-endpoint --base-url`
for an already-running OpenAI-compatible server, and omit `--dry-run` to let the
command start a local `mere.run api serve` process per requested model.

### `mere.run model capabilities`

Show this machine's supported models, recommended setup package, chat winners
by RAM band, and a short summary of what each model does.

```bash
swift run mere.run model capabilities
swift run mere.run model capabilities --all
```

### `mere.run model info`

Inspect a canonical model ID or a local model root. The Storage section reports
the layout, resolved payload size, wrapper size when different, and symlink
counts.

```bash
swift run mere.run model info image-zimage-nano
swift run mere.run model info /path/to/model/root --components
swift run mere.run model info text-chat-gemma4
```

### `mere.run model remove`

Delete an installed managed model by canonical ID.

```bash
swift run mere.run model remove image-zimage-nano
swift run mere.run model remove image-zimage-nano --force
```

### `mere.run model repair-manifests`

Write missing `mererun_model.json` manifests for known local model roots.

```bash
swift run mere.run model repair-manifests
swift run mere.run model repair-manifests --dry-run
```

### `mere.run api serve`

Start an OpenAI-compatible local API server.

```bash
swift run mere.run api serve [options]
```

Current endpoint surface:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `GET /runtime/status`
- `POST /runtime/models/{id}/load`
- `POST /runtime/models/{id}/unload`
- `GET/PATCH /runtime/models/{id}/settings`

The four `/runtime/models/{id}` load, unload, and settings operations target the
chat/text runtime pool only. Managed embedding, image, TTS, and ASR sidecar TTL/pinning is
configured through `mere.run model runtime set` or the settings file; sidecars
remain visible through `/v1/models` and `/runtime/status`.

Preflight mode:

- `--preflight --json` prints a structured serving plan without binding a port,
  loading a model, or starting the server.
- the report includes host, port, base URL, loopback/auth status, selected
  engine, requested/default model, model-store state, runtime limits, KV cache
  settings, companion model ids, diagnostics, and redacted follow-up actions.
- non-loopback binds without `--api-key` or `MERERUN_API_KEY` produce a blocked
  JSON report and nonzero exit so agents can fix configuration without parsing
  stderr.
- `--json` is supported with `--preflight` for `api serve`; the long-running
  server path remains the OpenAI-compatible HTTP surface.

Security defaults:

- loopback binds are local-first and do not require auth
- non-loopback binds require `--api-key` or `MERERUN_API_KEY`
- `POST /v1/chat/completions`, `POST /v1/embeddings`,
  `POST /v1/images/generations`, and `POST /v1/audio/speech` require
  `Content-Type: application/json`; `POST /v1/images/edits` and
  `POST /v1/audio/transcriptions` require `multipart/form-data`
- `--rate-limit-per-minute` applies basic request throttling to the
  OpenAI-compatible routes
- `--max-active-requests` controls fair FIFO admission for chat, embedding,
  image, TTS, and ASR inference; the default `1` serializes local inference
  across text and media activation peaks while exposing queue depth in status;
  queued client cancellations are removed from the FIFO instead of running
  later. Raising it is an explicit throughput and unified-memory tradeoff.
  Explicit runtime model load/unload maintenance shares the same queue.
  Values above `1` automatically engage supported Gemma4, Qwen-family,
  and LFM2 decode batching unless an engine-specific environment variable forces
  the serial path
- `--memory-guard` controls runtime memory pressure behavior. Accepted values
  are `off`, `safe`, `balanced`, `aggressive`, and `custom`; `custom` also
  requires `--memory-guard-custom-ceiling-gb`.
- elevated or critical memory pressure pauses extra concurrent admissions while
  letting one request run so the server can make progress
- embedding, image/image-edit, TTS, and ASR sidecar endpoints each keep only
  their most recently used runtime resident. Matching requests reuse loaded model state
  through an exclusive queue; selecting another model or ASR backend unloads
  the previous runtime before loading its replacement. Idle sidecars default to
  a 300-second TTL with autonomous expiry, re-read managed-model pin/TTL changes
  while idle, and participate in the same memory-pressure policy without
  evicting active or queued work. Cold loads are exclusive across lanes and
  project the catalog/path size against the hard guard before allocating;
  requests that still lack headroom after idle-resident eviction are rejected.
  Managed embedding, image generation, TTS, and ASR catalog IDs honor those
  settings; the special `qwen-image-edit` repository ID uses the
  default policy because `model runtime set` does not address it. Runtime status
  reports identities, residency, readiness, load/access state, queues, and
  counters.
- Gemma4, Qwen-family, and LFM2 chat use chunked prefill checkpoints for long
  prompts.
- Gemma4 uses in-memory prefix KV reuse by default in `api serve`; set
  `MERERUN_GEMMA4_PREFIX_KV_CACHE=0` for a baseline. Runtime status reports
  entries, hits, and reused tokens when a Gemma4 model is loaded, including
  semantic chat-prefix checkpoints before the final message when token prefixes
  match exactly.
- Qwen-family chat uses text-only in-memory prefix KV reuse by default in
  `api serve`; set `MERERUN_Q35_PREFIX_KV_CACHE=0` for a baseline. Vision
  prompts are excluded from reuse, and text-only requests use the same semantic
  chat-prefix checkpoints as Gemma4.
- LFM2 chat uses in-memory prefix KV reuse by default in `api serve`; set
  `MERERUN_LFM2_PREFIX_KV_CACHE=0` for a baseline. Checkpoints fork both
  attention KV and short-convolution state.
- Managed Gemma4 12B text and vision pulls install a companion MTP assistant.
  When `MERERUN_GEMMA4_MTP` is not disabled, greedy serial decode can use that
  assistant on the decode tail after prefill; sampled requests, continuous
  batching, raw local model paths, and prefix-KV seeded requests use baseline
  decode
- Gemma4, Qwen-family, and LFM2 decode batching engages automatically when
  `--max-active-requests` is above `1`. Their engine-specific continuous-batching
  variables can force the implementation on or force the serial path. Status
  reports actual batched decode steps and max observed batch size. Gemma4
  full-attention rows stay same-position because that path still uses scalar
  RoPE/cache offsets. Qwen-family and LFM2 full-attention rows use
  row-offset-aware ragged KV caches; Qwen-family linear attention and LFM2
  short-convolution layers use typed recurrent state, so compatible rows can
  batch across decode positions. The scheduler services the earliest decode
  position first by batching compatible rows there or advancing one lower-offset
  row until it can join a compatible batch
- Gemma4 can opt into experimental packed PolarKV with
  `--kv-quant-scheme polar --kv-bits 2`; use it for memory-pressure and
  long-context synthetic decode testing. It is not the default until checkpoint
  benchmarks prove the end-to-end model path.
- Per-model runtime settings can set `kvCacheMode` to `affine8` for Gemma4,
  Qwen-family, and LFM2 as a memory control relative to full-precision KV.
  Qwen-family and LFM2 dequantize the generic cache for attention, so this is an
  explicit tradeoff. Gemma Turbo already defaults to a smaller 4-bit TurboQuant
  cache, so forcing affine 8-bit can increase its KV residency. `default`
  restores the selected engine/model/server default, not necessarily full
  precision. Gemma4 also accepts `polar2` or `auto`; `auto` keeps the default KV
  path below 1024 prompt tokens and switches to decode-deferred packed PolarKV
  at or above that threshold.
- `/runtime/status` and `mere.run status` aggregate prefix hits, reused tokens,
  batched decode steps, completed chat requests, generated tokens, and average
  load/prefill/decode timings across loaded models under `cacheStats` and
  `benchmarkStats`
- generation parameters are bounded before execution; for example, `max_tokens` must fit the configured context size, and native chat engines receive that same context cap for prompt truncation
- LoRA adapters for the API server are selected by the operator with `--lora`; request bodies cannot provide local LoRA paths

Engine values:

- `text-code`
- `text-chat-klein`
- `text-chat-gemma4`
- `text-chat-q36`
- `text-chat-lfm2`
- `text-chat-deepseek-v4-flash`

OpenAI chat compatibility:

- DS4 raw-proxies the full `/v1/chat/completions` body to `ds4-server`.
- Native engines decode the common OpenAI Chat request shape and reject
  unsupported high-impact fields with `invalid_request_error`.
- `max_completion_tokens`, `developer` messages, function tools, image content
  parts, structured JSON mode, `stop` sequences, and streaming usage are
  capability-gated by engine.
- `tool_choice` accepts `none`, `auto`, `required`, and specific function
  choices by narrowing the advertised tool list to the named function.

OpenAI embeddings compatibility:

- `POST /v1/embeddings` serves native `text-embed-qwen3-0.6b` embeddings.
- Requests are bounded to 256 texts and 2 MiB of UTF-8 input. Rows are capped
  at 8,192 tokens and evaluated in length-aware batches with at most 8,192
  padded tokens before response order is restored.
- `input` may be a string or array of strings.
- `encoding_format` may be omitted or set to `float`; base64 encoding and
  dimension overrides are rejected.

OpenAI image/audio compatibility:

- `POST /v1/images/generations` serves native image generation models such as
  `image-zimage-nano`; it supports `prompt`, `size`, `n=1`, `response_format`
  `b64_json` or `url`, and local extensions such as `seed`. Each image
  dimension must be a multiple of 16 from 16 through 4,096 pixels, and total
  area is limited to 4,194,304 pixels; explicit inference steps are limited to
  1 through 100.
- `POST /v1/images/edits` accepts multipart `image` uploads, Open WebUI-style
  `image[]` repeated uploads, an optional `mask`, and an edit `prompt`; it uses
  the same image runtime with input-image conditioning and the same size
  limits. Masks are accepted for client compatibility; current native edit
  models use whole-image conditioning.
- `POST /v1/audio/speech` serves `speech-tts-qwen3-nano` and returns WAV by
  default, with `mp3`, `opus`, `aac`, and `flac` available when `ffmpeg` is
  installed. OpenAI model names such as `tts-1` map to the local default;
  normalized input plus voice instructions may total at most 32 KiB of UTF-8.
- `POST /v1/audio/transcriptions` accepts multipart uploads for
  `speech-asr-parakeet` or `speech-asr-qwen3`, with `json`, `text`,
  `verbose_json`, `srt`, and `vtt` response formats. OpenAI model names such as
  `whisper-1` map to the local default; `max_tokens` is limited to 1 through
  4,096.

Examples:

```bash
swift run mere.run api serve
swift run mere.run api serve --preflight --json
swift run mere.run api serve --engine text-chat-gemma4
swift run mere.run api serve --engine text-chat-lfm2
swift run mere.run api serve --engine text-code --model ./Qwen3-Coder-Next-Q4_K_M.gguf
swift run mere.run api serve --host 0.0.0.0 --preflight --json
swift run mere.run api serve --host 0.0.0.0 --port 11434 --api-key "$MERERUN_API_KEY" --rate-limit-per-minute 120 --max-active-requests 1
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  --data '{"model":"text-embed-qwen3-0.6b","input":"local RAG"}'
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
```

After starting a server, run `swift run mere.run status` from another terminal
to confirm `/health`, `/v1/models`, the runtime pool, and the served model.

### `mere.run setup`

Choose the public onboarding path. The default interactive command offers the
local Mere agent powered by Pi, a bring-your-own-agent handoff prompt, or manual
commands.

```bash
swift run mere.run setup
swift run mere.run setup --mode agent --agent-model small --dry-run
swift run mere.run setup --mode agent --agent-model tier --install --start
swift run mere.run setup --mode byoa
swift run mere.run setup --mode manual
```

Agent model choices:

- `small`: `text-agent-qwen35-9b`, a Qwen3.5 9B Q4 GGUF setup agent for 16 GB machines
- `tier`: the best supported local tier for this machine, currently 9B, Qwen3.6 nano, Qwen3-Coder Next, or DeepSeek V4 Flash on 96 GB+ machines
- `premier`: `text-agent-deepseek-v4-flash`, the preferred managed 96 GB+ setup-agent tier served by the bundled DS4 engine

North Mini Code (`text-code-north-mini`) is available as a managed native GGUF
coding model. It is pullable through `model pull`, can be served with
`api serve --engine text-code --model text-code-north-mini`, and can be started
through the Pi-backed `agent start` path like other `text-code` models once the
installed llama.cpp runtime supports the `cohere2moe` architecture.

Ornith (`text-agent-ornith-9b`) is available as an experimental native
MLX/OptiQ coding-agent model. It uses the Qwen-family runtime, so serve it with
`api serve --engine text-chat-q36 --model text-agent-ornith-9b`.
The local converted Ornith 35B MLX target (`text-agent-ornith-35b-mlx`) uses
the same native Qwen-family serving engine when installed.
The larger Ornith 35B GGUF target (`text-agent-ornith-35b`) is also available
for explicit evals and runs through:

```bash
swift run mere.run api serve --engine text-chat-q36 --model text-agent-ornith-35b-mlx
swift run mere.run api serve --engine text-code --model text-agent-ornith-35b
```

BYOA prints a ready-to-paste Claude/Codex prompt. Manual mode prints the
commands for capabilities, model pulls, serving, and optional Pi installation.
Pi auto-install uses the published macOS release assets; on Linux, put a `pi`
binary on `PATH` or pass `--pi-path` and the agent runs in the current terminal.

### `mere.run agent onboard`

Lower-level agent plumbing used by `mere.run setup`. Print a guided setup
summary for the current machine. Optional flags can pull the
recommended supported model package, install Pi, and write a Pi provider
extension that points at `mere.run api serve`.

```bash
swift run mere.run agent onboard
swift run mere.run agent onboard --pull-recommended --accept-model-license
swift run mere.run agent onboard --install-pi --configure-pi
swift run mere.run agent onboard --configure-pi --model text-agent-deepseek-v4-flash
swift run mere.run agent onboard --configure-pi --model text-code-north-mini --port 8080
swift run mere.run agent onboard --configure-pi --model text-agent-ornith-9b --port 8080
```

### `mere.run agent install-pi`

Install the latest `badlogic/pi-mono` release asset for the current macOS
architecture into the mere.run application-support directory.

```bash
swift run mere.run agent install-pi
```

### `mere.run agent start`

Start a local API server for a selected managed agent model and launch Pi
against the `mere-run` provider. GGUF code models use `--engine text-code`,
Qwen3.6 uses `--engine text-chat-q36`, and DeepSeek V4 Flash uses the DS4-backed
`--engine text-chat-deepseek-v4-flash`. If `--model` is omitted, `agent start`
uses the best installed startable setup agent first, then a valid persisted Pi
provider model, then the current machine's startable hardware tier. On 96 GB+
Apple Silicon Macs, DeepSeek V4 Flash is the preferred setup-agent tier; smaller
Qwen models are alternatives, not upgrades.

```bash
swift run mere.run model pull text-agent-deepseek-v4-flash
swift run mere.run agent install-pi
swift run mere.run agent start --model text-agent-deepseek-v4-flash
```

## Validation and smoke runs

Standard repo validation:

```bash
./scripts/check.sh
```

Fast smoke suite:

```bash
./scripts/e2e_smoke.sh --core
```

Installed-model sweep:

```bash
./scripts/e2e_smoke.sh --installed
```
