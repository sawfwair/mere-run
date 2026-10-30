# mere.run CLI reference

This page lists every `mere.run` command, then documents each one's options,
defaults, and output. Media commands are grouped by what they work on:
`image`, `text`, `speech`, `vision`, `geo`, `audio`, `music`, `sfx`, `video`,
and `world`. Separate families cover portable graphs, executors, run
artifacts, evaluation, models, adapters, configuration, serving, plugins, and
setup.

The `--help` output of each command is the authoritative list of its options.
For per-command advice, including prompting patterns, important flags, and
troubleshooting, read the offline [command cookbooks](/cookbooks) with
`mere.run guide <command path>`.

If you're new to mere.run, start with [Getting started](/getting-started).

## Overview

To print the top-level command list, run:

```bash
swift run mere.run --help
```

The following tree is generated from the binary. The repository tests it
against the CLI and fails if they disagree.

<!-- BEGIN GENERATED: CLI TREE -->
- [`mere.run guide`](/cookbooks) — Read offline mere.run command cookbooks.
- [`mere.run catalog`](/cli) — Inspect the machine-readable command capability contract.
  - `mere.run catalog show` — Print the capability contract, or one capability by id (the default).
  - `mere.run catalog resolve` — Show which runtime family a command line would run, and the options it rejects or ignores.
- [`mere.run image`](/runtime/image) — Generate and validate image models.
  - `mere.run image dataset` — Inspect image training datasets.
    - `mere.run image dataset discover` — Find image-caption dataset candidates under a root directory.
  - `mere.run image generate` — Generate images with local image models.
  - `mere.run image reconstruct-3d` — Reconstruct a colored object mesh from one image with native TripoSR.
  - `mere.run image reconstruct-3d-trellis2` — Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2.
  - `mere.run image reconstruct-3d-multiview` — Reconstruct a colored mesh from four or six supplied views with native InstantMesh.
  - `mere.run image run-plan` — Run a saved image workflow plan.
  - `mere.run image train-lora` — Train a local image LoRA adapter.
  - `mere.run image visualize-run` — Open a local LoRA training run viewer.
  - `mere.run image validate` — Run advanced deterministic validation for local image model families.
- [`mere.run text`](/runtime/text) — Run local chat, code, embedding, anonymization, classification, and decision workflows.
  - `mere.run text chat` — Run local chat with text chat models.
  - `mere.run text code` — Run local code generation with GGUF models via llama.cpp.
  - `mere.run text embed` — Generate native text, multimodal, or PPLX multi-vector embeddings.
  - `mere.run text anonymize` — Detect and redact PII using OpenAI Privacy Filter.
  - `mere.run text decide` — Evaluate typed questions with native Laya, Clef, or D1 decision models.
  - `mere.run text classify` — Classify text with native GLiNER2.5 Decide.
  - `mere.run text extract` — Extract entities, relations, and structures with native GLiNER2.5 Decide.
  - `mere.run text train-lora` — Train a native text or image-conditioned LoRA adapter from chat-style SFT JSONL.
- [`mere.run speech`](/runtime/speech) — Synthesize, transcribe, diarize, and manage voice profiles.
  - `mere.run speech synthesize` — Generate speech from text using a native TTS model.
  - `mere.run speech transcribe` — Transcribe or translate speech to text using native ASR backends.
  - `mere.run speech diarize` — Identify who spoke when in an audio file with native MLX diarization.
  - `mere.run speech diarize-live` — Stream Nemotron 3 speaker activity from a microphone or raw PCM stdin.
  - `mere.run speech listen` — Transcribe a macOS microphone with live Qwen or Whistle ASR.
  - `mere.run speech profile` — Manage saved voice clone profiles.
    - `mere.run speech profile list` — List saved speech voice profiles.
    - `mere.run speech profile create` — Create a speech profile from reference audio.
    - `mere.run speech profile delete` — Delete a speech profile by ID.
- [`mere.run vision`](/runtime/vision) — Embed, caption, inspect, face-analyze, segment, track, pose, depth, geometry, optical flow, and OCR visual media.
  - `mere.run vision embed` — Generate shared text/image embeddings with native Qwen3-VL.
  - `mere.run vision caption` — Generate training-friendly captions for images.
  - `mere.run vision inspect` — Describe or answer questions about an image using Qwen3-VL.
  - `mere.run vision face` — Detect faces, create identity embeddings, and compare faces locally.
    - `mere.run vision face detect` — Detect faces and five-point landmarks in an image.
    - `mere.run vision face embed` — Create a normalized ArcFace embedding for one face in an image.
    - `mere.run vision face compare` — Compare one face from each of two images with cosine similarity.
    - `mere.run vision face batch` — Analyze many images with one warm detector/recognizer session and emit JSONL.
  - `mere.run vision ground` — Ground text queries in an image with the native Falcon Perception runtime.
  - `mere.run vision serve` — Serve resident, binary-frame vision grounding over HTTP.
  - `mere.run vision segment` — Segment prompted objects in an image with the native SAM 3.1 runtime.
  - `mere.run vision track` — Track prompted objects through a video with the native SAM 3.1 runtime.
  - `mere.run vision track-live` — Capture from a camera and track text-prompted objects with the native SAM 3.1 runtime.
  - `mere.run vision pose` — Detect body, hand, and face landmarks in an image with the native platform runtime.
  - `mere.run vision flow` — Generate dense optical flow between two equal-size images.
  - `mere.run vision depth` — Estimate affine-invariant depth for a still image with native Marigold V2.
  - `mere.run vision depth-video` — Generate temporally consistent relative or metric video depth with native VDA-S.
  - `mere.run vision geometry` — Generate metric depth, normals, camera intrinsics, and a point cloud with native MoGe-2.
  - `mere.run vision geometry-multiview` — Solve native DA3-Small multi-view relative geometry, confidence, and cameras.
  - `mere.run vision image-to-3d` — Reconstruct a colored object mesh from one image with native TripoSR.
  - `mere.run vision image-to-3d-trellis2` — Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2.
  - `mere.run vision image-to-3d-multiview` — VFX alias for native four-view or six-view InstantMesh reconstruction.
  - `mere.run vision ocr` — Extract text from images using LightOnOCR, GLM-OCR, or Infinity-Parser2.
- [`mere.run geo`](/runtime/geo) — Run native geospatial inference models on local Earth-observation data.
  - `mere.run geo flood` — Run native TerraMind Flood tile inference with MLX on Apple Silicon.
  - `mere.run geo fire` — Run native TerraMind Fire tile inference with MLX on Apple Silicon.
  - `mere.run geo tessera` — Encode local Sentinel-1/2 time series with a native TESSERA v2 student.
  - `mere.run geo olmoearth` — Encode multisensor Earth observations with native OlmoEarth v1.2.
- [`mere.run audio`](/runtime/audio) — Generate, edit, and enhance audio locally.
  - `mere.run audio generate` — Generate audio from text with the native LTX-2.5 audio-only model.
  - `mere.run audio enhance` — Extend speech or general-audio bandwidth to 48 kHz.
  - `mere.run audio edit` — Generate or edit speech with native AuK instruction conditioning.
- [`mere.run music`](/runtime/music) — Generate, analyze, transcribe, and separate music locally.
  - `mere.run music analyze` — Analyze source audio with ACE-Step audio understanding.
  - `mere.run music generate` — Generate audio from a music prompt.
  - `mere.run music realtime` — Run Magenta RealTime 2 music generation.
  - `mere.run music serve` — Start an ACE-Step or MiniMax Music 3 generation API.
  - `mere.run music separate` — Separate or restore audio with native RoFormer models.
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
  - `mere.run video cosmos3` — Run native NVIDIA Cosmos3 generation and action modes.
  - `mere.run video dub-it` — Generate synchronized video and audio identity from one LTX 2.5 IC-LoRA reference.
  - `mere.run video export-latents` — Run native Swift/MLX distilled LTX denoising and export final latents.
  - `mere.run video generate` — Generate MP4 video with native Swift/MLX video models.
  - `mere.run video prepare-masks` — Prepare reviewable, palette-safe SCAIL-2 masks with native SAM 3.1.
  - `mere.run video retake` — Regenerate a timed video/audio region with native LTX 2.5.
  - `mere.run video session` — Keep an LTX 2.3 or LTX 2.5 runtime resident for JSONL generation requests.
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
  - `mere.run graph batch` — Distribute independent complete workflow graphs across a relay fleet.
    - `mere.run graph batch list` — List durable batches in the relay account.
    - `mere.run graph batch submit` — Prepare and submit a durable workflow batch.
    - `mere.run graph batch inspect` — Inspect batch progress and item output references.
    - `mere.run graph batch watch` — Watch batch changes as NDJSON until terminal.
    - `mere.run graph batch cancel` — Stop admission and cancel unfinished batch children.
    - `mere.run graph batch retry` — Retry failed items while retaining completed outputs.
    - `mere.run graph batch fetch` — Fetch verified outputs from completed batch items.
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
- [`mere.run relay`](/workflows#direct-relay-relay-serve) — Host the relay API surface directly on this machine.
  - `mere.run relay serve` — Serve the relay graph-job API from this machine and run submitted jobs locally.
- [`mere.run run`](/workflows#run-directories) — Inspect durable mere.run workflow reports and run directories.
  - `mere.run run list` — Find durable run directories, structured reports, and run plans under a root.
  - `mere.run run inspect` — Inspect a run directory, structured report, or run plan.
  - `mere.run run watch` — Watch the worker event stream for an SSH or relay graph job.
  - `mere.run run fetch` — Fetch a remote graph run into the standard local run-directory format.
  - `mere.run run cancel` — Cancel a graph run.
  - `mere.run run retry` — Retry a recorded image or transcription run, or an immutable relay graph job.
- [`mere.run eval`](/evaluation-packs) — Run reproducible evaluations from external, content-addressed packs.
  - `mere.run eval pack` — Inspect external evaluation packs without running models.
    - `mere.run eval pack validate` — Validate and hash an external evaluation pack.
  - `mere.run eval run` — Run a matched, adapter-aware evaluation from an external pack.
  - `mere.run eval promote` — Create a content-addressed receipt for a complete, gate-passing report.
- [`mere.run model`](/runtime/model-management) — List, pull, locate, remove, inspect, optimize, and clean up models.
  - `mere.run model list` — List all known models with install status.
  - `mere.run model location` — Manage read-only model catalog locations.
    - `mere.run model location list` — List the writable store, search roots, and explicit bindings.
    - `mere.run model location add` — Register a read-only root containing directories named for canonical model IDs.
    - `mere.run model location remove` — Unregister a search root without deleting its files.
    - `mere.run model location bind` — Bind a canonical model ID to an arbitrary read-only directory.
    - `mere.run model location unbind` — Remove explicit bindings without deleting model files.
  - `mere.run model pull` — Download a managed model into the local model store.
  - `mere.run model remove` — Remove a model from the local model store.
  - `mere.run model storage` — Inspect physical model storage, sharing, and reclaimable space.
  - `mere.run model gc` — Find or delete unreferenced model payloads and partial downloads.
  - `mere.run model info` — Print a model's manifest, validation status, and resolved component paths.
  - `mere.run model capabilities` — Show which managed models this machine can run.
  - `mere.run model runtime` — Read and update per-model API runtime settings.
    - `mere.run model runtime get` — Print typed API runtime settings for a managed model.
    - `mere.run model runtime set` — Update typed API runtime settings for a managed model.
  - `mere.run model optimize` — Build inference-only caches for a supported installed model.
  - `mere.run model benchmark` — Run focused local model benchmarks.
    - `mere.run model benchmark chat` — Run a small grounded-chat evaluation slice against local assistant models.
    - `mere.run model benchmark fused` — Run the versioned Mere Lite or Mere Comprehensive fused quality suite.
    - `mere.run model benchmark fused-fixture` — Stamp or verify normalized fused-benchmark JSONL fixture hashes.
    - `mere.run model benchmark tool-calls` — Run a small tool-call selection evaluation against local chat models.
    - `mere.run model benchmark tool-continuations` — Evaluate Gemma 4 continuation after completed tool calls.
    - `mere.run model benchmark code` — Run a real coding-evaluation slice against local coding models.
    - `mere.run model benchmark gemma4-kv` — Compare Gemma4 default KV cache decode against packed PolarKV.
    - `mere.run model benchmark gemma4-mtp` — Compare Gemma4 serial decode against verified MTP speculative decode.
    - `mere.run model benchmark q36-mtp` — Compare Qwen-family serial decode against adaptive and forced MTP speculative decode.
    - `mere.run model benchmark q38-verification` — Measure the Qwen-family target-only verification frontier.
    - `mere.run model benchmark kolibri-logprobs` — Score fixed token sequences with native Kolibri and export raw logits.
    - `mere.run model benchmark laguna-dflash` — Measure Laguna target-only and DFlash decode in one resident process.
    - `mere.run model benchmark parakeet-coreml` — Benchmark the prepared Parakeet Core ML pipeline in one resident process.
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
  - `mere.run plugin run` — Run an installed plugin without changing PATH.
  - `mere.run plugin rollback` — Restore a retained signed plugin bundle.
- [`mere.run setup`](/getting-started) — Choose a guided, BYOA, or manual mere.run setup path.
- [`mere.run agent`](/getting-started) — Install and start the optional guided local setup agent.
  - `mere.run agent onboard` — Summarize this machine's model capabilities and prepare the optional Pi agent.
  - `mere.run agent status` — Inspect local agent, Pi, provider, and model readiness.
  - `mere.run agent install-pi` — Install the most recent published Pi coding-agent release for use with mere.run.
  - `mere.run agent start` — Start Pi against a local mere.run setup-agent API server.
<!-- END GENERATED: CLI TREE -->

## Global model-store override

To use a different model store for one invocation, pass the global
`--models-root` option before the command:

```bash
swift run mere.run --models-root /Volumes/FastSSD/mererun-models model list
```

The option is equivalent to setting `MERERUN_MODELS_DIR`:

```bash
export MERERUN_MODELS_DIR=/Volumes/FastSSD/mererun-models
```

## Model scope check

Before a command queues for machine admission, resolves its model, or downloads
anything, the CLI checks the command line against the capability contract. It
picks the runtime family the command runs from the model options, the
command's default model, and any selector flags, and then:

- stops with the reason and the command to use when the managed model can't
  run this command, for example an ACE-Step language model passed to
  `music analyze --model`;
- stops when the family rejects an option;
- prints a `Warning:` line on stderr when the family ignores an option, and
  runs.

It reads the command line the way the parser does: a repeated single-value
option counts once, with its last value; grouped short options such as `-qm`
and `-m=<id>` spell the same options; and a value the command reads as not
passed, such as an empty `--negative-prompt` or a blank `text chat --image`,
counts as omitted.

A local model folder, an upstream repository id, or a managed model whose
checkpoint depends on what is installed is identified the way the command
identifies it, from its files, and checked as that family. One the CLI can't
identify passes, and the command checks it when it loads. The parser's own
requests (`--help`, `-h`, `-help`, `--experimental-dump-help`, `--version`,
and completion scripts) and listing flags such as `--list-devices` are never
checked.

A refusal exits with status 64 and prints the usage line, like the commands'
own option errors. It comes before any preflight work, so under
`--preflight --json` a refused command line prints no JSON report; the error is
on stderr. Warnings print after the whole command line has parsed and validated,
so a run that fails validation shows only its error, and `--quiet` leaves them
out.

A command that prints a JSON object on stdout also carries the warnings in it,
under `--quiet` too, as a top-level `warnings` array of the same sentences:

```json
{"event":"result","exit":0,"outputs":[{"kind":"text","path":"/abs/talk.txt"}],"warnings":["--max-tokens has no effect with Parakeet. It applies to Qwen3-ASR."]}
```

- It covers `--preflight --json` and `--dry-run --json` reports, `--json`
  results, `--receipt` lines, and the commands whose result is a JSON object
  (`audio edit`, `audio enhance`, `music analyze`, `music separate`,
  `sfx clap score`, `speech diarize --format json`, `text anonymize --json`,
  `text decide`, `text embed`, and `vision embed`).
- With no warnings the key is left out, so the JSON is unchanged.
- Only stdout carries them. A file the command writes, such as an `--output`
  copy or the `audio enhance` and `music separate` manifests, keeps the result
  alone.
- JSON Lines streams (`--jsonl`, `video session`), servers,
  and outputs that are the product itself rather than a report (`music
  transcribe --format json`, `vision ocr`) are unchanged. So is
  `video prepare-masks --json`, whose `warnings` counts mask-quality warnings;
  its single model has no options it ignores.

To see the decision without running anything, use
[`mere.run catalog resolve`](#mere-run-catalog-resolve).

## Canonical managed model IDs

For the full source story, including which IDs are pullable from Hugging Face,
see [Model sources](./model-sources.md). To list every managed ID with its
install state, run `mere.run model list`. Common managed IDs by area:

- Images: `image-flux1-dev`, `image-flux2-dev`, `image-flux3-klein`,
  `image-klein-nano`, `image-klein-9b`, `image-klein-base`,
  `image-klein-base-9b`, `image-klein-max`, `image-bonsai-binary`,
  `image-bonsai-ternary`, `image-zimage-nano`, `image-zimage-base`,
  `image-zimage-max`, `image-hidream-o1`, `image-hidream-o1-dev`,
  `image-sensenova-u1-5-8b-mot`, `image-krea2-raw`, `image-krea2-turbo`,
  `image-qwen-21`, `image-qwen-21-turbo`, `image-qwen-edit-2511`,
  `image-qwen-edit-2511-lightning`, `image-ideogram4-sdnq-uint4`
- Image to 3D: `image-3d-triposr`, `image-3d-trellis2-4b`,
  `image-3d-instantmesh-base`
- Text chat: `text-chat-gemma4`, `text-chat-gemma4-12b-4bit`,
  `text-chat-diffusiongemma-26b-optiq-4bit`, `text-chat-laguna-s-2-1`,
  `text-chat-laguna-xs-2-1`, `text-chat-inkling-small`,
  `text-chat-flint-4bit`, `text-chat-mebot`, `text-chat-psi-agent`,
  `text-chat-q36-nano`, `text-chat-bonsai-27b-1bit`,
  `text-chat-nemotron-35-lightning`, `text-chat-lfm25-2.6b-4bit`,
  `text-chat-lfm25-a1b-8bit`, `text-chat-lfm25-24b-a2b-bf16`
- Vision chat: `vision-chat-gemma4-12b`, `vision-chat-q38-27b`,
  `vision-chat-q38-27b-4bit`, `vision-chat-q38-flash-next-4bit`,
  `vision-chat-ornith-35b`, `vision-chat-muse-glimmer-30b`,
  `vision-chat-lfm25-3b-8bit`, `vision-chat-lfm25-3b-bf16`,
  `omni-chat-nemotron3-nano-30b-a3b-bf16`
- Text code and agents: `text-agent-qwen35-9b`, `text-agent-ornith-9b`,
  `text-agent-ornith-35b-mlx-4bit`, `text-agent-ornith-35b-mlx-6bit`,
  `text-agent-ornith-35b-mlx-8bit`, `text-agent-ornith-35b-mlx`,
  `text-agent-ornith-35b`, `text-agent-deepseek-v4-flash`,
  `text-code-north-mini`, `text-code-qwen3`
- Text embed: `text-embed-qwen3-0.6b`, `text-embed-embeddinggemma2`,
  `text-embed-pplx-v2-late-0.6b`, `text-embed-pplx-v2-late-9b`,
  `text-embed-pplx-v2-context-9b-preview`
- Text decide, classify, and anonymize: `text-decide-laya`,
  `text-decide-clef-4bit`, `text-decide-d1-3b-bf16`,
  `text-classify-gliner25-decide`, `text-anonymize-privacy-filter`
- Multimodal embed: `vision-embed-qwen3-vl-2b`
- Speech TTS: `speech-tts-qwen3-nano`, `speech-tts-qwen3-customvoice`,
  `speech-tts-breeze-2`, `speech-tts-phonon-31m`, `speech-tts-drama-3`
- Speech ASR: `speech-asr-parakeet`, `speech-asr-qwen3`, `speech-asr-whistle`,
  `speech-asr-nemotron-35-transcribe`,
  `speech-asr-nemotron-35-transcribe-bf16`
- Speaker diarization: `speech-diarization-sortformer`,
  `speech-diarization-nemotron3`
- Vision OCR: `vision-ocr-lighton`, `vision-ocr-lighton-3-0.8b`,
  `vision-ocr-lighton-3-4b`, `vision-ocr-infinity-pro-int8`,
  `vision-ocr-infinity-pro`
- Vision segmentation and tracking: `vision-segment-sam31`
- Vision grounding: `vision-ground-falcon-perception`
- Depth and geometry: `vision-depth-marigold-v2`, `vision-depth-vda-small`,
  `vision-depth-vda-small-metric`, `vision-geometry-moge2-small`,
  `vision-geometry-da3-small`
- Face detection and identity embeddings: `vision-face-buffalo-l`
- Geospatial segmentation: `vision-flood-terramind-base`, `vision-fire-terramind-base`
- Geospatial time-series embeddings: `vision-embed-tessera-v2-nano`,
  `vision-embed-tessera-v2-small`, `vision-embed-tessera-v2-medium`,
  `vision-embed-tessera-v2-large`, `vision-embed-tessera-v2-teacher`
- Geospatial multisensor embeddings: `vision-embed-olmoearth-v12-nano`,
  `vision-embed-olmoearth-v12-tiny`, `vision-embed-olmoearth-v12-small`,
  `vision-embed-olmoearth-v12-base`
- Audio: `audio-auk-base`, `audio-auk-flash`,
  `audio-enhance-ap-bwe-16kto48k`, `audio-enhance-universr-audio`
- Music: `music-acestep`, `music-acestep-xl-turbo`,
  `music-acestep-xl-turbo-lm4b`, `music-acestep-xl-sft`,
  `music-acestep-xl-base`, `music-acestep-lm-1.7b`, `music-acestep-lm-4b`,
  `music-minimax-music3`, `music-yue2`, `music-magenta-rt2-small`,
  `music-magenta-rt2-base`
- Music transcription and separation: `music-muscriptor-small`,
  `music-muscriptor-medium`, `music-muscriptor-large`,
  `music-separate-bs-roformer-viperx-1297`, `music-separate-bs-roformer-4stem`,
  `music-separate-mel-roformer-dereverb`, `music-separate-mel-roformer-denoise`
- SFX: `sfx-woosh-dflow`, `sfx-woosh-flow`, `sfx-woosh-clap`,
  `sfx-woosh-synchformer`, `sfx-woosh-vflow-8s`, `sfx-woosh-dvflow-8s`,
  `sfx-mmaudio-large-44k-v2`
- Video: `video-ltx-av`, `video-ltx23-av-mlx`, `video-ltx23-full-mlx`,
  `video-ltx23-a2vid-mlx`, `video-ltx25-distilled-bf16`,
  `video-ltx25-full-bf16`, `video-wan22-ti2v-5b-mlx`,
  `video-minimax-h3-fl2va-bf16-mlx`, `video-minimax-h3-ref2va-mlx`,
  `video-minimax-h3-fasth3-vsa-datafree-mlx`, `video-cosmos3-edge-mlx`,
  `video-scail2-14b-mlx`
- World models: `video-dreamx-world-5b-ar-mlx`

For subsystem-specific implementation guides, see the runtime pages:

- [Image runtime](./runtime/image.md)
- [Text runtime](./runtime/text.md)
- [Speech runtime](./runtime/speech.md)
- [Vision runtime](./runtime/vision.md)
- [Geospatial runtime](./runtime/geo.md)
- [Audio runtime](./runtime/audio.md)
- [Music runtime](./runtime/music.md)
- [SFX runtime](./runtime/sfx.md)
- [Video runtime](./runtime/video.md)
- [Persistent world runtime](./runtime/world.md)

## Common workflows

### Pull and inspect models

```bash
swift run mere.run model list
swift run mere.run status
swift run mere.run model capabilities
swift run mere.run model runtime get text-chat-gemma4
swift run mere.run model pull image-zimage-nano
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
  --model music-acestep-xl-sft \
  --lm-model music-acestep-lm-1.7b

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
executors. See [Portable workflows](./workflows.md).

### Start a persistent world session

```bash
mere.run world serve \
  --base-model video-wan22-ti2v-5b-mlx \
  --model video-dreamx-world-5b-ar-mlx \
  --prepare
```

For the HTTP lifecycle, camera controls, state artifacts, and authentication
boundary, see [Persistent world runtime](./runtime/world.md).

### Serve a local API

```bash
swift run mere.run api serve --engine text-chat-gemma4
```

### Install a companion plugin

Official companion plugins are distributed outside the Swift package. The CLI
reads the live catalog from `sawfwair/mere-run-plugins`, prints the exact install
command by default, and runs it only when you pass `--yes`.

```bash
swift run mere.run plugin list
swift run mere.run plugin info mere-runpod
swift run mere.run plugin install mere-runpod
swift run mere.run plugin install mere-runpod --yes
swift run mere.run plugin doctor mere-runpod
```

For installation verification and the typed graph-provider boundary, see
[Companion plugins](./plugins.md).

## Command reference

Model installation in the open-source repository is explicit. `mere.run model
pull` uses only cataloged Hugging Face snapshots. Supply local-path-only models
with the command-specific `--model` or `--model-root` options. See
[Configuration](./configuration.md) and [Model sources](./model-sources.md).

Public LoRA releases use the separate checksum-pinned adapter catalog:

```bash
mere.run adapter list
mere.run adapter pull mere-platform-assistant
mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora mere-platform-assistant \
  --prompt "Summarize the active project workspace."
```

### `mere.run guide`

Read the offline command cookbooks and model handbooks bundled with the CLI.
Pass a command path to read its cookbook.

```bash
mere.run guide --list
mere.run guide music generate
mere.run guide music generate --model music-minimax-music3
mere.run guide --list-models
```

- `--list`: list guide topics; add `--markdown` to render the list as a
  Markdown table
- `--list-models`: list model handbook families and their managed model ids
- `--model <id>`: focus the guide on one managed model
- `--json`: emit JSON instead of Markdown

For the topic list, see [Cookbooks](/cookbooks).

### `mere.run catalog show`

Print the machine-readable command capability contract, or one capability by
id. `mere.run catalog` with no subcommand runs `catalog show`.

```bash
mere.run catalog show
mere.run catalog show video.generate --json
```

`--json` emits stable JSON. For what the contract declares about receipts,
progress, and outputs, see
[Machine-readable receipts and progress](#machine-readable-receipts-and-progress).

### `mere.run catalog resolve`

Report which runtime family a command line runs, and what the
[model scope check](#model-scope-check) decides for it. Pass the command line
after `--`, without `mere.run`:

```bash
mere.run catalog resolve -- music analyze song.wav --model music-acestep
mere.run catalog resolve --json -- music analyze song.wav --model music-acestep-lm-4b
```

Nothing is loaded, downloaded, or admitted. The report names:

- `capability`: the capability id, such as `music.analyze`;
- `family` and `family_title`: the runtime family, when one is known;
- `model`: the model the family runs, when one is known;
- `source`: how the family was chosen. `model`, `default`, and `selector` come
  from the command line; `identified` from inspecting a local model;
  `unidentified` means a local model or unlisted id the CLI checks only when it
  loads; `excluded` is a managed model that can't run the command; `unmatched`
  is a model whose selector flags fit no family; `unrouted` is a command that
  loads no model, or a model it would refuse behind a listing flag such as
  `--list-devices`;
- `violations`: why the command would stop, empty when it runs;
- `warnings`: options the family would run without.

A listing flag such as `speech listen --list-devices` answers before the
command reads any other option, so its command line reports no violations or
warnings.

`--json` prints the report as the `MereRunFamilyResolutionReport` type from
`MereRunContract`.

### `mere.run plugin`

Discover and install official companion plugins from the live plugin catalog.
Plugins are separate executables, not code loaded into the `mere.run` process.

```bash
swift run mere.run plugin list
swift run mere.run plugin info mere-runpod
swift run mere.run plugin install mere-runpod --yes
swift run mere.run plugin doctor mere-runpod
```

| Command | What it does |
| --- | --- |
| `plugin list` | List official plugins from the catalog. |
| `plugin info <id>` | Show one plugin's catalog entry and install command. |
| `plugin install <id>` | Print the exact install command; with `--yes`, run it. |
| `plugin doctor <id>` | Run an installed plugin's doctor command. |
| `plugin run <entrypoint> [<arguments> ...]` | Run an installed plugin without changing `PATH`, forwarding the arguments. |
| `plugin rollback <id>` | Print the plan to restore a retained signed plugin bundle; with `--yes`, activate it. |

Options:

- `--catalog-url`: plugin catalog URL or local JSON path (`list`, `info`,
  `install`, `doctor`)
- `--json`: print catalog or plugin metadata as JSON (`list`, `info`)
- `--channel`: install channel; defaults to the catalog's default channel
  (`info`, `install`)
- `--yes`: run the install or rollback; without it, the command only prints
  what it would do
- `--force`: pass `--force` to `pipx install`
- `--source`: install from the catalog's pipx source instead of an advertised
  signed bundle
- `--bundle-manifest`, `--bundle-archive`: an explicit signed bundle manifest
  URL or path, or a local signed bundle DMG for offline installation.
  Publisher verification still applies, and the archive's signed hash must
  match.

### `mere.run image generate`

Generate a PNG with a local image model.

```bash
swift run mere.run image generate --prompt "<text>" [options]
```

Key options:

- `--prompt`, `-p`: required text prompt
- `--negative-prompt`, `-n`: negative prompt, used when `--cfg` is above 1.0
- `--model`, `-m`: canonical model id or local model path; defaults to
  `image-zimage-nano`
- `--output`, `-o`: output PNG path; defaults to
  `./mererun-image-<timestamp>.png`
- `--width`, `-W` and `--height`, `-H`: both default to `1024`
- `--steps`, `-s`: override the model-specific step default
- `--cfg` (alias `--cfg-scale`): override the model-specific CFG default
- `--seed`: random seed (UInt64)
- `--sigma-shift`: FlowMatch schedule shift
- `--max-sequence-length`: maximum text sequence length; defaults to `512`
- `--input`, `-i`: image-to-image source. SenseNova U1.5 uses it as an editing reference.
- `--mask`: black-and-white edit mask for `--input`. White pixels can change;
  black pixels are restored from the input.
- `--outpaint top,right,bottom,left`: expand the editable canvas by this
  padding inside `--width` and `--height`
- `--mask-feather`: blend radius in pixels at mask and outpaint edges;
  defaults to `8`
- `--ref-image`: repeatable Klein, Qwen Edit, HiDream O1, or SenseNova U1.5
  reference image
- `--keep-original-aspect`: preserve one HiDream reference image's aspect ratio
- `--strength` (alias `--str`): image-to-image/reference change strength from
  0.0 through 1.0. Defaults to 0.75 for `--input` and 0.0 for Klein
  `--ref-image`.
- `--structured-prompt`, `--json-prompt`: expand the prompt into a structured JSON caption with a local text chat model before image generation
- `--structured-prompt-model`: text chat model id for the adapter; defaults to
  `text-chat-gemma4-12b-4bit`. `--structured-prompt-model-root` points at a
  local root, and `--structured-prompt-max-tokens` (default `2048`) caps the
  adapter's output.
- `--structured-prompt-output`: write the generated structured JSON caption to a file
- `--lora PATH_OR_ID[=SCALE]`: load an image LoRA. Repeat it to stack ordered,
  independently scaled FLUX.1 or FLUX.2 adapters.
- `--lora-scale`: default scale for `--lora` values without an inline scale
- `--sigmas`: pre-shifted, descending FLUX.2 sigma values; a terminal zero is
  optional
- `--krea-base-quantization-bits 4|8`: quantize the frozen Krea transformer and
  load the text encoder, transformer, and VAE sequentially to reduce peak memory
- `--krea-conditioning-multiplier`, `--krea-conditioning-layer-weights`:
  experimental Krea 2 text-conditioning controls
- `--preflight`: inspect the generation request without loading the model or writing an image
- `--json`: with `--preflight`, emit a structured JSON report
- `--run-dir`: create a new [image run directory](/runtime/image#record-and-retry-an-image-run) with settings, input snapshots, and output
- `--quiet`, `-q`: print only the output path
- `--progress-json`: stream progress to stderr as JSON lines instead of
  human-readable text, one object per event, for example
  `{"event":"progress","stage":"denoising","step":2,"total_steps":4}`.
  `step` is the generator's 0-based step index; each stage emits a final event
  with `step == total_steps`. Takes precedence over `--quiet` for progress
  output, so wrappers can keep stdout limited to the output path while still
  observing per-step progress.
- `--receipt`: append a final `{"event":"result",...}` JSON line to stdout
  listing the PNG and, with `--structured-prompt-output`, the prompt JSON; see
  [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

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
  --model image-sensenova-u1-5-8b-mot \
  --prompt "replace the background with a snowy mountain pass" \
  --input ./subject.png \
  --width 2048 --height 2048 \
  --output ./sensenova-edit.png
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
and can be used with Klein generation through `image generate --lora`.

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

For short Krea smoke runs, `krea-fast-style` trains on `image-krea2-raw` for
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

If your images live under a larger project tree, find trainable dataset
folders first with [`image dataset discover`](#mere-run-image-dataset-discover).

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

- `--data`, `-d`: dataset directory with image + caption pairs
- `--output`, `-o`: required output `.safetensors` adapter path
- `--model`, `-m`: Raw/base model id or local model path; defaults to `image-krea2-raw`
- `--width`, `--height`: fixed training resolution; must be divisible by 16
- `--training-steps`, `--steps`: optimizer steps
- `--batch-size`: defaults to `1`
- `--learning-rate`, `--lr`
- `--rank`, `--alpha`: LoRA rank and alpha; alpha defaults to the rank
- `--max-text-length`: maximum prompt tokens; defaults to `512`
- `--scheduler-steps`: FlowMatch training timesteps; defaults to `1000`
- `--caption-dropout`: caption dropout probability from 0.0 through 1.0
- `--seed`: defaults to wall-clock time when omitted or `0`
- `--recipe`: named training recipe: `krea-fast-style`, `krea-cinematic-style`,
  or `klein-fast-style`
- `--lr-warmup-steps`, `--no-cosine-scheduler`, `--lr-min-factor`: Krea/Klein
  LR scheduler controls
- `--lite`: train only attention Q/V layers to reduce memory
- `--base-quantization-bits 4|8`: quantize the frozen Krea base transformer while training
- `--exclude-preview-images`: ignore `preview*.png`, `.jpg`, and `.webp`
  files in the dataset
- `--no-compile`: disable compiled train-step graphs to reduce peak GPU memory
- `--preflight`: inspect the training request without running training
- `--json`: with `--preflight`, emit a structured JSON report
- `--visualize`: start a loopback LoRA training dashboard for the run
- `--visualize-port`: port for the local dashboard; defaults to `8787`
- `--quiet`, `-q`: print only the output path

FLUX.2 Klein training also accepts:

- `--checkpoint-interval`: save intermediate adapters every N steps
- `--resume-from`: resume from a `.safetensors` checkpoint or checkpoint
  archive
- `--max-resolution`: preserve source aspect ratio up to a maximum side length
- `--progressive`: progressive resolution schedule up to `--width` and
  `--height`
- `--low-ram`: use the disk-backed latent cache
- `--gradient-checkpointing`: checkpoint transformer blocks during backprop to
  reduce peak GPU memory
- `--lora-target-preset`: exact target preset, such as `fal-klein-fast`
- `--lora-target-mode`: `suffix` or `transformer-linear-walk`
- `--lora-target-ranks`, `--lora-rank-preset`: per-suffix rank map, such as
  `.attn.to_q=128,.ff.linear_in=64`, or the `flux2-style-128` preset
- `--timestep-sampling`, `--timestep-loss-weighting`, `--loss-weighting`,
  `--timestep-low`, `--timestep-high`: timestep sampling and loss weighting
- `--adam-weight-decay`: AdamW weight decay
- `--sample-interval`, `--sample-prompt`, `--sample-model` (default
  `image-klein-9b`), `--sample-steps` (default `8`), `--sample-cfg`,
  `--sample-lora-scale`, `--sample-seed`: preview images during training
- `--benchmark-steps`, `--benchmark-warmup-steps`: time training steps and
  skip saving
- `--synthetic-samples`: train on synthetic samples for runtime smoke tests

### `mere.run image dataset discover`

Find image-caption dataset folders under a project root before you preflight a
training run. The command doesn't load models.

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

Key options:

- `--root`: required root directory to scan
- `--max-depth`: child-directory depth to scan; defaults to `4`
- `--min-usable-pairs`: usable image-caption pairs a candidate needs to be
  trainable; defaults to `1`
- `--training-output-root`, `--training-model`, `--training-recipe`: options
  for the emitted per-candidate `train-lora` preflight commands
- `--exclude-preview-images`: ignore `preview*` images
- `--json`: emit a structured JSON report

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
run-plan JSON files under a workspace root. Wrappers that don't know the exact
run, report, or plan path start here. `--max-depth` defaults to 4. With
`--executor relay:<profile>`, the command lists remote relay jobs instead, up
to `--limit` (default 50).

```bash
swift run mere.run run list --root ./runs --json
swift run mere.run run list --root . --max-depth 5 --json
swift run mere.run run list --root ./render.plan.json --json
swift run mere.run run list --executor relay:fleet --limit 20 --json
```

Listing and inspection read finished image and transcription records without
creating lock files, so a read-only archive lists cleanly; only an abandoned
nonterminal record in a writable directory is rewritten as interrupted.
Each entry includes its kind, relative path, inspection status, run state where
available, created/updated timestamps where known, command/format details,
diagnostic counts, and declarative actions for `run inspect --json`. The
top-level report also includes an
`inspect-selected` action for UI pickers. Legacy/plugin run folders are listed
as warning-level entries when their `run.json` can be identified but is not a
native mere.run training manifest.

### `mere.run run retry`

Retry a recorded image or transcription run, or an immutable relay graph job.
The reference is a local run directory, a record JSON path, or a `relay://`
job reference.

```bash
swift run mere.run run retry ./runs/camera
```

For a local run, the command creates a sibling directory and prints its path;
add `--json` to print the new record. For input checks and replay limits, see
[Record and retry an image run](/runtime/image#record-and-retry-an-image-run).
A `relay://` reference retries the immutable graph job.

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
directories also include a compact `metrics` object with a loss CSV summary,
most recent loss, step range, sample image count, checkpoint count, and adapter
count.

### `mere.run run watch`, `run fetch`, and `run cancel`

Follow, retrieve, and stop remote graph jobs by their `ssh://` or `relay://`
reference.

```bash
mere.run run watch relay://fleet/<job-id> --json-stream
mere.run run fetch relay://fleet/<job-id> --into ./runs/fetched
mere.run run cancel relay://fleet/<job-id> --json
```

- `run watch <ref>`: poll the worker event stream every `--poll-interval`
  seconds (default `2`). `--json-stream` prints worker NDJSON events, and
  `--json` prints one final job object.
- `run fetch <ref> --into <dir>`: fetch final outputs and reports into the
  standard local run-directory format. `--all-artifacts` adds intermediate
  artifacts, and repeatable `--artifact <name>` fetches named ones.
- `run cancel <ref>`: cancel the run.

For job references and the remote lifecycle, see
[Remote lifecycle](/workflows#remote-lifecycle).

### `mere.run image visualize-run`

Open the local LoRA training dashboard that `train-lora --visualize` starts,
for an existing run directory. The viewer serves on loopback port `8787` unless
you pass `--port`, and reads `run.json`, `*.loss.csv`, `*.events.jsonl`, samples,
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

- `--family`, `-m`: `zimage` (default) or `klein`
- `--test`, `-t`: `vae`, `encoder`, `transformer`, `pipeline`, or `all`
  (default)
- `--output`, `-o`: artifact directory; defaults to `./validation_output`
- `--save-reference`: save reference latents and outputs for later comparison
- `--compare`: compare against saved reference outputs
- `--reference-dir`: reference directory for `--compare`

### `mere.run text chat`

Run local text chat. The default model is `text-chat-gemma4-12b-4bit` on
Apple Silicon and `text-chat-q36-nano-gguf` on Linux CUDA. Other families
include DiffusionGemma, Laguna 2.1, Inkling-Small, Nemotron, Muse Glimmer,
Qwen3.6, Qwen3.8, Bonsai, Ornith, LFM2.5, and Psi.

```bash
swift run mere.run text chat --prompt "<text>" [options]
```

Key options:

- `--prompt`, `-p`: required user prompt
- `--system`, `-s`: system prompt
- `--model`: canonical model id; defaults to `text-chat-gemma4-12b-4bit`
- `--model-root`, `-m`: explicit local model root. It locates weights and
  skips auto-download; the runtime still comes from `--model` or the default
  model.
- `--image`: for vision checkpoints only: `vision-chat-gemma4-12b`, Muse
  Glimmer, Nemotron 3 Nano Omni, LFM2.5-VL, Bonsai 27B, the Ornith 1.5 35B
  vision ids, and Qwen3.8
- `--audio`, `--video`: local audio or video input for audio- and
  video-capable Omni chat models
- `--seed`: request seed for Qwen-family sampling and DiffusionGemma canvases
- `--reasoning-effort`: 0 through 1 for Qwen3.8 and Muse Glimmer, 0 through
  0.99 for Inkling-Small
- `--max-tokens`: maximum new tokens; defaults to `2048`
- `--context-size`: maximum prompt plus generation context. Qwen3.8 and Bonsai
  27B use their published 262,144-token limit by default. Inkling-Small
  advertises 1,048,576 tokens but uses a 32,768-token operational default
  because KV residency grows with context. Qwen3.8-Flash-Next uses learned QSA
  selection beyond 2,048 tokens in version 0.46.0 and later. Its 262,144-token
  limit is architectural, not a full-context memory qualification for a
  128 GB Mac; to bound retained history, select a smaller window such as
  `32768`.
- `--temperature`: defaults to 0.7, or the model's published value where one
  exists (Bonsai: 0.7; Qwen3.8, Laguna, and Ornith lanes: 1.0)
- `--top-p`: defaults to 0.9, or the model's published value (Laguna: 1.0;
  Qwen3.8, Bonsai, and Ornith: 0.95)
- `--top-k`: defaults to no cutoff, or the model's published value (Qwen3.8,
  Laguna, Bonsai, and Ornith: 20)
- `--min-p`: relative probability floor from 0 through 1; defaults to `0`
  (disabled), or `0.02` for Laguna S 2.1. For example, `0.05` removes tokens
  below 5% of the leading token's probability. It doesn't change greedy
  generation.
- `--kv-bits`: native Qwen-family, Inkling-Small, and LFM2.5 models accept
  affine 4-bit or 8-bit resident KV caches. Gemma4 also supports its
  model-specific cache schemes.
- `--kv-quant-scheme uniform|polar|turboquant`, `--kv-group-size`,
  `--quantized-kv-start`: Gemma4 KV cache quantization backend, group size,
  and the token offset where quantization begins
- `--response-format text|json_object`: require a complete JSON object from a
  native MLX Gemma or Qwen-family model. JSON mode forces thinking off and
  validates each token before streaming.
- `--thinking` / `--no-thinking` (aliases `--show-thinking` /
  `--no-show-thinking`): show reasoning output / disable reasoning
  generation. R1-style lanes (`text-agent-ornith-*`) generate with thinking
  enabled by default even though the reasoning stays hidden. Qwen3.8 and
  Bonsai 27B also default to thinking-enabled generation. LFM2.5 24B-A2B
  always reasons; `--no-thinking` hides its reasoning output.
- `--stream`: write generated text to stdout as tokens arrive
- `--markdown auto|always|never`: present streamed Markdown for a terminal.
  `auto` is the default and only renders interactive text output; `always`
  forces structural rendering, and `never` preserves the model's Markdown.
- `--show-unmasking`: show revision-aware DiffusionGemma canvas drafts on
  stderr
- `--stats`: includes user-visible `ttft_s`, decode-only `first_token_s`,
  separate LFM2 prefill and decode tokens/sec, and Gemma4 MTP state and
  accept/draft counts when those runtimes are used
- `--tools write_file,shell_exec`, `--tool-loop`: enable built-in tools and
  the generate, execute, and continue loop. `--sandbox-dir` sets the tool
  working directory (a temporary directory by default). Shell commands need
  `--allow-shell-exec`, writes outside the sandbox need
  `--allow-absolute-tool-paths`, and `--auto-approve-tools` skips the
  interactive confirmation.
- `--preflight`: inspect the request without loading or downloading a model;
  add `--json` for a structured report
- `--require-installed`: fail instead of downloading a missing model
- `--quiet`, `-q`: suppress progress output

Unless `--quiet` is set, diagnostics on stderr include the selected text
backend, for example native MLX/Metal for MLX models or llama.cpp/GGUF for GGUF
models. `--quiet` does not change the requested output mode: with `--stream`,
generated text still arrives incrementally on stdout while diagnostics remain
suppressed.

Interactive `--stream` output renders headings, lists, quotes, emphasis, and
fenced code as they arrive. Pipes and redirections receive the exact raw
Markdown so scripts remain stable, and `json_object` output is never decorated.
To send raw Markdown to a terminal too, set `--markdown never`. To render
structure when stdout is redirected, set `--markdown always`. Forced
non-terminal rendering does not emit ANSI escapes. The `NO_COLOR` environment
variable disables color while retaining useful typography and structure.

`--lora` accepts a compatible local adapter file or cataloged adapter id.
Native Gemma 4, Laguna XS 2.1, Inkling-Small, and LFM2.5 A1B adapters produced by
`text train-lora` load directly in their matching runtime; `--lora-scale`
scales the adapter.

Before it resolves, downloads, or loads a model, `text chat` checks every option
against the selected model's runtime. Options that runtime rejects, such as
`--image` on a text-only checkpoint, `--lora` on Muse Glimmer, or
`--response-format json_object` outside Gemma 4 and the Qwen family, fail with
one message that names the runtimes that accept them. Options the runtime
accepts but never reads, such as `--top-k` on Gemma 4 or `--tools` on the GGUF
lane, print a `Warning:` line on stderr and the run continues. Companion
drafters (`-mtp`, `-dflash`, `-dspark`, `-assistant`), the LTX text encoder,
MeBot, and DeepSeek V4 Flash are refused with the command they do run.

Examples:

```bash
swift run mere.run text chat --prompt "What is classifier-free guidance?"
swift run mere.run text chat --model text-chat-diffusiongemma-26b-optiq-4bit --max-tokens 256 --prompt "Explain block diffusion."
swift run mere.run text chat --model vision-chat-q38-27b --image ./diagram.png --prompt "Explain this diagram."
swift run mere.run text chat --model vision-chat-q38-flash-next-3bit-native-ple --context-size 32768 --max-tokens 256 --prompt "Explain sparse attention."
swift run mere.run text chat --model text-agent-ornith-35b-mlx-4bit --image ./screenshot.png --prompt "Describe this interface."
swift run mere.run text chat --model vision-chat-q38-flash-next-3bit --context-size 32768 --max-tokens 256 --prompt "Explain sparse attention."
swift run mere.run text chat --model vision-chat-q38-flash-next-mixed --context-size 32768 --max-tokens 256 --prompt "Explain sparse attention."
swift run mere.run text chat --model text-chat-bonsai-27b-1bit --context-size 262144 --kv-bits 4 --prompt "Plan a long-context repository review."
swift run mere.run text chat --model text-chat-bonsai-27b-2bit --context-size 262144 --kv-bits 4 --prompt "Compare two repository migration plans."
swift run mere.run text chat --model text-chat-inkling-small --context-size 32768 --prompt "Plan a recovery-safe repository migration."
swift run mere.run text chat --model text-chat-q36-nano --prompt "Explain speculative decoding."
swift run mere.run text chat --model text-chat-q36-nano --response-format json_object --prompt 'Return an object with a name and an array of tags.'
swift run mere.run text chat --model text-agent-ornith-9b --prompt "Write a compact Swift slugify helper."
swift run mere.run text chat --model text-chat-inkling-small --reasoning-effort 0.2 --prompt "Answer directly."
swift run mere.run text chat --model text-chat-lfm25-2.6b-4bit --prompt "Explain local inference in one paragraph."
swift run mere.run text chat --model text-chat-lfm25-a1b-8bit --prompt "Summarize LFM2 in one paragraph."
swift run mere.run text chat --model vision-chat-lfm25-3b-8bit --image ./photo.jpg --prompt "Describe this image."
swift run mere.run text chat --stream --prompt "Write a short welcome message."
swift run mere.run text chat --stream --markdown never --prompt "Write raw Markdown."
swift run mere.run text chat --thinking --stats --prompt "How would you design a tokenizer?"
```

`json_object` supports nested objects and arrays, Unicode and escaped strings,
numbers, booleans, and null. It is not JSON Schema or strict structured output.
The llama.cpp/GGUF Q36 lane used by Linux packages has no JSON grammar and
rejects `--response-format json_object`. For JSON output, use native MLX Q36
on Apple Silicon.

### `mere.run text train-lora`

Train a LoRA from reviewed chat-style supervised fine-tuning (SFT) JSONL.
Supported families are Gemma 4 text models, `vision-chat-gemma4-12b`,
`text-chat-laguna-xs-2-1`, `text-chat-inkling-small`, and
`text-chat-lfm25-a1b-8bit`.

```bash
swift run mere.run text train-lora \
  --model text-chat-inkling-small \
  --data ./inkling-sft.jsonl \
  --eval ./inkling-heldout.jsonl \
  --output ./inkling-assistant.safetensors \
  --reasoning-effort 0.2 \
  --rank 16 \
  --training-steps 600
```

Each non-empty line contains `sources`, at least system, user, and assistant
messages, and can include typed `tools`; the assistant message must be last.
Assistant `toolCalls` and matching `tool` results use the same typed fields as
inference. Native training templates receive the full schemas, including
parameters and required fields. Dataset identity and duplicate-controller
checks include the complete conversation state and schemas, so successive
tool-loop states can repeat a user question. `--dry-run --json` validates and
fingerprints the dataset and writes the family-specific manifest without
loading the model. Gemma 4 and Laguna default to
`q_proj,k_proj,v_proj,o_proj`. LFM2.5 defaults to
`q_proj,k_proj,v_proj,out_proj` under attention only. Inkling defaults to its
attention projections plus `gate_proj,up_proj,down_proj,lm_head`; expert MLPs
use shared-outer factors to keep the 256-expert adapter tractable. Inkling
`--reasoning-effort` accepts
0 through 0.99, defaults to 0.9, and is recorded in the training manifest; the
other trainers warn that it has no effect. Use the same value for
inference. See the [text runtime guide](/runtime/text) for the full dataset,
training-artifact, and behavioral validation flow.

Key options:

- `--data`, `--output`: required SFT JSONL dataset and output `.safetensors`
  adapter path
- `--model`: base text or Gemma 4 vision model id; defaults to
  `text-chat-gemma4-12b-4bit`. `--model-path` points at an explicit base model
  directory.
- `--eval`: held-out SFT JSONL for before-and-after assistant-token loss
- `--adapter-name`: adapter display name; defaults to `local-assistant`
- `--training-steps`, `--steps`: optimizer steps; defaults to `600`
- `--batch-size`: defaults to `1`
- `--learning-rate`, `--lr`: defaults to `0.0001`
- `--rank`, `--alpha`: LoRA rank (default `16`) and alpha (defaults to the
  rank)
- `--max-sequence-length`: defaults to `4096`
- `--target-modules`: comma-separated LoRA target suffixes
- `--seed`: defaults to `42`
- `--resume-from`: resume from a matching text LoRA checkpoint with Adam
  state. `--resume-step` supplies the completed step for a legacy checkpoint
  without embedded step state.
- `--dry-run`: validate data and write manifests without optimizer steps
- `--visualize`, `--visualize-port`: start the loopback training dashboard,
  on port `8787` by default
- `--json`: print a machine-readable JSON summary

For Gemma 4 vision training, add one dataset-relative `imageUrl` value to the
user message in every example. The command rejects remote URLs, absolute
paths, symbolic links, audio, video, multiple images, and `--batch-size`
values other than `1`. The manifest records image counts, bytes, and a content
fingerprint. The optimizer freezes the vision stack and base language model,
then trains q/k/v/o attention LoRA parameters with assistant-token loss.

Before you train LFM2.5, install the model with
`model pull text-chat-lfm25-a1b-8bit --accept-model-license`. The LFM Open
License v1.0 applies to the base model and derived adapters.

### `mere.run text code`

Run local code generation with GGUF models through the vendored `llama.cpp` runtime.

```bash
swift run mere.run text code --prompt "<text>" [options]
```

Key options:

- `--prompt`, `-p`: required user prompt
- `--system`, `-s`: system prompt; defaults to
  `You are a helpful coding assistant.`
- `--model`, `-m`: GGUF file path, or a managed code model id such as
  `text-code-north-mini`
- `--stream`: stream tokens as they're generated
- `--thinking`: show the model's reasoning before its code response (off by default)
- `--stats`: print generation timing and tokens/sec
- `--temperature`: defaults to `1.0`
- `--top-p`: defaults to `0.95`
- `--min-p`: defaults to `0.0`
- `--max-tokens`: defaults to `2048`
- `--quiet`, `-q`: suppress progress output

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

Ornith 1.5's native Swift/MLX lane has Q4/Q6/Q8 ids plus the BF16 compatibility
id `text-agent-ornith-35b-mlx`. Q4 is a single packaging snapshot containing
the official Q4 target, authoritative vision shard, and MTP head. Use
`model capabilities` to pick
the speed, balanced, or quality tier for this machine. Pull it explicitly;
runtime auto-download is disabled for these large checkpoints:

```bash
swift run mere.run model pull text-agent-ornith-35b-mlx-4bit
swift run mere.run text chat --model text-agent-ornith-35b-mlx-4bit --prompt "Sketch a small Swift Result helper."
```

### `mere.run text embed`

Generate embeddings and print them as JSON. The default model is
`text-embed-qwen3-0.6b`. EmbeddingGemma 2 also embeds images, audio, video, and
mixed inputs, and the PPLX v2 models produce late-interaction (multi-vector)
or contextual embeddings.

```bash
swift run mere.run text embed "semantic search query"
swift run mere.run text embed "foo" "bar" --output embeddings.json --pretty
swift run mere.run text embed --model text-embed-embeddinggemma2 --image ./photo.jpg
```

Key options:

- positional text arguments
- `--model`, `-m`: model id or path; defaults to `text-embed-qwen3-0.6b`
- `--max-tokens`: maximum tokens per input, clamped to the model maximum
- `--task`: `raw` (default), `query`, `document`, `code-retrieval`,
  `question-answering`, `fact-checking`, `classification`, `clustering`, or
  `similarity`. PPLX accepts `query` or `document` and defaults to `document`.
- `--title`: EmbeddingGemma 2 document title; requires `--task document`
- `--dimensions`: output dimensions. EmbeddingGemma 2 accepts 128, 256, 512,
  or 768; PPLX late models 128; PPLX contextual models 1024 or 2048.
- `--image`: independent local images for EmbeddingGemma 2 or PPLX late
  documents
- `--audio`: independent audio segments of up to 30 seconds each
  (EmbeddingGemma 2)
- `--video`: independent videos, sampled at 1 fps up to 32 frames
  (EmbeddingGemma 2)
- `--input-json`: ordered mixed-input JSON document, or `-` for stdin
  (EmbeddingGemma 2)
- `--chunks-json`: PPLX contextual input as `{documents:[[chunk,...],...]}`,
  or `-` for stdin
- `--normalize`: normalize PPLX contextual int8 vectors after dimension
  truncation
- `--output`, `-o`: also write the JSON to a file
- `--pretty`: pretty-print JSON

For model details, see [EmbeddingGemma 2](/runtime/embeddinggemma2) and
[PPLX Embed v2](/runtime/pplx-embed-v2).

### `mere.run text anonymize`

Detect and redact PII with the native OpenAI Privacy Filter model.

```bash
swift run mere.run text anonymize "My name is Dana Example and my email is dana@example.com"
swift run mere.run text anonymize --json --pretty "Phone: 555-1234"
cat notes.txt | swift run mere.run text anonymize --output redacted.txt
```

Key options:

- positional text arguments, or stdin when omitted
- `--model`, `-m`: defaults to `text-anonymize-privacy-filter`
- `--max-tokens`: maximum tokens per input, clamped to the model maximum
- `--replacement`: template supporting `{label}` and `{index}`; defaults to
  `[{label}]`
- `--json`: emit structured JSON with spans
- `--output`, `-o`: also write the result to a file
- `--pretty`: pretty-print JSON

### `mere.run text decide`

Evaluate typed questions with a native Laya, Clef, or D1 decision model. The
command reads a JSON request from `--input` or stdin and prints a JSON result.

```bash
swift run mere.run text decide --input ./decision.json --pretty
cat ./decision.json | swift run mere.run text decide --model text-decide-clef-4bit
```

Key options:

- `--input`, `-i`: JSON request file; omit it or pass `-` to read stdin
- `--model`, `-m`: managed Laya, Clef, or D1 id, or a local checkpoint
  directory; defaults to `text-decide-laya`
- `--output`, `-o`: also save the JSON result to a file
- `--pretty`: pretty-print JSON
- `--preflight`: validate inputs and report token budgets without loading
  weights

For request formats and model choices, see the [Laya](/runtime/laya),
[Clef](/runtime/clef), and [D1](/runtime/d1) runtime pages.

### `mere.run text classify` and `mere.run text extract`

Classify text, or extract entities, relations, and structures, with native
GLiNER2.5 Decide (`text-classify-gliner25-decide`). Pull the managed model
first. Both commands read one JSON request from `--input` or stdin and print a
JSON result. A `text classify` request has `text` and an ordered `tasks`
array; each task has a `name`, `labels`, and optional `prompt`,
`descriptions`, `multi_label`, and `threshold` fields.

```bash
swift run mere.run model pull text-classify-gliner25-decide
swift run mere.run text classify --input ./ticket.json --pretty
swift run mere.run text extract --input ./contract.json --long --pretty
```

Both commands share these options:

- `--input`, `-i`: JSON request file; omit it or pass `-` to read stdin
- `--model`, `-m`: managed GLiNER id or local checkpoint directory
- `--output`, `-o`: also save the JSON result to a file
- `--pretty`: pretty-print JSON
- `--preflight`: validate the request and report token usage without loading
  weights
- `--long`: split long text into overlapping chunks and merge the results
- `--batch`: read a JSON array of requests and return an array of results

For request schemas, see [GLiNER2.5 Decide](/runtime/gliner25-decide).

### `mere.run speech synthesize`

Generate a WAV file from text. The default model is `speech-tts-qwen3-nano`;
`--model` also accepts `speech-tts-qwen3-customvoice`, `speech-tts-breeze-2`,
`speech-tts-phonon-31m`, `speech-tts-drama-3`, or a local model path.

```bash
swift run mere.run speech synthesize "<text>" --output ./speech.wav [options]
```

Key options:

- `--output`, `-o`: required output WAV path
- `--model`, `-m`: TTS model id or local model path
- `--voice`, `-v`: voice description for style mode; defaults to
  `A calm female voice with clear pronunciation`. With `--speaker`, it's an
  optional delivery instruction, and the default description isn't sent.
- `--speaker`: a CustomVoice named speaker in style mode; for
  `speech-tts-qwen3-customvoice`, one of `aiden`, `dylan`, `eric`,
  `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, or `vivian`
- `--mode`: `style` (default) or `clone`
- `--profile`: saved profile id or name for clone mode
- `--ref-audio`, `--ref-text`: reference audio and an optional transcript
  override for clone mode
- `--save-profile`: save the reference as a reusable profile with this name
- `--language`: language hint; defaults to `auto`
- `--temperature`: defaults to `0.6`
- `--seed`: sampling seed for Breeze TTS 2, Phonon, or DRAMA-3
- `--cfg-scale`: Breeze TTS 2 voice guidance scale from 0 through 20; defaults
  to `4`
- `--stream`, `--stream-chunk-tokens`: streaming TTS, emitting a chunk every
  25 tokens by default
- `--quiet`, `-q`: suppress progress output
- `--progress-json`: stage and token-count events on stderr as JSON lines;
  Qwen3-TTS has no known total, so events carry `total_steps: 0`
- `--receipt`: final JSON result line listing the WAV; see
  [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

Examples:

```bash
swift run mere.run speech synthesize "Hello from mere.run" --output ./hello.wav
swift run mere.run speech synthesize "Welcome aboard" --voice "A calm British male voice" --output ./welcome.wav
swift run mere.run speech synthesize "Read this in my cloned voice" --mode clone --profile my-voice --output ./clone.wav
swift run mere.run speech synthesize "Welcome aboard" --model speech-tts-qwen3-customvoice --speaker ryan --output ./ryan.wav
```

### `mere.run speech transcribe`

Transcribe or translate local audio. The transcript goes to stdout unless you
pass `--output`.

```bash
swift run mere.run speech transcribe <audio.wav> [options]
```

Key options:

- positional audio path, or `-` for raw streaming stdin
- `--backend`: `auto` (default), `parakeet`, `qwen`, `whistle`, or
  `nemotron`. `auto` prefers Parakeet for transcription and Qwen3-ASR for
  translation. Nemotron 3.5 Transcribe (`speech-asr-nemotron-35-transcribe`)
  runs only when you select it and doesn't support live ASR, timestamps, or
  speaker timelines.
- `--task`: `transcribe` (default) or `translate` into English (Qwen3-ASR or
  Nemotron; `--language` has no effect with it)
- `--model`, `-m`: model id or local model path for the selected backend
- `--language`: optional language hint, such as `en` or `zh`
- `--max-tokens`: defaults to `448`
- `--provider mlx|coreml`: Parakeet encoder provider; defaults to `mlx`.
  `coreml` requires `--backend parakeet` and a Mere-built artifact directory
  in `--coreml-encoder`.
- `--whistle-weights cactus|fp32`, `--beam-size` (1 through 8, default 5),
  `--decoder-depth` (2 through 8, default 8), and repeatable `--keyword`:
  Whistle decoding controls
- `--stream`: streaming ASR with the selected backend.
  `--stream-chunk-ms` (default 200) sets the feed chunk size and
  `--stream-decode-ms` (default 2000) the decode interval.
- `--input-format pcm-s16le` (raw stdin only)
- `--sample-rate 16000` (raw stdin only)
- `--jsonl`: versioned live ASR events as JSON Lines (raw stdin only)
- `--timestamps` / `--no-timestamps`: include timestamped alignment lines when
  available; on by default
- `--output`, `-o`: transcript file
- `--run-dir`: keep a durable file-transcription run in a new directory
- `--quiet`, `-q`: suppress progress output
- `--receipt`: final JSON result line listing the `--output` transcript file
  (empty `outputs` when the transcript only went to stdout); not available
  with raw stdin, which already speaks `--jsonl`

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

### `mere.run speech diarize`

Identify who spoke when in an audio file with native MLX diarization. The
result is JSON or RTTM on stdout, or in the `--output` file.

```bash
swift run mere.run speech diarize ./meeting.wav
swift run mere.run speech diarize ./meeting.wav --format rttm --output ./meeting.rttm
```

Key options:

- `--model`, `-m`: Sortformer or Nemotron 3 model id, or a local model
  directory; defaults to `speech-diarization-sortformer`
- `--format`, `-f`: `json` (default) or `rttm`
- `--output`, `-o`: optional output file
- `--threshold`: speaker activity threshold from 0 through 1; defaults to
  `0.5`
- `--min-duration`: discard segments shorter than this many seconds; defaults
  to `0.25`
- `--merge-gap`: merge same-speaker segments separated by at most this many
  seconds; defaults to `0.25`
- `--latency`: Nemotron 3 input-buffer latency: `offline` (default), `1.04`,
  `0.64`, or `0.32` seconds
- `--quiet`, `-q`: suppress progress output

### `mere.run speech diarize-live`

Stream Nemotron 3 speaker activity as JSON Lines on stdout from a macOS
microphone or raw PCM on stdin.

```bash
swift run mere.run speech diarize-live --list-devices
swift run mere.run speech diarize-live --device <core-audio-uid> --latency 0.32
producer | swift run mere.run speech diarize-live --stdin
```

Key options:

- `--model`, `-m`: defaults to `speech-diarization-nemotron3`
- `--device`, `--list-devices`: select or list CoreAudio input devices; the
  system input is the default
- `--stdin`: read 16 kHz mono signed 16-bit little-endian PCM from stdin
- `--latency`: `1.04` (default), `0.64`, or `0.32` seconds
- `--threshold`: speaker activity threshold; defaults to `0.5`
- `--quiet`, `-q`: suppress diagnostics; JSON Lines stay on stdout

### `mere.run speech listen`

Capture a macOS input device and transcribe it in real time with Qwen3-ASR or
Whistle.

```bash
swift run mere.run speech listen [options]
```

To print stable CoreAudio UIDs, use `--list-devices`; to select one, pass
`--device <uid>`. The system input is the default. `--model` selects a Qwen or
Whistle model id or local path, and `--language` adds a hint. Whistle accepts
the same `--whistle-weights`, `--beam-size`, `--decoder-depth`, and
`--keyword` options as `speech transcribe`. `--decode-ms` (default 2000) sets
the partial decode cadence, and `--silence-ms` (default 900) the silence that
commits an utterance. `--jsonl` emits the same protocol-v1 event stream as raw
stdin, and `--quiet` suppresses diagnostics and interactive partials. Ctrl-C
flushes the active utterance before exiting.

### `mere.run speech profile`

Manage reusable voice clone profiles for `speech synthesize --mode clone`.

- `speech profile list`: list saved profiles
- `speech profile create --name <name> --audio <file>`: create a profile from
  reference audio. `--text` overrides the reference transcript, `--language`
  sets the language code (default `auto`), and `--quiet` suppresses progress.
- `speech profile delete --id <uuid>`: delete a profile

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

The command writes one `.txt` caption per image, next to the image or in
`--output-dir`. Without `--model`, it downloads
`mlx-community/Qwen3-VL-2B-Instruct-4bit`.

Key options:

- `--model`, `-m`: local model root directory
- `--output-dir`, `-o`: caption directory; defaults to beside each image
- `--prompt`: caption instruction; short instructions work best for LoRA data
- `--prompt-file`: read reusable caption instructions from a UTF-8 text file
- `--focus`: visible details for the captioner to prioritize
- `--trigger-token`: prefix each saved caption with an exact LoRA trigger token
- `--max-tokens`: defaults to `96`
- `--temperature`: defaults to `0.2`
- `--top-p`: defaults to `0.9`

### `mere.run vision inspect`

Describe an image or answer a question about it with Qwen3-VL. The response
goes to stdout, and diagnostics go to stderr.

```bash
swift run mere.run vision inspect ./diagram.png --prompt "What does this diagram show?"
swift run mere.run vision inspect ./diagram.png "What does this diagram show?"
```

Key options:

- `--prompt`: question about the image; defaults to `Describe this image.`
  Trailing positional words also work as the prompt.
- `--model`, `-m`: local model root directory; the default model downloads
  when omitted
- `--max-tokens`: defaults to `2048`
- `--temperature`: defaults to `0.7`
- `--top-p`: defaults to `0.9`

### `mere.run vision embed`

Generate shared text and image embeddings with native Qwen3-VL
(`vision-embed-qwen3-vl-2b`) and print them as JSON. Each `--text` and
`--image` value becomes an independent item. For mixed text-and-image records
or stable caller ids, use `--input-json`.

```bash
swift run mere.run vision embed --text "a white SUV"
swift run mere.run vision embed --image ./vehicle.jpg --dimensions 1024
swift run mere.run vision embed --input-json ./retrieval-inputs.json --pretty
```

The JSON input is an array or `{"inputs":[...]}`. Each record accepts an `id`,
`text`, `instruction`, and `image` or `images`:

```json
{"inputs":[{"id":"query","text":"a white SUV","instruction":"Retrieve similar vehicles"},{"id":"candidate-1","images":["./crop.jpg"]}]}
```

Key options:

- `--text`, `--image`: independent text and local image inputs
- `--input-json`: JSON batch path, or `-` for stdin
- `--instruction`: retrieval task instruction for direct inputs
- `--model`, `-m`: defaults to `vision-embed-qwen3-vl-2b`
- `--dimensions`: 1 through 2048; truncated vectors are renormalized
- `--max-tokens`: per item; defaults to `8192`
- `--min-pixels`, `--max-pixels`: image resize bounds; default `4096` and
  `1843200`
- `--output`, `-o`: also write the JSON to a file
- `--pretty`: pretty-print JSON

### `mere.run vision ground`

Ground text queries in an image with the native Falcon Perception runtime.
Without `--model`, the command uses `vision-ground-falcon-perception` from the
local model store.

```bash
swift run mere.run model pull vision-ground-falcon-perception
swift run mere.run vision ground ./street.jpg --query "cat" "person in red"
```

Key options:

- `--query`, `--prompt`: one or more grounding expressions
- `--model`, `-m`: managed model id or local Falcon Perception model root
- `--output`, `-o`: annotated image; defaults to `<image>_grounded.<ext>`
- `--json-output`: metadata path; defaults to `<image>_grounded.json`
- `--mask-output-dir`: optional per-detection PNG mask directory
- `--preflight`: validate the image, model, queries, and outputs without
  loading the model; add `--json` for a structured report
- `--quiet`, `-q`: print only the annotated image path
- `--receipt`: final JSON result line listing every output; see
  [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

### `mere.run vision serve`

Serve resident Falcon Perception grounding over HTTP. The model stays loaded
across requests. Clients send image bytes and one or more text queries as
`multipart/form-data` to `POST /v1/vision/ground`, or a bounded set of images
to `POST /v1/vision/ground-batch`.

```bash
swift run mere.run vision serve
swift run mere.run vision serve --preflight --json
```

The server binds to loopback by default. A non-loopback `--host` requires
`--api-key` or `MERERUN_API_KEY`.

Key options:

- `--port`, `-p`: defaults to `8091`
- `--host`: defaults to `127.0.0.1`
- `--model`, `-m`: managed model id or local Falcon Perception model root
- `--api-key`: bearer token required by inference endpoints
- `--max-frame-bytes`: maximum encoded image upload; defaults to `16777216`
- `--max-batch-size`: maximum image-query pairs per batch; defaults to `8`
- `--max-batch-bytes`: maximum combined image bytes per batch; defaults to
  `67108864`
- `--preflight`: validate the configuration without starting the server; add
  `--json` for a structured report

### `mere.run vision segment`

Segment prompted objects in an image using the native SAM 3.1 runtime.

```bash
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
```

Key options:

- `--prompt`: one or more text object prompts
- `--box`: one or more `x1,y1,x2,y2[,label]` geometry prompts, one object each
- `--point`: one or more `x,y,positive[,label]` or `x,y,negative[,label]` geometry prompts
- `--model`: managed model id or local SAM 3.1 model root
- `--output`: annotated image path
- `--json-output`: metadata path
- `--mask-output-dir`: optional per-object mask export directory
- `--threshold`: score cutoff from 0 through 1; defaults to `0.05`
- `--resolution`: square SAM 3.1 preprocessing resolution; defaults to `1008`
- `--show-boxes`: draw bounding boxes over the mask overlays
- `--multimask`: emit up to three candidates per geometry-prompted object
- `--preflight`: inspect the request without loading SAM or processing the
  image
- `--json`: only with `--preflight`
- `--quiet`, `-q`: suppress progress output
- `--receipt`: final JSON result line listing the annotated image, the
  detections JSON, and the mask directory; see
  [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

Defaults:

- annotated image: `<image-stem>_segmented.<ext>`
- JSON metadata: `<image-stem>_segmented.json`

Notes:

- still-image runs accept text, box, and point prompts in the same invocation
- boxes and points combine by label: a labeled `--point` refines the `--box`
  with the same label that contains it (else the first box with that label),
  or forms one object with the other points of that label; unlabeled points
  form one object together, or refine the one unlabeled `--box` when there is
  exactly one. Points that match no box must include a positive point; a
  negative point on its own is rejected because it describes no object
- `--mask-output-dir` writes one PNG mask per exported detection candidate
- empty detection sets still produce annotated output plus JSON metadata

Examples:

```bash
swift run mere.run vision segment ./photo.jpg --prompt "a cat"
swift run mere.run vision segment ./photo.jpg --prompt "a person" "a phone" --show-boxes
swift run mere.run vision segment ./photo.jpg --box "120,80,420,760,person" --mask-output-dir ./masks
swift run mere.run vision segment ./photo.jpg --point "512,384,positive,person" --point "700,200,negative,person"
swift run mere.run vision segment ./photo.jpg --box "120,80,420,760" --point "260,400,positive" --point "400,700,negative"
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
- `--box`: one or more `x1,y1,x2,y2[,label]` geometry prompts, one object each
- `--point`: one or more `x,y,positive[,label]` or `x,y,negative[,label]`
  geometry prompts, grouped with boxes by label the way `vision segment` groups
  them
- `--model`, `-m`: managed model id or local SAM 3.1 model root
- `--init-frame`: starting frame index for seeding; defaults to `0`
- `--end-frame`: optional inclusive final frame index
- `--output`: annotated video path
- `--json-output`: tracking metadata path
- `--mask-output-dir`: optional per-frame mask export directory
- `--threshold`: score cutoff; defaults to `0.05`
- `--resolution`: square SAM 3.1 preprocessing resolution; defaults to `1008`
- `--preflight`: inspect the request without loading SAM or processing video
- `--json`: only with `--preflight`
- `--show-boxes`: draw bounding boxes over tracked masks
- `--show-labels`: reserved for labeled video overlays
- `--quiet`, `-q`: suppress progress output
- `--receipt`: final JSON result line listing the annotated video, the
  tracking JSON, and the mask directory; see
  [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

Defaults:

- annotated video: `<video-stem>_tracked.mp4`
- JSON metadata: `<video-stem>_tracked.json`

Notes:

- text prompts seed objects on `--init-frame`, then the native tracker reuses geometry prompts for later frames
- box and point prompts seed explicit tracked objects directly on the init frame,
  grouped by label the way `vision segment` groups them; points act only on the
  init frame and when an object the propagated box lost is re-seeded from its
  seed geometry. They never act on the box propagated from the previous
  frame, because they were placed on the init frame and the object has moved.
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
- `--model`, `-m`: managed model id or local SAM 3.1 model root
- `--camera`: camera device index; defaults to `0`
- `--duration-seconds`: capture duration; defaults to `10`
- `--init-frame`: initial frame index used to seed tracking; defaults to `0`
- `--seed-search-frames`: additional frames to search when the init frame
  finds no objects; defaults to `30`
- `--output`: required annotated video path
- `--json-output`: optional tracking metadata path
- `--threshold`: score cutoff; defaults to `0.05`
- `--resolution`: square SAM 3.1 preprocessing resolution; defaults to `1008`
- `--show-boxes`: draw bounding boxes over tracked masks
- `--show-labels`: reserved for labeled video overlays

Notes:

- `track-live` records a camera clip first, then runs tracking over the recorded media
- live tracking searches a short warm-up window after the init frame so startup exposure or motion blur doesn't silently produce an unsegmented output
- live mode accepts only text prompts

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

The subcommands are:

- `vision face detect <image>`: detect faces and five-point landmarks.
  `--max-faces` caps detections after NMS, and `--include-embeddings` adds
  normalized 512-dimensional ArcFace embeddings.
- `vision face embed <image>`: embed one face. `--face-index` picks a
  zero-based face; the default is the largest face.
- `vision face compare <reference> <candidate>`: compare one face from each
  image with cosine similarity. `--reference-face-index` and
  `--candidate-face-index` default to the largest face.
- `vision face batch [<images> ...]`: analyze many images with one warm
  detector and recognizer session and write one success-or-error JSONL record
  per image. Use it to index large libraries. `--input-list` reads one path per
  line, `--jsonl-output` writes to a file instead of stdout, and `--fail-fast`
  stops at the first unreadable image.

All four accept `--model`, `--score-threshold` (default `0.65`), and
`--execution-provider auto|coreml|cpu` (default `auto`). `detect`, `embed`, and
`compare` print JSON on stdout with `--json` or write it with `--json-output`.

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
- `--max-hands`: maximum hand observations; defaults to `2`
- `--minimum-confidence`: filter landmarks below this confidence; defaults to
  `0.1`
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
maximum magnitude. `--accuracy` accepts `low`, `medium`, `high` (default), or
`very-high`. `--json` also prints the metadata on stdout.

### `mere.run vision depth`

Estimate affine-invariant depth for a still image with native Marigold V2
(`vision-depth-marigold-v2`).

```bash
swift run mere.run vision depth ./room.jpg --output ./room-depth --json
```

Key options:

- `--output`, `-o`: output directory
- `--model`: managed model id or model root
- `--max-edge`: longest inference edge, rounded to a multiple of 16; defaults
  to `1024`. `--native` runs at the source resolution instead.
- `--checkpoint`: `log-stage2`, `log-stage1`, `log-layered`, `uniform-base`,
  `uniform-layered`, `disparity-base`, or `disparity-layered`
- `--dry-run`: validate and print the execution plan without loading weights
- `--json`: print the structured result on stdout
- `--receipt`: final JSON result line listing every output

### `mere.run vision depth-video`

Generate temporally consistent relative or metric video depth with native
Video Depth Anything Small. Use `vision-depth-vda-small` for relative depth or
`vision-depth-vda-small-metric` for metric depth.

```bash
swift run mere.run vision depth-video ./walkthrough.mp4 \
  --model vision-depth-vda-small-metric \
  --output ./walkthrough-depth
```

Key options:

- `--output`, `-o`: output directory
- `--model`: relative or metric model id, pinned `.pth` path, or converted
  `model.safetensors` directory
- `--input-size`: longest network input edge from 14 through 1008, before
  aspect-ratio adjustment; defaults to `518`
- `--max-frames`: decode at most this many source frames, from 1 through 2400;
  defaults to `240`
- `--dry-run`: snapshot, hash, and decode the bounded input, verify limits and
  the checkpoint, and print the plan without inference
- `--json`: print the structured result on stdout

### `mere.run vision geometry`

Generate metric depth, normals, camera intrinsics, and a point cloud from one
image with native MoGe-2 (`vision-geometry-moge2-small`).

```bash
swift run mere.run vision geometry ./desk.jpg --output ./desk-geometry
```

Key options:

- `--output`, `-o`: output directory
- `--model`: managed model id, model root, or pinned `model.onnx` path
- `--resolution-level`: quality level from 0 through 9; defaults to `9`
- `--token-count`: override the DINO base-token count, from 1 through 3600
- `--max-points`: maximum points written to the PLY file
- `--dry-run`: validate and print the execution plan without loading weights
- `--json`: print the structured result on stdout

### `mere.run vision geometry-multiview`

Solve relative geometry, confidence, and cameras across several views with
native DA3-Small (`vision-geometry-da3-small`). Pass images in view order.

```bash
swift run mere.run vision geometry-multiview ./views/01.jpg ./views/02.jpg ./views/03.jpg \
  --output ./scene
```

Key options:

- `--output`, `-o`: output scene directory
- `--model`: managed model id or exact pinned DA3-Small directory or
  safetensors path
- `--cameras`: optional JSON document with one calibrated world-to-camera
  (W2C) camera per image
- `--process-resolution`: upper bound for the longest processed side;
  defaults to `504`
- `--reference-view`: `first`, `middle`, `saddle-balanced` (default), or
  `saddle-similarity-range`
- `--confidence-percentile`: discard points below this percentile; defaults
  to `40`
- `--max-points`: maximum colored points in scene exports; defaults to
  `1000000`
- `--dry-run`: verify the model and inputs and print the plan without
  inference
- `--json`: print structured JSON on stdout

### `mere.run vision image-to-3d` and `mere.run image reconstruct-3d`

Reconstruct an object mesh from images. Each command exists under both
`vision` and `image` with the same options:

| `vision` command | `image` command | Runtime and model |
| --- | --- | --- |
| `vision image-to-3d` | `image reconstruct-3d` | TripoSR colored mesh from one image (`image-3d-triposr`) |
| `vision image-to-3d-trellis2` | `image reconstruct-3d-trellis2` | MLX TRELLIS.2 512-resolution PBR O-Voxel mesh from one image (`image-3d-trellis2-4b`) |
| `vision image-to-3d-multiview` | `image reconstruct-3d-multiview` | InstantMesh colored mesh from four or six views (`image-3d-instantmesh-base`) |

```bash
swift run mere.run vision image-to-3d ./mug.png --output ./mug-mesh
swift run mere.run vision image-to-3d-trellis2 ./mug.png --output ./mug-pbr --seed 7
swift run mere.run image reconstruct-3d-multiview \
  --view ./views/front.png --view ./views/right.png \
  --view ./views/back.png --view ./views/left.png \
  --output ./mug-multiview
```

All three accept `--output`, `--model`, `--dry-run` (verify and print the plan
without loading weights), and `--json`. Other options:

- TripoSR: `--resolution` (density grid from 2 through 512, default `256`),
  `--density-threshold` (default `25.0`), `--foreground-ratio` (default
  `0.85`), `--already-framed` to skip the transparent-foreground crop, and
  `--no-vertex-colors` for geometry only. Transparent PNG foregrounds are
  cropped automatically.
- TRELLIS.2: the input needs transparent alpha unless you pass
  `--already-framed`. `--seed` (default `42`) fixes the run, and
  `--texture-seed` re-rolls only the texture while keeping the geometry.
  `--max-tokens` caps decoded O-Voxels (default `2097152`). `--no-remesh`
  exports the raw dual-grid crust instead of the watertight remesh,
  `--remesh-band` sets the remesh half-width in voxels (default `1.0`), and
  `--seal-radius` seals occluded cavities (default `12`; `0` disables it).
- InstantMesh: repeat `--view` four or six times in order. `--cameras` reads an
  optional camera JSON file, `--resolution` sets the SDF grid from 2 through
  256 (default `128`), and `--no-vertex-colors` skips vertex colors.

### `mere.run vision ocr`

Extract text from one or more images. Text goes to stdout unless you pass
`--output-dir`.

```bash
swift run mere.run vision ocr <images...> [options]
```

Key options:

- `--backend`, `-b`: `lighton` (default), `glm`, or `infinity`
- `--compare`: compare LightOn against the selected secondary backend; defaults
  to GLM when `--backend lighton`
- `--model`, `-m`: managed id or path to the LightOn OCR root when using the
  LightOn backend. The default `vision-ocr-lighton` downloads LightOnOCR 3-1B;
  `vision-ocr-lighton-3-0.8b` and `vision-ocr-lighton-3-4b` are native Qwen3.5
  variants.
- `--lighton-mode`: `plain` (default) for Markdown transcription or `grounding`
  for labeled boxes in normalized 0–1000 coordinates, descriptions, and chart data
- `--glmocr-cli`, `--glm-config`: path to the `glmocr` executable (found on
  `PATH` by default) and an optional GLM-OCR config YAML
- `--infinity-runtime`: `native` (default) or `external`; native uses the
  Swift Q35 runtime
- `--infinity-model`: native managed model id or local path, defaulting to
  `vision-ocr-infinity-pro-int8`; upstream model or server id when
  `--infinity-runtime external`
- `--infinity-parser-cli`: path to the Infinity-Parser2 `parser` executable for
  external runs
- `--infinity-backend`: external `vllm-server` (default), `vllm-engine`, or
  `transformers`
- `--infinity-api-url`, `--infinity-api-key`: external vLLM server settings;
  defaults to `http://localhost:8000/v1/chat/completions` and `EMPTY`
- `--infinity-task`: `doc2json` (default), `doc2md`, or `custom`. A `custom`
  task requires `--infinity-prompt`.
- `--infinity-output-format`: `md` (default) or `json`
- `--infinity-min-pixels`, `--infinity-max-pixels`: image pixel bounds;
  default `2048` and `16777216`
- `--infinity-batch-size`, `--infinity-model-cache-dir`: external parser
  batch size (default `1`) and model cache directory
- `--output-dir`, `-o`: write one `.txt` file per image
- `--max-tokens`: defaults to `4096`
- `--temperature`: defaults to `0.2`
- `--quiet`, `-q`: print only the OCR text, without file paths

Examples:

```bash
swift run mere.run model pull vision-ocr-lighton
swift run mere.run model pull vision-ocr-infinity-pro-int8
swift run mere.run vision ocr ./page.png --backend lighton --model ~/Library/Application\ Support/MereRun/models/vision-ocr-lighton
swift run mere.run vision ocr ./page.png --backend glm
swift run mere.run vision ocr ./page.png --backend infinity --infinity-task doc2md
swift run mere.run vision ocr ./page.png --compare --backend infinity
swift run mere.run vision ocr ./page.png --backend infinity --infinity-runtime external --infinity-api-url http://127.0.0.1:8000/v1/chat/completions
```

### `mere.run geo`

Run native geospatial models on local Earth-observation tensors. Each command
reads a safetensors input and writes a safetensors output.

| Command | Model ids | Input | Output |
| --- | --- | --- | --- |
| `geo flood` | `vision-flood-terramind-base` | Normalized S2L2A, S1RTC, and DEM tile batches | Flood logits |
| `geo fire` | `vision-fire-terramind-base` | Normalized S2L2A, S1RTC, and DEM tile batches | Fire logits |
| `geo tessera` | `vision-embed-tessera-v2-*` | Raw Sentinel-2 and Sentinel-1 observations with day-of-year tensors | Time-series embeddings |
| `geo olmoearth` | `vision-embed-olmoearth-v12-*` | `TIMESTAMPS` and one or more raw imagery tensors | Spatial embeddings |

```bash
swift run mere.run geo flood ./tiles.safetensors --output ./flood-logits.safetensors
swift run mere.run geo tessera ./series.safetensors \
  --model vision-embed-tessera-v2-small \
  --output ./embeddings.safetensors \
  --dimensions 64
```

All four require `--output` and accept `--model` (a managed id or converted
model root), `--preflight` (validate inputs and model availability without
loading weights), and `--json` (print execution metadata on stdout). Other
options:

- `geo tessera --dimensions`: 16, 32, 64, or 128 for students; 1024 for the
  teacher
- `geo olmoearth --patch-size`: 1, 2, 4 (default), or 8 pixels
- `geo olmoearth --input-resolution`: input ground sample distance in meters;
  defaults to `10.0`
- `geo olmoearth --include-tokens`: also write full space-time tokens

For input layouts, see the [geospatial runtime guide](/runtime/geo).

### `mere.run audio generate`

Generate audio from text with the LTX-2.5 audio-only pipeline, without
building or denoising video. It requires the full, gated LTX-2.5 checkpoint
(`video-ltx25-full-bf16`).

```bash
swift run mere.run audio generate "A quiet forest at dawn with distant birds"
swift run mere.run audio generate "Ocean surf" --duration 8 --output ./surf.wav
```

Key options:

- `--output`, `-o`: output WAV; defaults to `ltx25-audio.wav`
- `--model`, `-m`: managed LTX-2.5 full model id or local path;
  `--model-root` sets an explicit checkpoint root
- `--negative-prompt`: negative audio prompt; defaults to the upstream LTX
  negative prompt
- `--enhance-prompt`: expand the request with the native Gemma-4 LTX-2.5
  caption enhancer. `--prompt-enhancer-model` and
  `--prompt-enhancer-model-root` select the enhancer.
- `--duration`: seconds, snapped to the LTX frame grid. `--num-frames` sets
  the duration as 8n+1 frames, and `--auto-duration <min> <max>` predicts it
  with DurationHead. With none of these, the command predicts a duration
  within 1 to 20 seconds.
- `--fps`: video-clock frame rate that defines the audio duration; defaults to
  `24`
- `--steps`: defaults to `30`
- `--seed`: defaults to `10`
- `--audio-cfg-guidance-scale` (default `7.0`), `--audio-stg-guidance-scale`
  (default `1.0`), `--audio-rescale` (default `0.7`), repeatable
  `--audio-stg-block` (default `28`), and `--audio-skip-step` (default `0`):
  guidance controls
- `--sigmas`: custom descending sigma schedule ending in 0
- `--lora PATH[=STRENGTH]`: LTX LoRA; repeat to stack adapters
- `--quiet`: suppress progress diagnostics

### `mere.run audio enhance`

Extend speech or general-audio bandwidth to a 48 kHz mono float WAV with native
Swift/MLX. The default, `audio-enhance-ap-bwe-16kto48k`, is deterministic
16 kHz speech bandwidth extension. `audio-enhance-universr-audio` is
flow-matching super-resolution for speech, music, or sound effects from an
effective 8, 12, 16, or 24 kHz bandwidth. The command saves a provenance
manifest beside the output and prints it as JSON on stdout.

```bash
swift run mere.run model pull audio-enhance-ap-bwe-16kto48k
swift run mere.run audio enhance ./speech.mp3 --output ./speech-wideband.wav
swift run mere.run audio enhance ./music-12k.wav --model audio-enhance-universr-audio
```

Key options:

- `--model`, `-m`: managed enhancement model id; `--model-path` sets an
  explicit local snapshot root
- `--output`, `-o`: output WAV path
- `--overlap`: chunk overlap count; defaults to the published profile value
  (2)
- `--dtype float16|float32`: defaults to `float32`
- `--quiet`, `-q`: suppress progress on stderr

UniverSR also accepts `--input-rate` (8000, 12000, 16000, or 24000 Hz),
`--ode-method euler|midpoint|rk4` (default `midpoint`), `--ode-steps`
(default `4`), `--guidance-scale` (default `1.5`), `--seed` (default `42`),
and `--chunk-seconds` (default `10`, minimum 3).

### `mere.run audio edit`

Generate or edit speech from a natural-language instruction with native AuK.
AuK needs a base or Flash checkpoint and the separate Qwen2.5-Omni-3B encoder.
To edit or condition on a voice, pass `--audio`; text-only generation requires
`--duration`. This FP32 runtime is experimental; see
`mere.run guide audio edit` for qualification results.

```bash
swift run mere.run audio edit "Say 'welcome back' in a warm, slow voice" --duration 3
swift run mere.run audio edit "Make the speaker sound farther away" --audio ./line.wav --output ./line-far.wav
```

Key options:

- `--audio`: reference audio for editing or voice conditioning
- `--model`: `audio-auk-base` (default) or `audio-auk-flash`. Flash always
  uses its fixed four-step schedule with guidance disabled.
- `--model-path`, `--thinker-path`: local AuK and Qwen2.5-Omni-3B checkpoint
  directories
- `--output`, `-o`: defaults to `auk.wav`
- `--duration`: output seconds; otherwise follows the reference length
- `--steps`: base model Euler steps from 1 through 1000; defaults to `32`
- `--guidance`: base model classifier-free guidance; defaults to `2.0`
- `--seed`: defaults to `42`
- `--quiet`, `-q`: suppress progress diagnostics

For model details, see the [audio runtime guide](/runtime/audio).

### `mere.run music analyze`

Analyze an audio file with ACE-Step 5 Hz LM audio understanding and print JSON
metadata to stdout.

```bash
swift run mere.run music analyze "<audio>" [options]
```

Key options:

- `--model`, `-m`: managed ACE-Step model id or local model or checkpoints
  root; defaults to `music-acestep`
- `--checkpoints-root`: ACE-Step checkpoint root; discovered automatically when
  omitted
- `--decoder-subdirectory` (alias `--turbo-subdirectory`, default
  `acestep-v15-turbo`), `--vae-subdirectory` (default `vae`), and the legacy
  `--lm-subdirectory`: component directories under the checkpoint root
- `--lm-model`: managed planner id or local planner root; defaults to
  `music-acestep-lm-1.7b` when the selected checkpoint has no LM
- `--duration`: analyze the first N seconds instead of the full decoded input
- `--max-new-tokens`: defaults to `2048`
- `--lm-temperature`: defaults to `0.3`
- `--lm-top-k`: defaults to `0` (disabled)
- `--lm-top-p`: defaults to `0.9`
- `--include-raw-lm`: include the raw LM analysis text in the JSON
- `--include-audio-codes`: include serialized ACE audio-code tokens in the JSON
- `--quiet`, `-q`: suppress stderr diagnostics

Examples:

```bash
swift run mere.run music analyze ./song.mp3 \
  --model music-acestep-xl-sft \
  --lm-model music-acestep-lm-1.7b
swift run mere.run music analyze ./song.mp3 --duration 30 > ./song-analysis.json
```

### `mere.run music generate`

Generate music from a caption and print the output WAV path to stdout.
Progress and diagnostics go to stderr. The default model, `music-acestep`, uses
the native ACE-Step pipeline with optional lyrics. `--model` also selects
MiniMax Music 3 (`music-minimax-music3`), YuE2 (`music-yue2`), or Magenta RT2
(`music-magenta-rt2-small`, `music-magenta-rt2-base`). Each family fails before
loading on another family's controls, unless the control only carries its own
default; then the command prints a warning.

```bash
swift run mere.run music generate "<caption>" [options]
```

Key options:

- `--model`, `-m`: managed music model id or local model root
- `--output`, `-o`: output WAV path; defaults to
  `./mererun-music-<timestamp>.wav`
- `--checkpoints-root`: ACE-Step checkpoint root; discovered automatically when
  omitted
- `--lyrics`, `--lyrics-file`: lyrics text or a lyrics file; use one or the
  other
- `--instrumental`: generate without vocals using the selected model's lyric
  convention (upstream's `[Instrumental]` marker for ACE-Step); can't be
  combined with lyrics
- `--source-audio`: source song for ACE-Step cover conditioning; implies cover mode unless `--non-cover` is set
- `--analyze-source-audio`: use ACE-Step 5 Hz LM audio understanding to fill missing cover metadata from `--source-audio`
- `--reference-audio`: optional ACE-Step timbre reference audio file(s)
- `--duration`: output upper bound in seconds (YuE2: 360; MiniMax: 60;
  ACE-Step `song` and `final` quality can plan it)
- `--steps`, `-s`: denoising steps (YuE2: 32 midpoint steps; MiniMax: 30;
  ACE-Step Turbo: 8; Base and SFT: 50)
- `--quality`: adaptive ACE-Step quality: `draft`, `song`, `final`, or `edit`
- `--use-lm`, `--no-lm`: turn ACE-Step 5 Hz LM planning and semantic-code
  generation on, or off when a quality preset selects it
- `--lm-model`: defaults to `music-acestep-lm-1.7b` when the selected
  checkpoint has no LM
- `--lm-subdirectory`: legacy same-root component override
- `--lm-temperature`: planner sampling temperature from `0` to `2` (default `0.85`)
- `--lm-top-k`, `--lm-top-p`: planner candidate and nucleus sampling
- `--lm-repetition-penalty`: planner and semantic-code repetition penalty;
  `1.0` (default) disables it
- `--lm-cfg-scale`: classifier-free guidance applied only during semantic-code generation (upstream default `2.0`)
- `--lm-negative-prompt`: unconditional LM prompt used by semantic-code guidance (default `NO USER INPUT`)
- `--no-lm-caption-rewrite`: preserve the input caption while retaining LM metadata planning and semantic audio codes (upstream `use_cot_caption=false`)
- `--text-subdirectory`: ACE-Step text encoder directory; detected as
  `Qwen3-Embedding-0.6B` when omitted
- `--seed`: seed for deterministic generation
- `--quiet`, `-q`: suppress stderr diagnostics
- `--progress-json`: JSON progress lines on stderr for the MiniMax Music 3
  (`semantic`, `denoising`, `decoding`) and Magenta RT2 (`generating`) lanes;
  the ACE-Step pipeline exposes no per-step callback and emits none
- `--receipt`: final JSON result line listing the WAV first, then any
  candidates, stems, LRC, recipe, composition, profile, or DAW bundle it wrote;
  see [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

Magenta RT2 options:

- `--style-conditioning`: `streaming` (default) keeps the realtime C++ coarse
  style-token policy; `full` uses all MusicCoCa style tokens like the Python
  high-level generator
- `--temperature`: defaults to `1.0`
- `--top-k`: defaults to `100`
- `--cfg-musiccoca`, `--cfg-notes`, `--cfg-drums`: guidance scales; default
  `3.0`, `5.0`, and `1.0`
- `--drumless`: drumless mode
- `--unmask-width`, `--seed-rotation`: token unmask width and seed rotation;
  both default to `0`
- `--prefill-silence`, `--prefill-duration`: prefill state with silence for
  this many seconds (default `1.64`)

The command has further option groups. Run `mere.run music generate --help`
for the full list, and `mere.run guide music generate --model <id>` for a
model's contract:

- Output: `--export-format pcm16|pcm24|float32` (default `pcm24`),
  `--normalize none|peak` (default `peak`), `--target-peak-db`,
  `--fade-in-ms`, `--fade-out-ms`, `--no-dither`, `--recipe-output`,
  `--no-recipe`, `--daw-bundle`, and `--stems` (Base checkpoints only)
- Candidates: `--candidates` (alias `--best-of`) and `--keep-candidates`
- ACE-Step sampling and editing: `--shift`, `--sampler euler|heun`,
  `--guidance-scale`, `--task-type`, `--instruction`, the `--repaint-*` and
  `--flow-edit*` families, `--retake-seed`, `--retake-variance`,
  `--audio-cover-strength`, `--bpm`, `--keyscale`, `--timesignature`, and the
  `--vae-*` tiling controls
- ACE-Step adapters: `--adapter`, `--adapter-kind auto|lora|lokr`, and
  `--adapter-scale`
- MiniMax Music 3: `--compose` with `--composer-model` (default
  `text-chat-gemma4-12b-4bit`), `--lyrics-preflight off|warn|strict`,
  `--lrc-file`, `--lrc-output`, `--sample-rate`, `--memory-mode`,
  `--performance-mode`, `--sampling-tier`, `--flow-strategy`, and
  `--flow-solver`
- YuE2: `--score-mode full|melody|off`, `--abc-file`, `--abc-output`, and the
  `--semantic-*` sampling controls

To point ACE-Step at an existing checkpoint root, set
`MERERUN_MUSIC_ACESTEP_ROOT`.

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
  "indie pop with short, clearly separated vocal phrases" \
  --use-lm \
  --lm-temperature 0.7 \
  --lm-repetition-penalty 1.08 \
  --output ./controlled-phrasing.wav
swift run mere.run music generate \
  "cinematic electronic trailer score with rising builds and drops" \
  --use-lm \
  --instrumental \
  --duration 85 \
  --output ./instrumental-trailer.wav
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
- `--model-path`: explicit local model root. `--variant small|medium|large`
  names its architecture when the checkpoint doesn't.
- `--output`, `-o`: output path, or `-` for stdout; defaults to beside the
  input
- `--format`, `-f`: `midi` (default), `json`, or `jsonl`
- `--instruments`: comma-separated expected instrument groups
- `--list-instruments`
- `--sampling` and `--temperature`
- `--max-tokens-per-chunk`
- `--strict-eos`
- `--beam-size`
- `--chunk-batch-size`: upper bound on independent five-second chunks decoded
  together for greedy, sampling, or beam mode (default `4`). The runtime can
  reduce it for available unified-memory headroom and model/beam saturation;
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

- `--model`, `-m`: `music-magenta-rt2-small` (default),
  `music-magenta-rt2-base`, or a local Magenta RT2 root
- `--duration`: run duration in seconds; defaults to `30`
- `--output`, `-o`: optional WAV capture path
- `--play`, `--no-play`: play through the default output device; on by default
- `--style-conditioning`, `--temperature`, `--top-k`, `--cfg-musiccoca`,
  `--cfg-notes`, `--cfg-drums`, `--drumless`, `--unmask-width`,
  `--seed-rotation`, `--prefill-silence`, and `--prefill-duration`: the same
  Magenta RT2 controls and defaults as `music generate`
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

### `mere.run music separate`

Separate or restore audio with pinned, MIT-licensed RoFormer checkpoints
running locally in Swift/MLX. Output WAVs are 44.1 kHz stereo float. The
command saves a provenance manifest and also prints it on stdout.

| Model | Output |
| --- | --- |
| `music-separate-bs-roformer-viperx-1297` (default) | Vocals and instrumental |
| `music-separate-bs-roformer-4stem` | Drums, bass, other, and vocals |
| `music-separate-mel-roformer-dereverb` | Dereverberated audio |
| `music-separate-mel-roformer-denoise` | Denoised audio |

```bash
swift run mere.run model pull music-separate-bs-roformer-viperx-1297
swift run mere.run music separate ./song.mp3
swift run mere.run music separate ./song.wav \
  --model music-separate-bs-roformer-4stem \
  --output-dir ./song-stems
```

Key options:

- `--model`, `-m`: managed RoFormer model id
- `--model-path`: explicit local root of the selected pinned snapshot
- `--output-dir`, `-o`: stem directory; defaults to beside the input
- `--overlap`: chunk overlap count; defaults to the model's published value
- `--dtype float16|float32`: compute type; defaults to `float16`
- `--quiet`, `-q`: suppress progress on stderr

### `mere.run music serve`

Start a local HTTP generation API for ACE-Step or MiniMax Music 3.

```bash
swift run mere.run music serve
swift run mere.run music serve --model music-minimax-music3 --memory-mode resident
```

Key options:

- `--host`: defaults to `127.0.0.1`
- `--port`, `-p`: defaults to `8081`
- `--model`, `-m`: managed ACE-Step or MiniMax model id, or local checkpoint
  root; defaults to `music-acestep`
- `--api-key`: bearer token; also read from `MERERUN_API_KEY`
- `--memory-mode resident|staged`: MiniMax Music 3 loading. `resident` keeps
  all weights warm; `staged` lowers peak residency by reloading stages per
  request.
- `--performance-mode`: MiniMax Music 3 execution: `reference`, `optimized`
  (default), `q8`, `q4`, `q8-lm`, or `q4-lm`
- `--checkpoints-root`, `--decoder-subdirectory`, `--vae-subdirectory`,
  `--lm-subdirectory`, `--text-subdirectory`, and `--lm-model`: the same
  ACE-Step component options as `music generate`
- `--adapter`, `--adapter-kind auto|lora|lokr`, `--adapter-scale`: keep
  ACE-Step adapters resident

### `mere.run music train-adapter`

Train a native ACE-Step LoRA or LoKr adapter from a JSON or JSONL dataset
manifest. Use the result with `music generate --adapter`.

```bash
swift run mere.run music train-adapter \
  --dataset ./style-songs.jsonl \
  --output ./style-lora.safetensors \
  --steps 1000
```

Key options:

- `--dataset`, `--output`: required manifest and output `.safetensors` path
- `--model`, `-m`: managed ACE-Step model id or local checkpoint root;
  defaults to `music-acestep`
- `--kind lora|lokr`: defaults to `lora`
- `--rank`: defaults to `8`
- `--alpha`: defaults to `16.0`
- `--factor`: LoKr factorization target; `-1` (default) picks the closest
  balanced factors
- `--steps`: defaults to `1000`
- `--learning-rate`, `--weight-decay`: AdamW settings; both default to
  `0.0001`
- `--seed`: defaults to `42`
- `--max-duration`: crop each example to this many seconds; defaults to `30`
- `--checkpoints-root`, `--decoder-subdirectory`, `--vae-subdirectory`,
  `--text-subdirectory`: ACE-Step component locations
- `--log-every`: progress interval in steps; defaults to `10`

### `mere.run sfx generate`

Generate a mono WAV sound effect from a text prompt. The default model uses the
native Sony Research Woosh DFlow path; the original Woosh Flow checkpoint is
also available as `sfx-woosh-flow`. `sfx-mmaudio-large-44k-v2` selects the
native 44.1 kHz MMAudio large-v2 runtime.

```bash
swift run mere.run sfx generate "<prompt>" [options]
```

Key options:

- `--model`, `-m`: `sfx-woosh-dflow` (default), `sfx-woosh-flow`,
  `sfx-mmaudio-large-44k-v2`, or a matching local model root
- `--negative-prompt`: negative text conditioning for MMAudio
- `--output`, `-o`: output WAV; defaults to `./mererun-sfx-<timestamp>.wav`
- `--duration`: seconds; defaults to 5 for Woosh and 8 for MMAudio
- `--steps`, `-s`: defaults to 4 for Woosh DFlow and 25 for MMAudio
- `--cfg`: classifier-free guidance scale; defaults to `4.5`
- `--renoise`: renoise amount, or a comma-separated DFlow schedule in [0, 1]
- `--seed`: seed for deterministic generation
- `--quiet`, `-q`: suppress stderr diagnostics
- `--progress-json`: one `denoising` JSON progress line per step on stderr
- `--receipt`: final JSON result line listing the WAV; see
  [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

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

Decode input must have shape `[1, 128, frames]`. Both subcommands accept
`--output` (a timestamped file in the current directory by default), `--model`
(defaults to `sfx-woosh-dflow`), and `--quiet`.

### `mere.run sfx condition text`

Export Woosh text-conditioning tensors for a prompt. The output safetensors file
contains `embeddings` and `mask` arrays.

```bash
swift run mere.run sfx condition text "glass breaking" -o ./glass-condition.safetensors
```

The command accepts `--output`, `--model` (defaults to `sfx-woosh-dflow`), and
`--quiet`.

### `mere.run sfx clap score`

Score a text prompt against an audio file with the native Woosh-CLAP text and
PaSST audio towers. The command prints JSON to stdout.

```bash
swift run mere.run sfx clap score "glass breaking" ./glass.wav
```

The command accepts `--model` (defaults to `sfx-woosh-clap`) and `--quiet`.

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

Key options:

- `--model`, `-m`: Woosh VFlow or DVFlow id, MMAudio id, or local checkpoint
  root; defaults to `sfx-woosh-dvflow-8s`
- `--synchformer-model`: Synchformer model for raw video input; defaults to
  `sfx-woosh-synchformer`
- `--negative-prompt`: negative text conditioning for MMAudio
- `--output`, `-o`: output WAV; defaults to
  `./mererun-sfx-video-<timestamp>.wav`
- `--duration`: seconds; defaults to `8`
- `--steps`, `-s`: DVFlow usually uses 4 and VFlow 32
- `--cfg`: Woosh guidance; defaults to 3.0 for DVFlow and 4.5 for VFlow
- `--renoise`: renoise amount, or a comma-separated DVFlow schedule in [0, 1]
- `--seed`: seed for deterministic generation
- `--sync-batch-size`: Synchformer segment batch size; defaults to `1`
- `--clip-batch-size`: MMAudio CLIP frame batch size; defaults to `4`
- `--preflight`, `--json`: inspect the request without loading MLX
- `--quiet`, `-q`: suppress stderr diagnostics

`--preflight --json` reports the input mode, output path state, VFlow/DVFlow
model availability, raw-video conditioning requirements, duration, effective
step count, CFG, renoise schedule, and follow-up actions. `.npy` feature inputs
don't require Synchformer during preflight or generation.

### `mere.run video cosmos3`

Run the complete pinned `nvidia/Cosmos3-Edge` checkpoint
(`video-cosmos3-edge-mlx`) through native Swift/MLX:

```bash
mere.run model pull video-cosmos3-edge-mlx

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

`--mode` accepts `text-to-video` (default), `text-to-image`, `image-to-image`,
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

Key options:

- `--output`, `-o`: output PNG or MP4; `--actions-output` writes predicted
  actions as JSON
- `--image`, `--video`: conditioning image or video. `--keep-video-tail` uses
  the tail of a long video, and `--condition-latent-frame` sets the
  conditioned latent frames for video-to-video (default `0, 1`).
- `--negative-prompt`: classifier-free negative prompt
- `--width`, `--height`: divisible by 16; default `1280` and `720`
- `--num-frames`: video modes require 4n+1; defaults to `189`
- `--steps`, `--guidance-scale`, `--shift`: default to the published value
  for the mode
- `--schedule nvidia|published-karras`: UniPC timestep schedule; defaults to
  `nvidia`
- `--seed`: defaults to `0`
- `--fps`: output frames per second
- `--action-domain` (default `camera_pose`), `--action-file`,
  `--action-chunk-size` (default `16`), `--action-resolution`
  (`256`, `480` (default), `704`, or `720`), and `--action-viewpoint`
  (`ego_view` (default), `third_person_view`, `wrist_view`, or
  `concat_view`): action controls
- `--max-new-tokens` (default `256`), `--temperature` (default `0.2`; `0` is
  greedy), `--top-p` (default `0.9`), and `--max-video-frames` (default `32`):
  reasoner controls
- `--quiet`, `-q`: suppress progress diagnostics

For sampling defaults, action JSON, exact upstream pins, and the resident
world-server recipe, run `mere.run guide video cosmos3`.

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

The required image and video masks use seven-color SCAIL-2 segmentation; to
prepare them, use [`video prepare-masks`](#mere-run-video-prepare-masks).

`--profile fast` is the default: 832x480, the separately pulled
`scail2-lightx2v-4step` adapter, and the published no-CFG four-step Euler
recipe with shift 5. `--profile quality` selects the configurable 40-step
UniPC/CFG recipe.

Key options:

- `--model`, `-m`: managed SCAIL-2 id or local MLX root; defaults to
  `video-scail2-14b-mlx`. `--model-root` takes precedence over `--model`.
- `--mode animation|replacement`: defaults to `animation`
- `--output`, `-o`: output MP4 path
- `--width`, `--height`: divisible by 32; default `832` and `480`
- `--steps` (default 40), `--guidance-scale` (default 5), `--shift`
  (default 3), and `--sampler unipc|euler` (default UniPC): quality-profile
  controls. The fast profile fixes four Euler steps, shift 5, and no CFG.
- `--distilled-adapter`, `--distilled-adapter-strength`: an installed
  adapter id or local distilled Wan 2.1 I2V safetensors file, and its
  multiplier (default `1.0`)
- `--additional-reference`, `--additional-reference-mask`: repeatable pairs
  for more subjects, matched by position
- `--seed`: defaults to `42`
- `--fps`: defaults to `16`
- `--segment-length`, `--segment-overlap`: pixel frames per long-video
  segment and clean-history overlap; both must equal 1 modulo 4 and default to
  `81` and `5`
- `--tail-policy drop|pad-trim`: defaults to `drop`. `pad-trim` preserves legal
  `1 mod 4` clip lengths and pads only the final incomplete temporal segment
  to the next legal length.
- `--audio-source none|driving`: defaults to `none`
- `--negative-prompt`: optional negative prompt
- `--preflight`, `--json`: validate without loading models
- `--quiet`, `-q`: suppress diagnostics on stderr

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
unique legal palette colors, project-materialized reference images,
text/box/positive/negative selectors, and optional point/box/dense painted-PNG
keyframe corrections. `--model` selects the SAM 3.1 model (default
`vision-segment-sam31`), and `--quiet` suppresses diagnostics. The output
directory must be new.

Preview mode (`--preview-frame`) segments references plus one selected
driving frame. Reference preparation aspect-fits each image and its mask into
the requested canvas; replacement mode mattes non-subject reference pixels to
black. Full mode tracks both directions, records gaps and quality warnings, and
emits prepared reference image/mask pairs, a ProRes 4444 categorical driving
mask, overlay MP4, contact sheet, tracking/quality JSON, normalized driver, and
a canonical hashed manifest.

### `mere.run video generate`

Generate MP4 video with the native LTX 2.3, LTX 2.5, MiniMax-H3, and Wan2.2
pipelines. The command prints the output MP4 path to stdout; progress and
diagnostics go to stderr. The default is a fast LTX 2.3 draft checkpoint with
video-only output, and the default model depends on the operation.

```bash
swift run mere.run video generate "<prompt>" [options]
```

Key options:

- `--model`, `-m`: managed video model id or local model root.
  `--model-root` takes precedence over `--model`.
- `--quality`: `draft` selects the fast LTX 2.3 standalone-distilled
  checkpoint; `final` selects the LTX 2.3 dev + distilled-LoRA pipeline or
  LTX 2.5 distilled
- `--output-mode`: `video-only` (default) or synchronized `audio-video`
- `--variant`: compatibility selector; `distilled` defaults to draft
  video-only and `unified-av` defaults to final audio-video. Don't combine it
  with `--quality` or `--output-mode`; an explicit `--model` still wins.
- `--output`, `-o`: output MP4 path
- `--width`, `--height`: omitted LTX 2.5 requests use the selected upstream
  recipe's native canvas
- `--num-frames`: omitted LTX 2.5 requests predict a duration within the
  official 1 to 20 seconds
- `--duration`: target seconds; overrides `--num-frames` using the model's
  native cadence
- `--fps`: defaults to `24`; LTX accepts fractional rates such as `23.976`
- `--seed`
- `--steps`: denoising schedule points; defaults to 40 for Wan, and MiniMax-H3
  selects 9, 16, or 21 from packed geometry
- `--h3-weight-mode`: `auto`, `quantized`, or `resident-bf16`
- `--h3-acceleration`: `quality`, `balanced`, `maximum`, or the experimental
  `layers-45`, `layers-40`, `velocity-reuse-2`, and `token-reduction` A/B arms
- `--h3-render-width`, `--h3-render-height`: optional same-aspect internal H3
  canvas. Both must be 32px multiples no larger than the output canvas; decoded
  frames are high-quality upscaled back to `--width` and `--height`
- `--h3-adapter`: installed MiniMax-H3 adapter catalog id or local safetensors
  path
- `--h3-adapter-strength`: MiniMax-H3 runtime adapter multiplier
- `--h3-frame`: repeatable zero-based `FRAME:PATH` FL2VA image injection
- `--h3-window-frames`: resident H3 sliding-window size in `17*n+5` frames
- `--h3-window-overlap`: motion/audio overlap in `17*n+1` frames (default `18`)
- `--guidance-scale`, `--shift`: Wan2.2 guidance and flow shift; both default
  to `5.0`
- `--negative-prompt`: negative prompt for Wan or full LTX generation;
  defaults to the selected pipeline's official prompt
- `--audio`: source audio path; automatically selects native LTX 2.3 A2Vid
- `--audio-start-time`: source segment offset in seconds (default `0`)
- `--a2v-guidance-scale`: audio-modality guidance (default `3`)
- `--video-cfg-guidance-scale`: full/A2Vid video text CFG (default `3`)
- `--audio-cfg-guidance-scale`: full unified-AV audio text CFG (default `7`)
- `--v2a-guidance-scale`: full unified-AV video-to-audio modality guidance
  (default `3`)
- `--a2v-steps`: full/dev stage-one steps (default `30`)
- `--image`, `--image-strength`: optional start frame for image-to-video and
  its strength from 0 through 1 (default `1.0`)
- `--end-image`, `--end-image-strength`: optional end keyframe so the clip
  interpolates a directed start-to-end motion; requires `--image`
- `--reference`: repeatable ordered MiniMax-H3 Ref2VA input in
  `image:path`, `video:path`, or `audio:path` form
- `--preflight`: print a structured plan without loading a model
- `--json`: only with `--preflight`
- `--timings`: native LTX 2.3 split-distilled, unified-AV, or A2Vid phase
  timings on stderr; unavailable for legacy merged distilled roots
- `--timings-output`: write those timings as JSON
- `--quiet`, `-q`: suppress progress output
- `--progress-json`: JSON progress lines on stderr for the MiniMax-H3 and Wan
  lanes (stage plus `denoising` steps, and `window` for H3 sliding windows);
  the native LTX lanes expose no per-step callback and emit none
- `--receipt`: final JSON result line listing the MP4 (or the EXR directory
  with `--skip-mp4`) and, on LTX lanes, the `--timings-output` JSON; see
  [Machine-readable receipts and progress](#machine-readable-receipts-and-progress)

LTX 2.5 adds its own option groups for distilled, full, HQ, keyframe, Diffusion
Fidelity Rendering (DFR), IC-LoRA, HDR, and generated-AV workflows. Run
`mere.run video generate --help` for every option, and see
[Use LTX 2.5 pipelines](/runtime/video#use-ltx-2-5-pipelines) for recipes:

- Pipeline: `--ltx-preset standard|hq`, `--ltx-pipeline
  two-stage|keyframe-interpolation|dev-one-stage`, `--ltx-sampler`,
  `--ltx-sigmas`, `--ltx-stage-2-sigmas`, `--skip-stage-2`, `--dfr`, and
  `--temporal-upsample-rounds`
- Conditioning: `--image-conditioning PIXEL_FRAME:PATH[:STRENGTH[:CRF]]`,
  `--generated-keyframe`, `--num-generated-keyframes`, `--auto-duration`,
  `--enhance-prompt` with `--prompt-enhancer-model`, and `--text-embeddings`
- Adapters: `--lora PATH[=STRENGTH]`, `--video-conditioning`,
  `--detailing-lora`, the `--conditioning-attention-*` and
  `--reference-*-factor` overrides, and the `--distilled-lora-strength-stage-*`
  pair
- Guidance: the `--video-*` and `--audio-*` STG, rescale, and skip-step
  controls, plus `--audio-max-duration`
- HDR and decoding: `--hdr srgb-linear|acescg|acescct`, `--hdr-transfer`,
  `--high-quality-hdr`, `--skip-mp4`, `--video-decoder
  convolutional|diffusion`, `--spatial-tile`, and `--spatial-overlap`
- Performance: `--ltx-transformer-execution eager|compiled`,
  `--ltx-guidance-projection-cache`, and the `--ltx-teacache` family

Environment:

- `MERERUN_VIDEO_LTX_MODEL_ROOT`
- `MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT` for an external
  `mlx-community/gemma-3-12b-it-4bit` checkout used by `video-ltx23-av-mlx`,
  `video-ltx23-full-mlx`, and the legacy `video-ltx23-a2vid-mlx`

For `--output-mode audio-video`, keep `--fps 24` unless you're deliberately making
a retimed clip. LTX 2.3 unified AV is trained around 24 fps; using 8 fps can
make generated motion look slow while audio remains normal. Use `--duration`
for clip length so the CLI can choose the nearest legal `8n+1` frame count.
Use the default `--quality draft` lane for fast iterations. Use `--quality
final` for the LTX 2.3 high-quality two-stage checkpoint. Output is a separate
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

MiniMax-H3 always emits synchronized 24 fps video and 32 kHz stereo audio.
FL2VA accepts `--image`, optional `--end-image`, and up to 12 repeatable
`--h3-frame FRAME:PATH` conditions at exact zero-based output indices. Ref2VA
accepts ordered `--reference` values and rejects FL2VA keyframe flags. H3 dimensions snap to
32-pixel multiples, frame counts snap upward to `17*n+5`, and its
CFG-distilled transformer needs one evaluation per schedule step. Ref2VA is an
explicit managed pull with an 8-bit transformer and conditioner; 8-bit is the
published Ref2VA quality floor.
When `--steps` is omitted, H3 selects 9, 16, or 21 schedule points from packed
row cost. Maximum acceleration caps the automatic schedule at 12 points. The
compact BF16 and Q8 cache pack selects exact 5, 9, 12, 16, 21, or 31-point
tables at shifts 12/3 and exact 5- and 9-point shifts-6/3 tables for the LightX2V
768p recipe. Custom schedules interpolate from the densest table and disclose
that they are not bit-exact. `--h3-weight-mode auto` keeps compact quantized
weights on MacBooks below 96 GiB and expands to the faster resident BF16 path on memory-qualified
desktops and 96+ GiB MacBooks when the requested geometry leaves the required
runtime reserve.

`--h3-acceleration quality` is the dense, exact default. At 12,000 or more
packed rows, `balanced` and `maximum` add dynamic block-sparse attention for
target-video queries. Prefix queries and keys, neighboring video blocks, the
first two layers, the leading schedule region, and the final evaluation stay
dense. Skipped blocks retain a summary correction, and the Metal path must pass
a once-per-shape dense-route numerical gate before it can run.

Without an H3 adapter, `balanced` and `maximum` also use a modality-aware
adaptive first-block cache. Every evaluation still runs block 1, then measures
global and worst-time-slice drift for video and audio against the last full
refresh. A qualifying step reuses only the target residual from blocks 2
through 50. `balanced` admits at most two adjacent hits and reserves the final
two evaluations; `maximum` admits four adjacent hits and always executes the
final evaluation in full. Both require two complete evaluations before cache
reuse. These modes remain approximate: the same prompt and seed can follow a
different motion or composition trajectory. Use `quality` when exact-seed
fidelity matters.

`velocity-reuse-2` is a separate experimental h3.c transfer arm. It retains the
quality schedule, runs the first and final denoise evaluations in full, and
linearly extrapolates complete video and audio velocity outputs from the two
most recent full evaluations on intervening odd steps. The two modalities use their
independent shifted schedules. It does not compose with dynamic-sparse
attention or either block-cache policy. Use it only for controlled same-seed
FL2VA and Ref2VA comparisons until the quality envelope is published.

Ref2VA reference images preserve aspect ratio and are downscaled only when
their area exceeds the internal render canvas. Standalone reference audio keeps
its complete 2-15 second duration; all ordered reference audio remains capped
at 15 seconds total. Condition augmentation and target video/audio latents use
the released independent seeded streams.

`layers-45` and `layers-40` are separate gate-ranked thinning arms. They rank
the cached schedule's mean absolute attention/MLP AdaLN gates, always protect
blocks 0, 1, and 49, and skip the lowest remaining scores. They save
block execution only: mere.run still retains every loaded transformer weight,
so these modes don't claim h3.c's weight-residency reduction.

`token-reduction` is the isolated h3.c token-pairing arm. Blocks 0 through 3
run on the full packed sequence. Only adjacent horizontal target-video tokens
are averaged; text, condition media, references, target audio, and odd trailing
video tokens remain exact. The reduced sequence runs through block 39 for the
first ten denoise evaluations and through block 29 afterward. Reconstruction
adds each reduced token's change from its pooled baseline back to both saved
full-resolution source tokens before the remaining full-grid blocks. This mode
does not compose with sparse attention, layer thinning, or denoise-step reuse
and remains non-default pending FL2VA and Ref2VA quality receipts.

For reduced-canvas comparisons, set both internal dimensions explicitly. For
example, `--width 512 --height 512 --h3-render-width 384
--h3-render-height 384` runs DiT and VAE decode at 75% linear resolution, then
uses the pinned oracle's high-quality vImage ARGB8888 scaler to emit 512x512
frames. The internal canvas must preserve aspect exactly. A reduced canvas
can't be combined with sliding-window continuation, because the continuation
would condition on the wrong resolution.

`--h3-window-frames` enables resident sliding windows for FL2VA or Ref2VA. The
window count must be `17*n+5`; `--h3-window-overlap` must be `17*n+1` and leave
at least 22 target frames. The runtime conditions each new window on prior
overlap motion, its final boundary frame, and matching generated stereo audio,
then appends only new frames and samples. The output duration and all
`--h3-frame` indices stay on one global timeline. Transformer, conditioner,
AdaLN table, reference encodings, and VAEs remain loaded between windows.

`--h3-adapter minimax-h3-turbo-4step` selects the separately pulled EMA-850
LoRA, while the `minimax-h3-lightx2v-*` ids select pinned LightX2V PEFT LoRAs.
The FL2VA releases target `video-minimax-h3-fl2va-bf16-mlx`. The legacy releases use five
schedule points (four transformer evaluations). The v1.0 8-step release
defaults to nine schedule points and accepts the published five-point fallback;
the v1.0 four-step 768p release uses five points, video/audio shifts 6/3, and
alpha 128. The v1.0 eight-step 768p release accepts only nine points and uses
shifts 6/3 and alpha 8.
EMA-850
runs in activation space; LightX2V is fused once into the BF16 transformer
before denoising and adds no LoRA matmuls to the generation loop. Both require
dense execution of all 50 blocks and prohibit denoise-step cache reuse, but
can use the attention-only `balanced` or `maximum` path. The separate
`minimax-h3-lightx2v-ref2v-4step-v0.1` adapter targets
`video-minimax-h3-ref2va-mlx`, requires ordered references, and selects five
schedule points with shifts 12/3 and alpha 8. Its managed INT8 transformer is
expanded to resident BF16 before fusion, so forced quantized execution is
rejected. The preflight report resolves the managed
adapter path, verifies its presence, and preserves the adapter id, strength,
and resolved schedule in the declarative action.

The pinned compact BF16 and Q8 roots contain the exact nine-point shifts-6/3
AdaLN table used by the eight-step 768p adapter. Full BF16 source roots compute
the same schedule from their AdaLN weights. Only schedules outside the cache
pack report `interpolating-adaln-cache-not-bit-exact`.

Preflight mode:

- `--preflight --json` prints a structured plan without loading MLX, loading a
  video model, creating directories, or writing an MP4.
- the report includes model availability, output path state, source/end image
  and audio state, resolved dimensions, resolved frame count/duration, source
  audio offset, seed, input mode, the resolved H3 steps/weight/acceleration
  policy, timed-frame count, and sliding-window geometry/count when applicable,
  whether audio conditions generation, whether the
  source soundtrack is preserved, diagnostics, and declarative actions.
- blockers such as a missing model root, missing image, invalid frame rate, or
  `--end-image` without `--image` produce JSON and a nonzero exit. Requesting
  phase timings on a legacy merged distilled root or Wan2.2 is also blocked.
- notes such as dimension/frame snapping remain machine-readable so a UI can
  explain the exact render that runs.

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

swift run mere.run model pull video-minimax-h3-ref2va-mlx --accept-model-license
swift run mere.run video generate \
  "keep the reference subject and follow the camera movement" \
  --model video-minimax-h3-ref2va-mlx \
  --reference image:./subject.png \
  --reference video:./camera.mp4 \
  --num-frames 124 \
  --output ./h3-ref2va.mp4
```

### `mere.run video session`

Keep a standalone distilled or full LTX 2.3 or LTX 2.5 runtime resident while
processing serial JSONL generation requests. The default model is
`video-ltx23-av-mlx`.

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
don't reload or mutate the base transformer.

Key options:

- `--model`, `-m`, `--model-root`: managed LTX 2.3 or LTX 2.5 id, or local
  root; `--model-root` takes precedence
- `--prompt-cache-capacity`: materialized prompt embeddings kept resident;
  defaults to `8`, and `0` disables the cache
- `--video-decoder convolutional|diffusion`: LTX 2.5 VAE decoder
- `--ltx-transformer-execution eager|compiled`: defaults to `eager`; compiled
  execution can change floating-point results slightly
- `--ltx-guidance-projection-cache automatic|disabled|enabled`: defaults to
  `disabled`
- `--ltx-teacache`, `--ltx-teacache-threshold`: session defaults for
  calibrated TeaCache reuse on full LTX 2.5 requests
- `--quiet`, `-q`: suppress session diagnostics

### `mere.run video export-latents`

Run native distilled LTX denoising and export the final-stage latents as
safetensors.

```bash
swift run mere.run video export-latents \
  --model-root /path/to/distilled-ltx \
  --output out.safetensors \
  "a cinematic drone flyover at sunrise"
```

The default model is `video-ltx-av`; `--model-root` takes precedence over
`--model`. `--width` and `--height` must be divisible by 64 and default to
`768` and `512`. `--num-frames` must satisfy 8n+1 and defaults to `65`.
`--seed` defaults to `42`, and `--quiet` suppresses progress.

### `mere.run video retake`

Regenerate a timed video and audio region with native LTX 2.5, keeping the rest
of the source timeline.

```bash
swift run mere.run video retake "the actor turns and smiles at the camera" \
  --source ./scene.mp4 \
  --start-time 2.0 \
  --end-time 4.5 \
  --output ./scene-retake.mp4
```

Key options:

- `--source`: required source MP4 or MOV, or an EXR-frame directory (which
  also needs `--frame-rate`)
- `--start-time`, `--end-time`: required inclusive start and exclusive end in
  seconds
- `--preserve-video`, `--preserve-audio`: retake only audio or only video
- `--model`, `-m`: official LTX 2.5 id or local root; defaults to
  `video-ltx25-distilled-bf16`. `--model-root` sets an explicit root.
- `--output`, `-o`: output MP4 path
- `--seed`: defaults to `10`
- `--negative-prompt`, `--steps` (default `30`), and `--sigmas`: settings
  for an optional full checkpoint
- `--enhance-prompt`, `--prompt-enhancer-model`,
  `--prompt-enhancer-model-root`: native Gemma-4 prompt enhancement
- `--lora PATH[=STRENGTH]`: repeatable LTX LoRA
- `--video-*` and `--audio-*` guidance controls: CFG (video `3.0`, audio
  `7.0`), STG scale and block, rescale, modality scale, and skip step
- `--video-decoder diffusion|convolutional`: defaults to `diffusion`
- `--hdr`, `--hdr-transfer`: HDR source and working space
- `--quiet`, `-q`: suppress progress diagnostics

### `mere.run video dub-it`

Generate synchronized video and audio that keep the identity from one LTX 2.5
IC-LoRA reference, using the official two-stage distilled Dub-It recipe. The
frame count and frame rate come from the reference video. Its video conditions
both stages, and its audio is appended as clean reference tokens.

```bash
swift run mere.run video dub-it "The speaker performs on a rain-lit street" \
  --reference-video ./speaker.mp4 \
  --ic-lora ./dub-it.safetensors \
  --output ./speaker-street.mp4
```

Key options:

- `--reference-video`: required container with both video and audio tracks
- `--ic-lora`: required, exactly one Dub-It IC-LoRA adapter.
  `--ic-lora-strength` defaults to `1.0`.
- `--reference-strength`: video reference strength from 0 through 1;
  defaults to `1.0`
- `--model`, `-m`, `--model-root`: defaults to `video-ltx25-distilled-bf16`
- `--output`, `-o`: output MP4 path
- `--width`, `--height`: divisible by 64; default `1536` and `1024`
- `--seed`: defaults to `10`
- `--image-conditioning PIXEL_FRAME:PATH[:STRENGTH[:CRF]]`: repeatable timed
  image guide
- `--stage-1-sigmas`, `--stage-2-sigmas`: explicit sigma schedules
- `--enhance-prompt`, `--prompt-enhancer-model`,
  `--prompt-enhancer-model-root`: native Gemma-4 prompt enhancement
- `--video-decoder diffusion|convolutional`: defaults to `diffusion`
- `--quiet`, `-q`: suppress progress diagnostics

For workflow guidance, see [Retake a region](/runtime/video#retake-a-region)
and [Transfer identity with Dub-It](/runtime/video#transfer-identity-with-dub-it).

### `mere.run world serve`

Serve one warm native world-model session over HTTP. The default backend is
DreamX (`video-dreamx-world-5b-ar-mlx` on the `video-wan22-ti2v-5b-mlx` base);
`--backend cosmos3` selects Cosmos3.

```bash
swift run mere.run world serve \
  --base-model video-wan22-ti2v-5b-mlx \
  --model video-dreamx-world-5b-ar-mlx \
  --prepare
```

Key options:

- `--host`: defaults to `127.0.0.1`
- `--port`: defaults to `8791`
- `--api-key`: bearer token for world endpoints; also read from
  `MERERUN_API_KEY`
- `--backend dreamx|cosmos3`: defaults to `dreamx`
- `--base-model`, `--model`: Wan2.2 TI2V base and converted DreamX model ids
  or local roots
- `--state-directory`: directory for transition videos and terminal state
  frames
- `--prepare`: load and warm all models before accepting requests
- `--disable-scene-memory`: turn off DreamX geometry-guided revisit memory.
  The `--scene-memory-*` options tune recycling strength (default `0.08`),
  retained frames (default `96`), minimum gap, and revisit and exact-restore
  tolerances.

For the HTTP lifecycle, camera controls, and authentication boundary, see
[Persistent world runtime](/runtime/world).

### `mere.run graph`

Validate, materialize, run, and submit portable workflow graphs. Each command
takes a workflow graph JSON file and, where relevant, an `--inputs-json` file
of input values. Most commands accept `--json` for a structured report. For
the graph format, job bundles, and executors, see
[Portable workflows](/workflows).

| Command | What it does |
| --- | --- |
| `graph catalog` | List registered workflow node contracts. |
| `graph dataset discover <root>` | Find and rank image-caption dataset directories without loading models. `--max-depth` defaults to 6 and `--limit` to 100. |
| `graph validate <file>` | Validate a typed graph without executing it. |
| `graph preflight <file> --executor <ref>` | Preflight a graph against an executor. |
| `graph materialize <file> --run-dir <dir>` | Create an immutable job bundle in a run directory. |
| `graph export-job <file> --output <path>` | Export an immutable job bundle for any executor. |
| `graph run <file> --run-dir <dir>` | Run a graph locally. `--resume` reuses verified outputs from an interrupted run, and `--json-stream` emits NDJSON events. |
| `graph run-job --bundle <path> --run-dir <dir>` | Run an existing job bundle locally, with the same `--resume` and `--json-stream` options. |
| `graph submit <file> --executor <ref> --run-dir <dir>` | Submit a graph job to an SSH or relay executor. |
| `graph submit-job --bundle <path> --executor <ref> --run-dir <dir>` | Submit an existing job bundle to SSH or relay. |
| `graph worker probe` | Report worker capabilities. |
| `graph worker execute --bundle <path> --run-dir <dir>` | Execute a materialized job bundle with the worker protocol. |
| `graph worker inspect --run-dir <dir>` | Inspect a worker run manifest. |
| `graph worker cancel --run-dir <dir>` | Request cooperative worker cancellation. |

```bash
mere.run graph validate workflow.json --inputs-json inputs.json --json
mere.run graph run workflow.json --inputs-json inputs.json --run-dir ./runs/local --json-stream
mere.run graph submit workflow.json --inputs-json inputs.json --executor relay:fleet --run-dir ./runs/fleet
```

### `mere.run graph batch`

Distribute independent, complete workflow graphs across a relay fleet. Every
batch command requires `--executor relay:<profile>` and always prints
machine-readable JSON.

| Command | What it does |
| --- | --- |
| `graph batch submit <file> --items-jsonl <file> --run-dir <dir>` | Prepare and submit a durable batch from `{key, inputs}` rows. Reuse the run directory to resume submission. `--max-in-flight` (1 through 20, default 8) caps active children, and `--max-attempts` (1 through 4, default 1) sets infrastructure attempts per child. |
| `graph batch list` | List durable batches in the relay account. |
| `graph batch inspect <batch-id>` | Show batch progress and item output references. |
| `graph batch watch <batch-id>` | Stream batch changes as NDJSON until the batch finishes. `--timeout-seconds` stops early; `0` (default) waits indefinitely. |
| `graph batch cancel <batch-id>` | Stop admission and cancel unfinished children. |
| `graph batch retry <batch-id>` | Retry failed items and keep completed outputs. |
| `graph batch fetch <batch-id> --into <dir>` | Fetch verified outputs, one run per completed item. |

Use retries only for replay-safe workflows. For scheduling and limits, see
[Fleet workflow batches](/workflow-batches).

### `mere.run executor`

Manage the local, SSH, and relay executor profiles that `graph preflight`,
`graph submit`, and `run` commands use. Each command accepts `--json`.

| Command | What it does |
| --- | --- |
| `executor add ssh <name> --destination <user@host> --remote-root <dir>` | Add an SSH executor. Optional `--port`, `--identity-file`, and `--mere-run-path` (default `mere.run`). |
| `executor add relay <name> --url <url>` | Add a relay executor. No default URL is assumed; `--token-file` reads a bearer token from a file. |
| `executor list` | List executor profiles. |
| `executor inspect <ref>` | Inspect a profile. |
| `executor probe <ref>` | Probe executor capabilities. |
| `executor login <ref>` | Sign in to a relay through its advertised device authorization flow. |
| `executor auth-status <ref>` | Inspect relay authentication without printing credentials. |
| `executor logout <ref>` | Remove the saved relay credential. |
| `executor fleet <ref>` | Show relay node identity, eligibility, inventory, and recent work. |
| `executor node-refresh <ref> <device-id>` | Ask a connected node to rescan its capabilities and installed models. |
| `executor node-configure <ref> <device-id>` | Update a node's scheduling policy with `--enable`, `--disable`, `--drain`, `--undrain`, `--priority` (0 through 100), or `--display-name`. |
| `executor remove <ref>` | Remove a profile. |

```bash
mere.run executor add ssh gpu-box --destination dana@gpu.example.com --remote-root ~/.cache/mere.run
mere.run executor add relay fleet --url https://relay.example.com
mere.run executor probe ssh:gpu-box --json
```

For profile references and authentication, see
[Executor profiles](/workflows#executor-profiles).

### `mere.run relay serve`

Serve the relay graph-job API from this machine and run submitted jobs
locally. The command prints its address and a pairing code that other
machines exchange for a bearer token.

```bash
mere.run relay serve --name studio
```

Key options:

- `--host`: interface to bind; defaults to `0.0.0.0`
- `--port`: defaults to `6373`
- `--name`: display name; defaults to the machine's hostname
- `--pairing-window`: minutes after startup during which new devices can
  pair; defaults to `15`

For pairing and security, see
[Direct relay](/workflows#direct-relay-relay-serve).

### `mere.run eval`

Run reproducible evaluations from external, content-addressed packs.

```bash
mere.run eval pack validate ./packs/assistant-quality --json
mere.run eval run ./packs/assistant-quality --model text-chat-gemma4 --dry-run
mere.run eval run ./packs/assistant-quality \
  --model text-chat-gemma4 \
  --adapter mere-platform-assistant \
  --checkpoint ./eval-checkpoint.json \
  --output ./report.json
mere.run eval promote ./report.json --output ./receipt.json
```

- `eval pack validate <pack>`: validate and hash a pack without running
  models
- `eval run <pack>`: run a matched, adapter-aware evaluation
- `eval promote <report>`: create a content-addressed receipt for a complete,
  gate-passing report; `--output` sets the receipt path

`eval run` options:

- `--model slot=model-id`: repeatable model binding; a lone id binds the base
  slot
- `--adapter slot=catalog-id-or-path`: repeatable adapter binding; a lone
  reference binds the candidate slot
- `--trials`, `--max-tokens`, `--context-size`: override the pack values
- `--logprobs none|summary|tokens|top`, `--top-logprobs`: override logprob
  capture
- `--allow-external-scorer`: authorize the pack-pinned scorer process
- `--log-responses`: include model responses in the local report
- `--dry-run`: print the full plan without loading models or running scorers
- `--checkpoint`, `--resume`, `--case-trial-limit`: checkpoint every completed
  row, resume a matching checkpoint, and stop cleanly after a number of new
  rows (requires `--checkpoint`)
- `--output`: write the report JSON atomically
- `--json`: print the report as JSON on stdout

For pack layout, scorers, and gates, see
[External evaluation packs](/evaluation-packs).

### `mere.run model list`

List all managed model IDs and their shallow availability without recursively
scanning payloads. Pass `--measure-sizes` to calculate referenced sizes; those
values follow symlinks and are not additive when models share payloads.

```bash
swift run mere.run model list
swift run mere.run model list --measure-sizes
swift run mere.run model list --json
```

`--json` prints the typed inventory, declared context windows, and usage
terms.

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

### `mere.run model location`

Use models stored outside the writable model store without copying or
deleting them. Search roots hold directories named for canonical model ids;
bindings map one id to any directory. Both are read-only.

```bash
mere.run model location list --json
mere.run model location add /Volumes/Archive/mererun-models
mere.run model location bind image-zimage-nano /Volumes/Archive/zimage-nano
mere.run model location unbind image-zimage-nano
mere.run model location remove /Volumes/Archive/mererun-models
```

- `model location list`: show the writable store, search roots, and bindings
- `model location add <path>`: register an existing search root
- `model location remove <path>`: unregister a search root without deleting
  files
- `model location bind <model-id> <path>`: bind an id to a directory. A model
  with listed third-party terms needs `--accept-model-license`.
- `model location unbind <model-id> [<path>]`: remove one binding, or every
  binding for the id when you omit the path, without deleting files

For details, see
[Registered catalog locations](/runtime/model-management#registered-catalog-locations).

### `mere.run model optimize`

Build reusable inference-only artifacts for a supported installed model, such
as MiniMax-H3 MLX or LTX 2.5:

```bash
mere.run model optimize ./MiniMax-H3-FL2VA-full-MLX
mere.run model optimize video-ltx25-full-bf16
mere.run model optimize /path/to/LTX-2.5 --text-encoder-only
mere.run model optimize ./MiniMax-H3-FL2VA-full-MLX --json
```

The generated `adaln_cache.index.json` binds exact AdaLN tables for 5, 9, 12,
16, 21, and 31 points at shifts 12/3 plus the LightX2V 768p 5- and 9-point
schedules at shifts 6/3. Compatible H3 generations skip the 13B-parameter
AdaLN/time-embedding branch. A custom schedule interpolates from the densest
compatible table and emits a visible non-bit-exact diagnostic. `--force`
atomically rebuilds a pack from a full legacy root; a pruned root can validate
its existing pack but cannot synthesize a missing exact table. Managed compact
BF16, Q8, legacy Q4, and Ref2VA artifacts already include source-bound caches.

For official or offline LTX 2.5 roots, the command streams the distilled
transformer and, for a full root, the dev transformer into source-bound BF16
native packs. It also writes a compact connector pack so text setup does not
open the 42 GB official transformer. Transformer packs use mere.run's
module-key namespace and every artifact preserves its tensor payload bytes.
Managed Distilled and Full downloads are already pre-keyed, so the command
reports their existing artifacts without creating another copy.
`--text-encoder-only` reproduces the Distilled Q4 text pack from an official
BF16 root for offline and development use; it retains the BF16 projection and
tokenizer assets while quantizing eligible Gemma language weights.

For Nemotron Omni or DRAMA-3, `--output <dir>` writes a standalone native
checkpoint to a directory. `--json` prints the result as JSON.

### `mere.run model remove`

Remove an install by canonical id and reclaim backing payloads that no
remaining managed or legacy link uses. The command asks for confirmation unless
you pass `--force`.

```bash
mere.run model remove image-zimage-nano
mere.run model remove image-zimage-nano --force --json
mere.run model remove image-zimage-nano --keep-cache
```

`--keep-cache` removes the model links but keeps unshared Hub payloads.
`--json` prints a structured result and requires `--force`.

### `mere.run status`

Show a local snapshot: whether the API server answers, which model it
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
- `--timeout-seconds`: network probe timeout; defaults to `1.0`
- `--json`: emit a structured snapshot for scripts and agents

### `mere.run model runtime get`

Read typed per-model API runtime settings from the active model store. The
argument is a managed model id or a configured alias.

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

Each setting has a matching `--clear-*` flag, such as `--clear-alias`,
`--clear-ttl`, `--clear-min-p`, or `--clear-engine`, that removes the value.
`--unpinned` clears `--pinned`, and `--json` prints the updated settings.

`--engine` sets an engine override, validated against the curated catalog:
`text-code`, `text-chat-klein`, `text-chat-gemma4`,
`text-chat-diffusiongemma`, `text-chat-flint`, `text-chat-kolibri`,
`text-chat-laguna`, `text-chat-q36`, `text-chat-q35`, `text-chat-lfm2`,
`text-chat-deepseek-v4-flash`, `text-chat-muse-glimmer`,
`text-chat-nemotron-h`, or `text-chat-nemotron-omni`.

`--kv-cache-mode` accepts `default`, `affine4`, `affine8`, `polar2`, or
`auto`. Gemma4, Qwen-family, and LFM2 accept
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
swift run mere.run model pull image-zimage-nano
swift run mere.run model pull image-zimage-nano --preflight --json
swift run mere.run model pull video-minimax-h3-fl2va-bf16-mlx \
  --cache-dir /Volumes/Models/MereRun/hub \
  --accept-model-license
swift run mere.run model pull --all
```

Use `--preflight --json` to check support, install state, download source,
model store path, hub cache path, disk headroom, and next actions before any
download starts. The report includes `pull-model` or `pull-models` actions and
exits nonzero after printing JSON when hard blockers are present.

`--cache-dir PATH` selects the content-addressed Hub cache for that pull,
including disk estimates and downloads. The installed model root links to the
selected cache, so disconnecting an external cache volume makes the model
unavailable until it is reconnected.

`--force` re-downloads an installed model, and `--quiet` suppresses progress.
`--allow-unsupported` bypasses the Apple Silicon and unified-memory support
check; use it only when you intentionally accept the runtime risk.

Models that are access-gated or carry material non-commercial, research-only,
or revenue-limited terms require `--accept-model-license` before a
download. A custom license alone does not trigger the flag.
The preflight reports the exact component terms and blocks without
acceptance; `--all` skips restricted entries unless the flag is present.
`--accept-license-terms` is an equivalent, clearer alias for
`--accept-model-license`. Passing either spelling and continuing with the download confirms that
the user reviewed and accepts the listed terms and agrees to comply. mere.run
records the immutable source revisions and accepted term URLs in the installed
`mererun_model.json`, but the user remains responsible for
deciding whether their intended use complies. See
[`model-sources.md`](./model-sources.md#restricted-model-downloads).

### `mere.run adapter list` and `mere.run adapter pull`

List the built-in public adapter catalog or install one immutable release:

```bash
mere.run adapter list
mere.run adapter list --json
mere.run adapter pull mere-platform-assistant
mere.run adapter pull flux2-dev-turbo-8step --accept-license
mere.run adapter pull scail2-lightx2v-4step
mere.run adapter pull minimax-h3-turbo-4step
mere.run adapter pull minimax-h3-lightx2v-4step
mere.run adapter pull minimax-h3-lightx2v-8step-v1
mere.run adapter pull minimax-h3-lightx2v-4step-v1-768p
mere.run adapter pull minimax-h3-lightx2v-8step-v1-768p
mere.run adapter pull minimax-h3-lightx2v-ref2v-4step-v0.1
mere.run adapter pull minimax-h3-fasth3-vsa-datafree-4step --accept-license
```

The pull verifies the cataloged byte count and SHA-256 before atomically
installing the adapter. `--force` re-downloads a verified install,
`--quiet` suppresses progress, and `--accept-license` confirms the listed
adapter and repository terms where the catalog requires it. Stdout contains only the installed path; progress and
verification diagnostics go to stderr. Use the adapter id directly with
`image generate --lora`, `text chat --lora`, `api serve --lora`, or the
matching SCAIL `video animate --distilled-adapter` option. `video animate
--profile fast` selects `scail2-lightx2v-4step` and its fixed four-step
schedule.
`image generate --lora flux2-dev-turbo-8step` selects the Turbo adapter's
eight-step schedule and guidance default for `image-flux2-dev`. Repeat
`--lora PATH_OR_ID[=SCALE]` to add local FLUX.2-dev adapters to the same run.
For `image-flux1-dev`, repeat the same option to stack compatible FLUX.1
adapters. FLUX.1, FLUX.2-dev, and Klein adapters aren't interchangeable.
Use the MiniMax-H3 adapter ids with `video generate --h3-adapter`. FL2VA
adapters support the compact BF16 and affine Q8 FL2VA bases and reject the
legacy Q4 compatibility package. The Ref2V adapter uses the managed Ref2VA base
expanded to resident BF16. All bases remain subject to MiniMax-H3 Community
License acceptance.

The preferred FastH3 path is a single managed model pull:

```bash
mere.run model pull video-minimax-h3-fasth3-vsa-datafree-mlx \
  --accept-model-license
mere.run video generate "a lighthouse in a winter storm" \
  --model video-minimax-h3-fasth3-vsa-datafree-mlx \
  --output ./lighthouse-fasth3.mp4
```

This package contains the premerged affine Q8/group-64 FastH3 student, Q8
compression gates, Q8 text encoder, both VAEs, tokenizer, and source-bound
AdaLN cache. It doesn't require another download or preparation step after the
model pull. The lower-level adapter command remains available to developers who
build or verify the package. FastH3 accepts text-only FL2VA, requires adapter
strength `1.0`, and uses the quality acceleration mode. Managed affine Q8
FastH3 roots automatically use the Metal tiled MLP; developers can set
`MERERUN_H3_EXACT_KERNELS=disabled` for a portable-path comparison.

For a cross-command decision guide, see [Benchmarking](./benchmarking.md). The
following subsections provide the command reference for each benchmark lane.

### `mere.run model benchmark chat` and `tool-calls`

Run small evaluation slices against local chat models. `chat` runs the
grounded-chat `mere-chat-slice` suite against the local chat comparison lane.
`tool-calls` checks tool-call selection and defaults to Q36 and Gemma 4 12B
4-bit.

```bash
swift run mere.run model benchmark chat --dry-run --json
swift run mere.run model benchmark tool-calls --models text-chat-gemma4-12b-4bit --log-responses --json
```

Both commands share these options:

- `--models`: comma-separated model ids
- `--cases`: comma-separated case ids
- `--max-tokens`: per case; defaults to `96` for `chat` and `192` for
  `tool-calls`
- `--temperature` (default `0.0`), `--top-p` (default `1.0`), `--top-k`
  (default `0`, disabled), `--min-p`, and `--context-size`
- `--laguna-path`, `--laguna-dflash-path`, `--laguna-dflash-tokens`
  (1 through 15, default 12), and `--laguna-dflash-min-tokens` (default 32):
  local Laguna target and DFlash companion overrides
- `--dry-run`: print the plan without loading models
- `--log-responses`: include responses in the report
- `--json`: emit machine-readable JSON

`chat` also accepts `--concurrency` (default `1`) and
`--laguna-dflash-routing` (default `automatic`).

### `mere.run model benchmark fused`

Run the versioned Mere Lite or Mere Comprehensive fused quality suite. Code
cases execute generated Python, so real runs need `--allow-code-execution`.

```bash
swift run mere.run model benchmark fused --suite lite --models text-chat-gemma4-12b-4bit --dry-run --json
swift run mere.run model benchmark fused \
  --suite lite \
  --models text-chat-gemma4-12b-4bit \
  --allow-code-execution \
  --checkpoint ./fused-checkpoint.json \
  --json
```

Key options:

- `--suite lite|comprehensive`: defaults to `lite`
- `--models`: comma-separated model ids
- `--manifest`: override the bundled suite manifest
- `--external-cases`: hash-verified normalized JSONL fixtures for
  reference-only sources
- `--cases`, `--capabilities`: select cases by id or by capability tags
- `--trials`, `--max-tokens`: override the suite's trial count and per-case
  budget
- `--context-size`: defaults to `32768`
- `--logprobs summary|tokens|top` (default `summary`) and `--top-logprobs`
  (default `5`)
- `--performance-lane none|native`: add a separate lane with logprob capture
  off; defaults to `none`
- `--allow-code-execution`, `--execution-timeout` (default `5.0` seconds),
  `--python` (default `python3`), and `--sandbox
  auto|macos-sandbox-exec|bubblewrap|none` (default `auto`)
- `--checkpoint`, `--resume`, `--case-trial-limit`: checkpoint each completed
  case-trial, resume a matching plan, and stop after a number of new
  case-trials
- `--dry-run`, `--log-responses`, `--json`

`model benchmark fused-fixture <file>` stamps hashes into a normalized JSON or
JSONL fixture and writes the records to stdout; `--check` verifies existing
hashes instead. For the suite design and receipts, see
[Mere fused model evaluation](/benchmark-fused).

### `mere.run model benchmark laguna-dflash`

Measure Laguna S 2.1 target-only and DFlash decode in one resident process,
from local checkpoint directories.

```bash
swift run -c release mere.run model benchmark laguna-dflash \
  --laguna-path /path/to/Laguna-S-2.1-NVFP4-mlx \
  --laguna-dflash-path /path/to/Laguna-S-2.1-DFlash \
  --include-automatic \
  --json
```

Key options:

- `--laguna-path`, `--laguna-dflash-path`: required target and DFlash
  checkpoint directories
- `--decode-token-values`: exact decode lengths; defaults to
  `8,12,16,24,32,48`
- `--repetitions`: per mode and length; defaults to `3`
- `--laguna-dflash-tokens`: speculative tokens per round, 1 through 15;
  defaults to `12`
- `--prompt`, `-p`, `--prompt-file`, or `--fixture
  deterministic-prose|grounded-email|code-completion` (default
  `deterministic-prose`)
- `--context-size`: defaults to `4096`
- `--concurrency-values`, `--warmup-repetitions` (default `1`), and
  `--mixed-fixtures`: resident concurrency runs
- `--include-automatic`: also measure the adaptive routing policy
- `--temperature` (default `0.0`), `--top-p` (default `1.0`), `--top-k`
  (default `0`), and `--min-p` (default `0.02`)
- `--log-responses`, `--json`

### `mere.run model benchmark kolibri-logprobs`

Score fixed token sequences with native Kolibri and export raw logits to a new
directory with a receipt.

```bash
swift run mere.run model benchmark kolibri-logprobs \
  --model-root /path/to/kolibri \
  --suite ./sequences.json \
  --output ./kolibri-logits
```

`--suite` is a JSON array of fixed token sequences and scoring boundaries.
`--chunk-size` sets the prefill chunk size (default `32`), and
`--calibration-output` writes calibration-only expert input second moments to
a new safetensors file. For the runtime, see [Kolibri](/runtime/kolibri).

### `mere.run model benchmark parakeet-coreml`

Measure a prepared Parakeet Core ML artifact in one optimized resident process.
The command loads and verifies the artifact once, runs unmeasured warmups, and
reports feature extraction, encoder, decoder, alignment, merge, and total
duration for each repetition.

```bash
.build/release/mere.run model benchmark parakeet-coreml ./sample.wav \
  --artifact /path/to/parakeet-coreml \
  --warmups 2 \
  --repetitions 5 \
  --json
```

`--warmups` defaults to `2` and `--repetitions` to `5`. `--language` copies
an optional language label into the transcript result.

The command rejects Debug builds because compiler optimization changes the
Swift feature extraction and host-side decoder work. The report includes
transcript consistency across repetitions. It isn't a transcript-quality
evaluation.

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
supports `text-chat-q36-nano` (default), Qwen3.8 27B BF16 and Q4, the
Flash-Next mixed, Q3, native-PLE Q3, and Q4 ids, `text-agent-ornith-9b`, and the
official Ornith 1.5 Q4/Q6/Q8/BF16 ids. It runs the selected model with three
policies:

- `baseline`: MTP disabled with `MERERUN_Q35_MTP_SPECULATION=0`.
- `adaptive`: production policy (short-prompt MTP for Ornith 1.5; the measured
  long-context threshold for Qwen3.6).
- `forced`: MTP enabled with `MERERUN_Q35_MTP_SPECULATION=1` and a configurable
  forced threshold.

```bash
swift run mere.run model benchmark q36-mtp \
  --prompt-repeat-values 8,80,150 \
  --temperature-values 0,0.7 \
  --decode-tokens 32 \
  --json
```

The command forces exactly `--decode-tokens` per variant. Output includes prompt
tokens, generated tokens, load time, prefill time, decode time, TTFT, prefill
tok/s, decode tok/s, end-to-end tok/s, process resident memory, acceleration
counters, adaptive/forced speedups versus baseline, and greedy output SHA-256
parity. Greedy forced MTP uses the native
block verifier; non-greedy forced MTP stays on the exact probabilistic
speculative path. Use `--mtp-block-size` to test a different greedy draft block
cap and `--forced-mtp-min-prompt-tokens` to adjust the forced policy threshold.
Each fresh variant receives an untimed warm-up with prefix caching and
continuous batching disabled, so reported timing compares equivalent warm
single-request paths. `--warmups`, `--warmup-tokens`, and `--repetitions`
(each default `1`, with 16 warm-up tokens) control the warm-up and measured
requests, and `--variants` selects and orders `baseline,adaptive,forced`.

### `mere.run model benchmark q38-verification`

Measure Qwen-family target passes at linear verification widths. Flash-Next
accepts widths from one through 32. Qwen3.8 27B and Ornith 1.5 Q4 accept widths
from one through nine. The benchmark first generates an oracle sequence with the same
target. It then checks every verification width against that sequence for exact
greedy parity.

```bash
swift run -c release mere.run model benchmark q38-verification \
  --model-root /path/to/flash-next \
  --widths 1,4,8,16,32 \
  --tokens 128 \
  --trials 2 \
  --json
```

The timed region excludes model loading, prompt prefill, and oracle generation.
This command measures an upper bound for accepted target tokens. It excludes
draft cost, branch construction, and tree-shaped recurrent state. A faster wide
pass doesn't qualify a speculative decoder by itself.

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
specific explicit comparison. The installed `vision-chat-q38-27b` and
`vision-chat-q38-27b-4bit` code-generation lanes are available as explicit
targets but are not added to the default comparison. The BF16 checkpoint is
55.59 GB; the 4-bit target plus its MTP and official vision components is
19.49 GB.

```bash
swift run mere.run model benchmark code \
  --allow-code-execution \
  --json

swift run mere.run model benchmark code \
  --models vision-chat-q38-27b-4bit \
  --thinking \
  --max-tokens 3072 \
  --allow-code-execution \
  --json

MERERUN_Q35_MTP_SPECULATION=1 swift run mere.run model benchmark code \
  --models vision-chat-q38-27b-4bit \
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
the larger Ornith GGUF eval target. `--execution-timeout` (default 5 seconds)
and `--python` (default `python3`) control test execution, and the
`--laguna-*` options point at local Laguna target and DFlash checkpoints.

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

Pass `--models text-agent-ornith-35b-mlx-4bit` to exercise the recommended Ornith
Q4 target plus BF16 vision companion against the same fixtures. Use
`vision-chat-ornith-35b` for the optional full-BF16 reference lane.

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
swift run mere.run model capabilities --recommended --json
```

`--all` includes unsupported models with their reasons, `--recommended` shows
only models recommended for first setup, and `--json` prints a
machine-readable report.

### `mere.run model info`

Inspect a canonical model ID or a local model root. The Storage section reports
the layout, resolved payload size, wrapper size when different, and symlink
counts.

```bash
swift run mere.run model info image-zimage-nano
swift run mere.run model info /path/to/model/root --components
swift run mere.run model info text-chat-gemma4 --json
```

`--components` resolves and prints component paths. `--json` prints the stored
manifest, or a catalog-derived manifest for an external binding.

### `mere.run model repair-manifests`

Write missing `mererun_model.json` manifests for known local model roots.

```bash
swift run mere.run model repair-manifests
swift run mere.run model repair-manifests --dry-run
swift run mere.run model repair-manifests --dry-run --json
```

`--json` returns the operation mode, healthy/proposed-or-completed/skipped
counts, and one typed status entry per known model. Combine it with `--dry-run`
for a non-mutating repair preview suitable for Studio and other thin clients.
`--model <id>` repairs only one model, and `--accept-model-license` records
that you accept the model's listed third-party terms when writing a missing
manifest.

### `mere.run api serve`

Start an OpenAI-compatible local API server. The default engine is
`text-chat-q36`, which serves `text-chat-q36-nano`.

```bash
swift run mere.run api serve [options]
```

Key options:

- `--host`: defaults to `127.0.0.1`
- `--port`, `-p`: defaults to `8080`
- `--engine`: serving engine; see the engine values later in this section
- `--model`, `-m` (alias `--model-path`): model id, path, or root for the
  selected engine. For `text-code`, pass a GGUF file.
- `--lora`: default cataloged adapter id or local LoRA path for all requests
- `--api-key`: bearer token; also read from `MERERUN_API_KEY`
- `--rate-limit-per-minute`: global inference request limit; defaults to `60`
- `--max-active-requests`: defaults to `1`
- `--memory-guard off|safe|balanced|aggressive|custom`: defaults to
  `balanced`; `custom` requires `--memory-guard-custom-ceiling-gb`
- `--context-size`: defaults to `32768`
- `--warmup` / `--no-warmup`: warm supported default-model graphs before the
  server reports healthy; on by default
- `--kv-bits`, `--kv-quant-scheme uniform|polar|turboquant`,
  `--kv-group-size`, `--quantized-kv-start`: Gemma4 KV cache quantization
- `--image-run-records`, `--transcription-run-records`: keep durable image
  runs, or file-transcription runs with their uploaded audio, under a root
- `--preflight`, `--json`: print a structured serving plan without starting

Supported endpoint surface:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/videos/generations`
- `POST /v1/vision/geometry`, `POST /v1/vision/geometry/multiview`,
  `POST /v1/vision/image-to-3d`, `POST /v1/vision/image-to-3d-multiview`, and
  `POST /v1/vision/depth-video`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/diarizations` and `POST /v1/audio/diarizations/stream`
- `POST /v1/text/classifications`, `POST /v1/text/extractions`, and
  `POST /v1/text/decisions`
- `GET /runtime/status`
- `POST /runtime/models/{id}/load`
- `POST /runtime/models/{id}/unload`
- `GET/PATCH /runtime/models/{id}/settings`

The `/v1/videos` and `/v1/vision` routes accept loopback clients only. For
each route's purpose, see [HTTP routes](/runtime/api-server#http-routes).

Chat, image, video, speech, transcription, and diarization requests go through
the [model scope check](#model-scope-check): a field the selected model refuses
returns HTTP 400 before anything loads, and one it ignores adds an
`x-mere-warning` response header. For details, see
[Model scope](./runtime/api-server.md#model-scope).

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

Security defaults:

- loopback binds are local-first and do not require auth
- non-loopback binds require `--api-key` or `MERERUN_API_KEY`
- `POST /v1/chat/completions`, `POST /v1/embeddings`,
  `POST /v1/images/generations`, and `POST /v1/audio/speech` require
  `Content-Type: application/json`; `POST /v1/images/edits` and
  `POST /v1/audio/transcriptions` and `POST /v1/audio/diarizations` require
  `multipart/form-data`
- `--rate-limit-per-minute` (default 60) applies basic request throttling to
  the OpenAI-compatible routes
- `--max-active-requests` controls fair FIFO admission for chat, embedding,
  image, TTS, and ASR inference; the default `1` serializes local inference
  across text and media activation peaks while exposing queue depth in status;
  queued client cancellations are removed from the FIFO instead of running
  later. Raising it is an explicit throughput and unified-memory tradeoff.
  Explicit runtime model load/unload maintenance shares the same queue.
  Values above `1` automatically engage supported Gemma4, Qwen-family,
  and LFM2 decode batching unless an engine-specific environment variable forces
  the serial path
- `--warmup` is enabled by default for Gemma 4 Turbo and Qwen3.8-Flash-Next.
  The server evaluates a small representative request before it starts
  listening, so model loading and the first lazy target/draft graph
  materializations are paid before `/health` reports ready.
  Use `--no-warmup` only when startup latency matters more than the first
  request. MLX does not expose compiler time separately, so startup telemetry
  reports `warmupPrefillSeconds` and `warmupDecodeSeconds`, and marks graph
  compilation as included in those warmup measurements instead of presenting
  an invented compile-only duration.
- `--memory-guard` controls runtime memory pressure behavior. Accepted values
  are `off`, `safe`, `balanced` (default), `aggressive`, and `custom`; `custom`
  also requires `--memory-guard-custom-ceiling-gb`.
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
  `MERERUN_LFM2_PREFIX_KV_CACHE=0` for a baseline. It retains exact prompts and
  the stable conversation prefix before the final message instead of cloning
  every intermediate prefill chunk. Checkpoints fork both attention KV and
  short-convolution state.
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
  benchmarks qualify the end-to-end model path.
- API serving keeps Gemma 4 Turbo KV full precision by default because the
  previous token-zero TurboQuant default caused severe long-prompt decode
  latency. Operators can still opt into the memory tradeoff explicitly with
  `--kv-bits 4 --kv-quant-scheme turboquant --quantized-kv-start 0`.
  Per-model runtime settings can set `kvCacheMode` to `affine8` for Gemma4,
  Qwen-family, and LFM2 as a memory control relative to full-precision KV.
  Qwen-family and LFM2 dequantize the generic cache for attention, so this is an
  explicit tradeoff. `default` restores the selected engine/model/server
  default. Gemma4 also accepts `polar2` or `auto`; `auto` keeps the default KV
  path below 1024 prompt tokens and switches to decode-deferred packed PolarKV
  at or above that threshold.
- `/runtime/status` and `mere.run status` aggregate prefix hits, reused tokens,
  batched decode steps, completed chat requests, generated tokens, and average
  load/prefill/decode timings across loaded models under `cacheStats` and
  `benchmarkStats`. Model entries include startup load/warmup timing and the
  last completed request's TTFT and throughput. Admission status includes the
  phase and generated-token updates for active requests plus cancellation and
  slot-release timestamps. Busy model counters are deferred so the status
  route remains responsive during long prefill/decode work.
- non-streaming chat responses expose `x-mere-request-id` immediately, send
  JSON-safe whitespace heartbeats while the final object remains buffered, and
  finish with declared HTTP trailers. Those trailers include standard
  `Server-Timing` entries for load, prefill, KV packing, decode, and TTFT plus
  `x-mere-*` token, throughput, KV-mode, and final runtime-status fields. This
  lets an outbound disconnect cancel native generation without leaking partial
  text or malformed tool calls. Closing either a streaming or non-streaming
  client releases its active-request slot; `/runtime/status` records the
  disconnect receipt, cancellation completion, and slot release separately.
- generation parameters are bounded before execution; for example, `max_tokens` must fit the configured context size, and native chat engines receive that same context cap for prompt truncation
- LoRA adapters for the API server are selected by the operator with `--lora`; request bodies cannot provide local LoRA paths

Engine values:

- `text-chat-q36` (default)
- `text-code`
- `text-chat-klein`
- `text-chat-gemma4`
- `text-chat-diffusiongemma`
- `text-chat-laguna`
- `text-chat-flint`
- `text-chat-kolibri`
- `text-chat-lfm2`
- `text-chat-deepseek-v4-flash`
- `text-chat-muse-glimmer`
- `text-chat-nemotron-h`
- `text-chat-nemotron-omni`

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
- `input` can be a string or array of strings.
- `encoding_format` can be omitted or set to `float`; base64 encoding and
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
  limits. Masks are accepted for client compatibility; native edit
  models use whole-image conditioning.
- `POST /v1/audio/speech` serves `speech-tts-qwen3-nano` and returns WAV by
  default, with `mp3`, `opus`, `aac`, and `flac` available when `ffmpeg` is
  installed. OpenAI model names such as `tts-1` map to the local default;
  normalized input plus voice instructions can total at most 32 KiB of UTF-8.
- `POST /v1/audio/transcriptions` accepts multipart uploads for
  `speech-asr-parakeet` or `speech-asr-qwen3`, with `json`, `text`,
  `verbose_json`, `srt`, and `vtt` response formats. OpenAI model names such as
  `whisper-1` map to the local default; `max_tokens` is limited to 1 through
  4,096.
- `POST /v1/audio/diarizations` accepts one multipart audio upload for
  `speech-diarization-nemotron3` or `speech-diarization-sortformer`, returning
  versioned JSON or RTTM. Nemotron 3 also accepts `latency` settings `offline`,
  `1.04`, `0.64`, and `0.32`.

Examples:

```bash
swift run mere.run api serve
swift run mere.run api serve --preflight --json
swift run mere.run api serve --engine text-chat-gemma4
swift run mere.run api serve --engine text-chat-diffusiongemma --model text-chat-diffusiongemma-26b-optiq-4bit
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
curl http://127.0.0.1:8080/v1/audio/diarizations \
  -F model=speech-diarization-nemotron3 \
  -F file=@meeting.wav
```

After starting a server, run `swift run mere.run status` from another terminal
to confirm `/health`, `/v1/models`, the runtime pool, and the served model.

### `mere.run gate`

Run the end-to-end quality gate against installed models. Each check runs this
executable's real subcommands. Deterministic checks run twice and compare
hashes, and video checks decode the MP4 and its promised audio. Correctness
failures exit nonzero; performance regressions only warn unless you pass
`--strict-perf`. Checks whose models aren't installed are skipped unless you
pass `--require-all`.

On the first run on a machine, record baselines. Rerun with
`--update-baselines` after intentional output-changing changes.

```bash
mere.run gate --update-baselines
mere.run gate
mere.run gate --suite text --json-output ./gate.json --strict-perf
mere.run gate --all-installed --list
```

Key options:

- `--suite`: comma-separated fixed suites: `text`, `speech`, `vision`,
  `image`, `embed`, or `video`; defaults to `all`. With `--all-installed`, use
  managed model category names.
- `--all-installed`: replace the fixed regression set with one fail-closed
  inference check for every installed managed model. `--skip-model` records
  quarantined skips and `--only-model` reruns specific models; both are valid
  only with `--all-installed`.
- `--require-all`: fail instead of skipping when a selected check's model
  isn't installed
- `--update-baselines`: record current outputs and timings as baselines
- `--strict-perf`: treat performance regressions as failures
- `--json-output`: write the full report as JSON
- `--list`: list available checks and exit

For what each suite covers, see [Quality gate](/gate).

### `mere.run config`

Get and set persisted mere.run configuration, such as a Hugging Face token.
The supported keys are `hf-token` and `hf-endpoint`. Commands that print
values mask secrets.

```bash
mere.run config set hf-token --from-env HF_TOKEN
mere.run config get hf-token
mere.run config list
mere.run config unset hf-token
mere.run config path
```

- `config set <key> [<value>]`: store a value, or read it from an
  environment variable with `--from-env <name>`
- `config get <key>`: print one value; `--reveal` shows a secret in full
- `config unset <key>`: remove a value
- `config list`: show all values
- `config path`: print the config file path

For precedence and storage, see [Configuration](/configuration).

### `mere.run open-webui quickstart`

Start a mere.run API server, run the official Open WebUI Docker image, and
connect the two. Docker must be running.

```bash
mere.run open-webui quickstart --dry-run
mere.run open-webui quickstart --pull
```

Key options:

- `--host`, `--port`: mere.run API address; default `0.0.0.0` and `8080`
- `--api-key`: API key for mere.run; defaults to `MERERUN_API_KEY`, or
  `change-me` when unset
- `--engine`: serving engine for the text model; defaults to the managed
  model's API engine
- `--text-model` (default `text-chat-gemma4-12b`), `--vision-model` (default
  `vision-chat-gemma4-12b`), `--embedding-model` (default
  `text-embed-qwen3-0.6b`), `--image-model` (default `image-zimage-nano`),
  `--tts-model` (default `speech-tts-qwen3-nano`), and `--stt-model` (default
  `speech-asr-parakeet`): installed models to expose
- `--tts-format`: speech response format; defaults to `wav`
- `--webui-host`, `--webui-port`: Open WebUI address; default `127.0.0.1` and
  `3000`
- `--container-name`, `--volume-name`, `--image`: Docker settings; the image
  defaults to `ghcr.io/open-webui/open-webui:main`
- `--admin-email`, `--admin-password`: Open WebUI sign-in used while
  configuring a fresh container. The password defaults to
  `MERERUN_OPEN_WEBUI_ADMIN_PASSWORD`, or `admin` when unset.
- `--pull`: pull the configured models first; add `--accept-model-license` for
  restricted models
- `--skip-server`, `--skip-docker`, `--skip-configure`: reuse a running
  mere.run server or Open WebUI instance, or skip admin API configuration
- `--reset`: remove an existing Open WebUI container and volume first
- `--wait-seconds`: health-check timeout; defaults to `180`
- `--dry-run`: print the planned commands without starting anything
- `--quiet`, `-q`: suppress model pull progress

Because the API binds to `0.0.0.0` by default, replace the default API key and
admin password before you use the setup on a shared network. For details, see
[Open WebUI companion](/runtime/api-server#open-webui-companion).

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

Key options:

- `--mode agent|byoa|manual`: skip the interactive choice
- `--agent-model small|tier|premier`: agent model choice; defaults to `tier`
- `--install`: install the selected dependencies and models
- `--start`: start the selected path after installing prerequisites
- `--dry-run`: print the planned steps without installing or starting anything
- `--host`, `--port`: local API address for the Pi-backed agent; default
  `127.0.0.1` and `8080`
- `--pi-path`: Pi executable; defaults to the managed install or `PATH`
- `--quiet`, `-q`: suppress install progress

Agent model choices:

- `small`: `text-agent-ornith-9b`, a tool-capable native OptiQ setup agent for 16 GB machines
- `tier`: the best supported tool-capable local tier for this machine: Ornith 9B, Gemma 4, Qwen3.6 nano on Linux, or DeepSeek V4 Flash on 96 GB+ machines
- `premier`: `text-agent-deepseek-v4-flash`, the preferred managed 96 GB+ setup-agent tier served by the bundled DS4 engine

North Mini Code (`text-code-north-mini`) is available as a managed native GGUF
coding model. It is pullable through `model pull`, can be served with
`api serve --engine text-code --model text-code-north-mini`, and can be used
for direct code sessions and evaluations. The `text-code` API lane rejects tool calls,
so it is not exposed to Pi.

Ornith (`text-agent-ornith-9b`) is available as an experimental native
MLX/OptiQ coding-agent model. It uses the Qwen-family runtime, so serve it with
`api serve --engine text-chat-q36 --model text-agent-ornith-9b`.
The official Ornith 1.5 35B-A3B Q4/Q6/Q8/BF16 MLX family uses the same native
Qwen-family serving engine after an explicit managed pull. Use
`model capabilities` to select its RAM-appropriate id.
The larger Ornith 35B GGUF target (`text-agent-ornith-35b`) is also available
for explicit evaluations and runs through:

```bash
swift run mere.run api serve --engine text-chat-q36 --model text-agent-ornith-35b-mlx-4bit
swift run mere.run api serve --engine text-code --model text-agent-ornith-35b
```

BYOA prints a ready-to-paste Claude/Codex prompt. Manual mode prints the
commands for capabilities, model pulls, serving, and optional Pi installation.
Pi auto-install uses the published macOS release assets; on Linux, put a `pi`
binary on `PATH` or pass `--pi-path` and the agent runs in the active terminal.

### `mere.run agent onboard`

Print a guided setup summary for this machine. `mere.run setup` uses this
lower-level command. Optional flags pull the recommended supported model
package (`--pull-recommended`, with `--accept-model-license` for restricted
models), install Pi (`--install-pi`), and write a Pi provider extension that
points at `mere.run api serve` (`--configure-pi`). `--host` and `--port`
(default `127.0.0.1` and `8080`) set the provider address, and `--model`
picks the exposed model; it defaults to this machine's recommended tier.

```bash
swift run mere.run agent onboard
swift run mere.run agent onboard --pull-recommended --accept-model-license
swift run mere.run agent onboard --install-pi --configure-pi
swift run mere.run agent onboard --configure-pi --model text-agent-deepseek-v4-flash
swift run mere.run agent onboard --configure-pi --model text-agent-ornith-9b --port 8080
```

### `mere.run agent status`

Inspect this machine, the Pi installation, the generated provider extension, the selected
provider endpoint/model, recommended setup tier, and every startable agent
model without mutating the setup.

```bash
swift run mere.run agent status
swift run mere.run agent status --json
swift run mere.run agent status --pi-path /path/to/pi --json
```

The JSON shape is the readiness contract used by the macOS Studio's Server
domain. Dates use ISO 8601 and paths identify the provider configuration and
extension that the CLI uses.

### `mere.run agent install-pi`

Install the most recent published `earendil-works/pi` release asset for this macOS
architecture into the mere.run application-support directory.

```bash
swift run mere.run agent install-pi
```

To re-download and replace the installed release, add `--force`.

### `mere.run agent start`

Start a local API server for a selected tool-capable managed agent model and
launch Pi against the native `mere-run` provider. Qwen-family models use a
native chat engine and DeepSeek V4 Flash uses the DS4-backed
`--engine text-chat-deepseek-v4-flash`. If `--model` is omitted, `agent start`
uses the best installed startable setup agent first, then a valid persisted Pi
provider model, then this machine's startable hardware tier. On 96 GB+
Apple Silicon Macs, DeepSeek V4 Flash is the preferred setup-agent tier; smaller
Qwen models are alternatives, not upgrades. Qwen3.8 advertises its native
`low`, `medium`, and `xhigh` reasoning levels to Pi; Pi's `minimal`, `high`, and
`max` selections map to the nearest native level.

```bash
swift run mere.run model pull text-agent-deepseek-v4-flash
swift run mere.run agent install-pi
swift run mere.run agent start --model text-agent-deepseek-v4-flash
```

By default, `agent start` downloads a missing GGUF model and installs Pi
before it starts. Key options:

- `--model`: managed agent model id to serve through Pi
- `--host`, `--port`: local API address; default `127.0.0.1` and `8080`
- `--skip-server`: use an already-running mere.run API server
- `--inline`: run Pi in the active terminal instead of opening Terminal.app
- `--prompt`: initial prompt for Pi; defaults to a setup-guide prompt
- `--pi-argument`: repeatable argument forwarded to Pi
- `--pi-path`, `--working-directory`: Pi executable and working directory
- `--allow-unsupported`: start even if the hardware catalog marks the model
  unsupported
- `--no-bootstrap`: refuse to download a missing GGUF or install Pi
- `--quiet`, `-q`: suppress bootstrap progress

### Machine-readable receipts and progress

Long-running generation commands share two machine-readable surfaces, so
wrappers and the Studio app don't need to probe the file system or parse prose.

`--receipt` appends one final JSON line to stdout after the command's normal
output. Human output before it is unchanged, so parse the last line and ignore
the rest:

```json
{"event":"result","exit":0,"outputs":[{"kind":"image","path":"/abs/out.png"},{"kind":"json","path":"/abs/out-prompt.json","role":"structured-prompt"}]}
```

- `outputs[0]` is the primary artifact; sidecars follow it with a `role`
  (`structured-prompt`, `timings`, `detections`, `tracking`, `masks`,
  `candidate`, `stem`, `lyrics`, `recipe`, `composition`, `profile`,
  `daw-bundle`).
- `kind` is one of `image`, `video`, `audio`, `text`, `json`, `directory`.
- The receipt is printed only after a successful run, so `exit` is always `0`;
  a failed run exits nonzero without a receipt. `--receipt` is rejected
  together with `--preflight`, which prints a report and produces no result.
- A `warnings` array follows when the [model scope check](#model-scope-check)
  warned about an option; without warnings the key is absent.
- Supported by `image generate`, `video generate`, `music generate`,
  `sfx generate`, `speech synthesize`, `speech transcribe`, `vision depth`,
  `vision ground`, `vision segment`, and `vision track`.

`--progress-json` streams one JSON object per progress event on stderr,
replacing the human-readable progress text and taking precedence over
`--quiet`:

```json
{"event":"progress","stage":"denoising","step":2,"total_steps":4}
```

One convention holds for every lane:

- `step` is 0-based while a stage is in progress: `step: 2` of
  `total_steps: 4` means the third step is running.
- Every determinate stage (`total_steps > 0`) ends with exactly one event
  whose `step == total_steps`. It is written when the stage completes (the
  next stage begins, or the same stage restarts for the next sliding window)
  and, for the last stage, when the pipeline returns, so a wrapper can wait
  for it without hanging at `total_steps - 1`.
- `total_steps: 0` marks an indeterminate stage, for example
  `speech synthesize` token counts; it carries no terminal event, so treat the
  receipt or the exit status as completion.
- Supported by `image generate` (all models), `video generate` (MiniMax-H3
  and Wan lanes; `window` counts H3 sliding windows), `music generate`
  (MiniMax Music 3 and Magenta RT2 lanes), `sfx generate` (Woosh and MMAudio),
  and `speech synthesize`. The native LTX video lanes and the ACE-Step music
  pipeline expose no cheap per-step callback and emit no progress events.

`mere.run catalog --json` declares both flags per capability and carries the
additive option metadata (`default_value`, `group`, `tier`, `range`,
`depends_on`) that shells use to build forms without a second copy of the
command surface.

Each capability's `output` describes what one successful run leaves behind.
`kind` covers the run with no destination named: `text` prints to stdout,
`service` runs until it's stopped, and `file` or `directory` always writes the
artifact. `flag` names the option that carries the destination path, and
`optional` marks an artifact written only when that flag is passed, so
`speech transcribe` reads as a `text` run that additionally saves the
transcript to `--output` on request. `flag` is absent only where the command
picks the path itself: `adapter pull` installs into the managed adapter store,
and `image run-plan` writes wherever the saved plan says.

## Validation and smoke runs

Standard repository validation:

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
