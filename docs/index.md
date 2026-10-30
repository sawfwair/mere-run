---
layout: home

hero:
  name: mere.run
  text: Your machine can make anything.
  tagline: One native CLI for images, text, video, music, speech, vision, 3D, and persistent worlds. Your weights. Your files. Nothing leaves unless you tell it to.
  actions:
    - theme: brand
      text: Install and run
      link: /getting-started
    - theme: alt
      text: Explore capabilities
      link: /runtime/image

features:
  - icon: 01 / CREATE
    title: Make
    details: Generate images, synchronized video and audio, music, speech, sound effects, 3D assets, and persistent worlds from one executable.
    link: /runtime/image
  - icon: 02 / UNDERSTAND
    title: Understand
    details: Caption, ground, segment, track, transcribe, diarize, inspect, and embed media without uploading the source.
    link: /runtime/vision
  - icon: 03 / AUTOMATE
    title: Automate
    details: Build typed workflow graphs with resumable runs, immutable inputs, events, artifacts, and SHA-256 receipts.
    link: /workflows
  - icon: 04 / SERVE
    title: Serve
    details: Expose local chat, embeddings, images, speech, and transcription through OpenAI-compatible endpoints.
    link: /runtime/api-server
---

## Try six commands

Install a signed build from [mere.run releases](https://mere.run/releases), or
build from a clone with `swift build`. Before you download anything, check
which models this machine can run:

```bash
mere.run model capabilities --recommended
```

Each of the following commands writes one file or directory with the command's
default model. Some commands download their default model on first use; for
the others, run `mere.run model pull MODEL_ID` first. Each command's `--help`
names its default.

```bash
mere.run image generate --prompt "a ceramic mug centered on a plain black background" --output mug.png
mere.run music generate "arena rock anthem, stacked vocals" --output anthem.wav
mere.run speech synthesize "Hello from mere.run" --output hello.wav
mere.run sfx generate "a ceramic mug shattering on tile" --output smash.wav
mere.run video generate "a drone flight over snowy peaks" --output flight.mp4
mere.run vision image-to-3d-trellis2 ./mug.png --already-framed --output ./mug-3d
```

TRELLIS.2 expects an object image with a transparent background. The
`--already-framed` option accepts an opaque image instead, if it shows one
centered object on a black background, which is why the image prompt asks for
one.

Every command uses the same binary and argument style. None of them needs an
inference API key.

## The full command surface

The following table is generated from the CLI, and each row links to the page
that documents the command.

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

## Choose a path

### Get it running

Install the CLI, pull a first model, and generate a first result.

- [Getting started](/getting-started)
- [Linux quickstart](/linux-quickstart)
- [CLI reference](/cli)
- [Offline cookbooks](/cookbooks): `mere.run guide` works without a network
- [Configuration](/configuration)
- [Model sources](/model-sources)
- [macOS deep links](/macos-deep-links): preview or import local artifacts from
  launchers, automations, agents, and other apps

### Make and analyze

Each runtime page lists what the family produces, which model to pull, and the
options that change the output.

- [Image generation](/runtime/image): text-to-image, editing, and local LoRA
  training
- [Text](/runtime/text): chat, code, embeddings, tool use, PII redaction,
  classification, extraction, and typed decisions
- [Speech](/runtime/speech): synthesis, voice cloning, transcription, and
  speaker diarization
- [Vision and 3D](/runtime/vision): captioning, segmentation, tracking, depth,
  meshes, and OCR
- [Audio enhancement](/runtime/audio): speech bandwidth extension and audio
  super-resolution
- [Music](/runtime/music): generation, covers, stem separation, realtime MIDI,
  and transcription
- [Sound effects](/runtime/sfx): Foley from text or from a silent video
- [Video](/runtime/video): clips with synchronized audio, and subject animation
- [Persistent worlds](/runtime/world): persistent conditioned-video world
  sessions
- [Geospatial inference](/runtime/geo): flood and fire scores and
  Earth-observation embeddings

These model families also have their own pages:
[Recursive Flint](/runtime/flint), [Nemotron 3.5 Bolt](/runtime/nemotron-bolt),
[Nemotron 3.5 Transcribe](/runtime/nemotron-transcribe),
[Whistle](/runtime/whistle), [Kolibri-1](/runtime/kolibri),
[GLiNER2.5 Decide](/runtime/gliner25-decide), [Laya](/runtime/laya),
[Clef](/runtime/clef), [D1](/runtime/d1),
[PPLX Embed v2](/runtime/pplx-embed-v2), and
[EmbeddingGemma 2](/runtime/embeddinggemma2).

### Put it to work

- [Business use cases](/business/): start with a support ticket or a shared
  archive
- [Portable workflows, executors, and run artifacts](/workflows)
- [Model and adapter management](/runtime/model-management)
- [Local API server and Open WebUI](/runtime/api-server)
- [Official companion plugins](/plugins)
- [Raycast example integration](/raycast): a launcher client for the macOS
  deep-link routes
- [Quality gate](/gate): end-to-end checks against installed models
- [Benchmarking](/benchmarking)

### Work on the code

- [Repository tour](/repository-tour)
- [Development workflow](/development-workflow)
- [Testing guide](/testing)
- [Documentation style](/documentation-style)
- [Documentation style audit](/documentation-style-audit)
- [Architecture reading map](/architecture)
- [CLI and runtime internals](/internals/cli-and-runtime)
