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

## Six commands, six kinds of thing

Install a signed build from [mere.run/releases](https://mere.run/releases), or
`swift build` from a clone. Then:

```bash
mere.run image generate --prompt "a ceramic mug in morning light" --output mug.png
mere.run music generate "arena rock anthem, stacked vocals" --output anthem.wav
mere.run speech synthesize "Hello from mere.run" --output hello.wav
mere.run sfx generate "a ceramic mug shattering on tile" --output smash.wav
mere.run video generate "a drone flight over snowy peaks" --output flight.mp4
mere.run vision image-to-3d-trellis2 ./mug.png --output ./mug-3d
```

Same binary. Same grammar. No inference API key in any of them.

Start with `mere.run model capabilities` — it reads your hardware and tells you
which models this machine can actually run before you spend a gigabyte finding
out.

## The full command surface

All of it ships in that one executable. The table below is generated from the
CLI itself, so it cannot drift from what your copy really does — and every row
links to the page that owns that command.

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
| [`mere.run run`](/workflows#run-directories) | Inspect durable mere.run workflow reports and run directories. |
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

Install, pull a first model, and make something in the next ten minutes.

- [Getting Started](/getting-started)
- [macOS Deep Links](/macos-deep-links) — preview or import local artifacts from launchers, automations, agents, and apps
- [Linux QuickStart](/linux-quickstart)
- [CLI Reference](/cli)
- [Offline Cookbooks](/cookbooks) — `mere.run guide` works without a network
- [Configuration](/configuration)
- [Model Sources](/model-sources)

### Make things

Each family is its own page: what it generates, which model to pull, and the
flags that actually change the output.

- [Image](/runtime/image) — text-to-image, edits, and local LoRA training
- [Text](/runtime/text) — chat, code, embeddings, tool use, PII redaction
- [Speech](/runtime/speech) — synthesis, voice cloning, live transcription
- [Vision and 3D](/runtime/vision) — caption, segment, track, depth, mesh, OCR
- [Music](/runtime/music) — generation, covers, realtime MIDI, transcription
- [Sound Effects](/runtime/sfx) — Foley from text or from a silent video
- [Video](/runtime/video) — clips with synchronized audio, subject animation
- [Persistent Worlds](/runtime/world) — a scene that remembers where you walked

### Put it to work

Serve it, schedule it, measure it, and keep it honest.

- [Portable Workflows, Executors, and Run Artifacts](/workflows)
- [Model and Adapter Management](/runtime/model-management)
- [Raycast Example Integration](/raycast) — one launcher client for the macOS deep-link routes
- [Local API Server and Open WebUI](/runtime/api-server)
- [Official Companion Plugins](/plugins)
- [Quality Gate](/gate) — the check that catches a bad build before you do
- [Benchmarking](/benchmarking)

### Work on the code

- [Repository Tour](/repository-tour)
- [Development Workflow](/development-workflow)
- [Testing Guide](/testing)
- [Architecture Reading Map](/architecture)
- [CLI and Runtime Internals](/internals/cli-and-runtime)
