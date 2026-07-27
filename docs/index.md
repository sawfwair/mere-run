---
layout: home

hero:
  name: mere.run
  text: Local inference from one machine to a portable fleet
  tagline: A public Swift package and CLI for multimodal inference, persistent worlds, typed workflow graphs, local APIs, an optional macOS studio, and headless Linux executors.
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: CLI Reference
      link: /cli
    - theme: alt
      text: Portable Workflows
      link: /workflows
    - theme: alt
      text: Linux QuickStart
      link: /linux-quickstart

features:
  - title: Multimodal by design
    details: Run image, text, speech, vision, 3D, music, sound-effect, video, and persistent-world workloads from one typed CLI.
  - title: Portable workflow graphs
    details: Validate once, materialize immutable job bundles, and run the same graph locally, over SSH, or through a relay executor.
  - title: Live and resident runtimes
    details: Stream microphone transcription, keep text and video models warm, and preserve causal world state across requests.
  - title: Managed models and adapters
    details: Inspect machine capabilities, pull pinned model snapshots and verified LoRAs, configure runtimes, and run deterministic quality gates.
  - title: Local APIs and companions
    details: Serve OpenAI-compatible chat, embeddings, images, speech, and transcription; connect Open WebUI or install official companion plugins.
  - title: Mac-first, headless when needed
    details: Use the native macOS studio on Apple Silicon or install CLI-only Linux packages for compatible CPU and CUDA hosts.
---

## Current public command surface

This inventory is generated from the CLI configuration. Each command links to
the page that owns its public documentation.

<!-- BEGIN GENERATED: CLI TOP LEVEL -->
| Command | Purpose |
| --- | --- |
| [`mere.run guide`](/cookbooks) | Read offline mere.run command cookbooks. |
| [`mere.run catalog`](/cli) | Inspect the machine-readable command capability contract. |
| [`mere.run image`](/runtime/image) | Generate and validate image models. |
| [`mere.run text`](/runtime/text) | Run local chat, code, embedding, and anonymization workflows. |
| [`mere.run speech`](/runtime/speech) | Synthesize, transcribe, and manage voice profiles. |
| [`mere.run vision`](/runtime/vision) | Caption, inspect, face-analyze, segment, track, pose, depth, geometry, optical flow, and OCR visual media. |
| [`mere.run music`](/runtime/music) | Generate music locally. |
| [`mere.run sfx`](/runtime/sfx) | Generate sound effects locally. |
| [`mere.run video`](/runtime/video) | Generate and understand video with native Swift/MLX pipelines. |
| [`mere.run world`](/runtime/world) | Run persistent local conditioned-video world sessions. |
| [`mere.run graph`](/workflows) | Validate, materialize, run, and submit portable workflow graphs. |
| [`mere.run executor`](/workflows#executor-profiles) | Manage local, SSH, and relay workflow executors. |
| [`mere.run run`](/workflows#run-directories) | Inspect durable mere.run workflow reports and run directories. |
| [`mere.run model`](/runtime/model-management) | List, pull, remove, inspect, and clean up models. |
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

### Install and run locally

- [Getting Started](/getting-started)
- [Linux QuickStart](/linux-quickstart)
- [CLI Reference](/cli)
- [Offline Cookbooks](/cookbooks)
- [Configuration](/configuration)
- [Model Sources](/model-sources)

### Automate and operate

- [Portable Workflows, Executors, and Run Artifacts](/workflows)
- [Model and Adapter Management](/runtime/model-management)
- [Local API Server and Open WebUI](/runtime/api-server)
- [Official Companion Plugins](/plugins)
- [Quality Gate](/gate)
- [Benchmarking](/benchmarking)

### Explore runtime families

- [Image](/runtime/image)
- [Text](/runtime/text)
- [Speech](/runtime/speech)
- [Vision and 3D](/runtime/vision)
- [Music](/runtime/music)
- [Sound Effects](/runtime/sfx)
- [Video](/runtime/video)
- [Persistent Worlds](/runtime/world)

### Contribute to the repo

- [Repository Tour](/repository-tour)
- [Development Workflow](/development-workflow)
- [Testing Guide](/testing)
- [Architecture Reading Map](/architecture)
- [CLI and Runtime Internals](/internals/cli-and-runtime)
