# Repository tour

This page maps the repository's directories and SwiftPM targets to what each
one owns. Use it to get from the end-user docs to the source tree. For the
runtime reading order, see the [architecture reading map](./architecture.md).

## Top-level layout

```text
mere-run/
  Package.swift         SwiftPM manifest: products, targets, and dependency pins
  Sources/              library, runtime, and CLI targets
  Tests/                SwiftPM test targets
  apps/
    macos/              optional macOS Studio (StudioKit, StudioUI, MereRunStudio)
    ios/                XcodeGen-managed iOS Studio app
  docs/                 VitePress documentation site
  scripts/              validation, packaging, conversion, and qualification scripts
  vendor/               vendored native runtimes and the MLX shader bundle
  assets/               DMG packaging artwork
  integrations/         agent integrations, such as Pi
  skills/               agent skills; the macOS app bundles use-mere-run
```

## SwiftPM products

`Package.swift` defines these public library products:

| Area | Products |
| --- | --- |
| Contracts and clients | `MereRunContract`, `MereRunEvaluation`, `MereRunRelayKit` |
| Model identity, storage, and scheduling | `MereRunModelKit`, `MereRunExecution`, `MereRunAdmission`, `MereRunResidency` |
| Shared runtime | `MereRunTensor`, `MereRunDecode`, `MereRunKVCache`, `MereRunTextEncoder` |
| Model runtimes | `MereRunQwenModel`, `MereRunGemmaModel`, `MereRunLTXModel`, `MereRunH3Model`, `MereRunLagunaModel`, `MereRunImageModels`, `MereRunAudioModels` |
| Speech runtimes | `AudioQwen3ASRModel`, `AudioQwen3TTSModel`, `AudioParakeetModel`, `AudioSortformer` |
| Orchestration | `MereRunCore`, `AudioCore`, `AudioCodecs`, `AudioSTT`, `AudioTTS` |
| Media | `MediaIO` (when its source target is present) |

It also defines two executable products:

- `mere.run`, backed by the `MereRunCLI` target.
- `mere.run.app`, the optional macOS Studio, backed by the `MereRunApp` target.
  The Linux package graph omits it.

Many runtime targets are internal: they build as part of the package but
aren't products. They include `MereRunCoreMLModel`, `MereRunD1Model`,
`MereRunFlintModel`, `MereRunGLiNERModel`, `MereRunKolibriModel`,
`MereRunLayaModel`, `MereRunNemotronBoltModel`, `AudioBreezeTTSModel`,
`AudioDrama3Model`, `AudioNemotronTranscribeModel`, `AudioPhononModel`, and
`AudioWhistleModel`. Platform adapters (`llama`, `magentart`, and
`COnnxRuntimeCoreML`), the `MediaIOSmoke` executable, and the
`MereRunMLXTestSupport` test fixtures are internal too.

Every target and product is registered in `scripts/package-policy.json`, which
records its role, supported package variants, and allowed dependencies.
`./scripts/check.sh` fails on an unregistered target. See
[Package policy](./internals/package-policy.md).

The Linux compatibility boundary is the headless `mere.run` CLI and reusable
library code. `mere.run.app`, SwiftUI views, app bundling, installer behavior,
and DMG packaging are macOS-only. Linux release artifacts are headless CLI
tarballs and Debian packages, not app bundles. Build and validate them on the
Linux host class they target.

## Apps

The [`apps/` directory](https://github.com/sawfwair/mere-run/tree/main/apps)
holds the two Apple clients. Shared runtime and relay behavior stays in SwiftPM
library targets under `Sources/`, so neither app becomes a second backend.

### `apps/macos`

The optional macOS Studio is split into three parts:

- `StudioKit`: a SwiftUI-free library for command, job, library, and layout
  state.
- `StudioUI`: the SwiftUI views and theme.
- `MereRunStudio`: the executable that owns scenes, menus, and Sparkle wiring.

Studio doesn't own runtime behavior. It turns Studio requests into `mere.run`
arguments, launches the CLI as a child process, streams stdout and stderr, and
saves local library metadata. The Command view and the Command Console expose
the raw command surface, both rendered from the shared capability contract.

Keep these targets out of Linux compatibility work. Linux contributors validate
the CLI and local API surfaces directly.

### `apps/ios`

The XcodeGen-managed iOS Studio app, widget extension, unit and UI tests,
entitlements, and simulator release gate. It links `MereRunRelayKit` for relay
workflows and `MereRunCore` for on-device inference. See
[iOS Studio](./ios-studio.md). Maintainer-only provisioning, archive export, and
App Store Connect upload automation live outside this repository.

## CLI

### `Sources/MereRunCLI`

The public command-line surface. To understand the CLI end to end, start here.

- `MereRunCLI.swift`: the top-level command tree.
- `Commands/`: modality-scoped subcommands.
- `Support/`: shared CLI bootstrap, model inventory, output helpers, the local
  API server, and workflow execution.
- `Guides/`: the offline cookbooks that `mere.run guide` prints.
- `BenchmarkSuites/`: suite definitions for `model benchmark fused`.

## Contracts and clients

- `Sources/MereRunContract`: portable CLI capability and result contracts.
  Studio and the docs render command forms from these definitions.
- `Sources/MereRunEvaluation`: the versioned external evaluation-pack format,
  validation, and content hashing.
- `Sources/MereRunRelayKit`: the portable relay client, executor profiles and
  authentication, and workflow wire types shared by the CLI and both apps.

## Model identity, storage, and scheduling

- `Sources/MereRunModelKit`: model identities, manifests, configured paths,
  registered locations, artifact pins, and installed lookup. It depends on
  Foundation and Crypto. Callers supply catalog descriptors and runtime
  validation for installed lookup.
- `Sources/MereRunExecution`: durable operation storage, run-directory leases,
  and terminal states.
- `Sources/MereRunAdmission`: machine reservations, request queues,
  memory-pressure policy, and admission leases, without inference, HTTP, or
  parser dependencies. See
  [Inference admission](./internals/inference-admission.md).
- `Sources/MereRunResidency`: resource leases and runtime eviction. See
  [Runtime residency](./internals/runtime-residency.md).

## Shared runtime libraries

- `Sources/MereRunTensor`: checkpoint loading and shared tensor kernels.
- `Sources/MereRunDecode`: shared autoregressive token sampling, streaming, and
  logprob diagnostics. Core keeps model loading, tokenizer adapters, resource
  cleanup, and family orchestration. See
  [Shared decode boundaries](./internals/decode-runtime-boundaries.md).
- `Sources/MereRunKVCache`: attention cache storage and optional quantization,
  shared by Core and the speech libraries.
- `Sources/MereRunTextEncoder`: the shared Qwen text encoder and vision tower.

## Model runtime libraries

Each runtime library owns model configuration and tensor computation. Core
keeps checkpoint loading, prompts, request handling, and output for the same
family.

| Target | Owns | Boundary guide |
| --- | --- | --- |
| `MereRunQwenModel` | Qwen dense and hybrid attention, expert routing, vision layers, and MTP draft state | [Qwen runtime boundaries](./internals/qwen-runtime-boundaries.md) |
| `MereRunGemmaModel` | Gemma text and vision layers, expert routing, attention caches, quantization, and MTP drafts | [Gemma runtime boundaries](./internals/gemma-runtime-boundaries.md) |
| `MereRunLTXModel` | LTX transformer, video and audio VAEs, diffusion decoder, upsampling, and model caches | [LTX runtime boundaries](./internals/ltx-runtime-boundaries.md) |
| `MereRunH3Model` | H3 transformer and VAE computation | [H3 and Laguna runtime boundaries](./internals/h3-laguna-runtime-boundaries.md) |
| `MereRunLagunaModel` | Laguna target and draft computation | [H3 and Laguna runtime boundaries](./internals/h3-laguna-runtime-boundaries.md) |
| `MereRunAudioModels` | BigVGAN vocoder layers shared by H3 and MMAudio | [H3 and Laguna runtime boundaries](./internals/h3-laguna-runtime-boundaries.md) |
| `MereRunImageModels` | FLUX.2, FLUX 3, and ZImage transformer layers and the shared VAE | [Image runtime boundaries](./internals/image-runtime-boundaries.md) |
| `MereRunKolibriModel` | Kolibri attention, sigmoid expert routing, stacked projections, and decoder arithmetic | [Native Kolibri-1](./runtime/kolibri.md) |
| `MereRunFlintModel` | Recursive Flint attention, sigmoid routing, and expert projections | [Run Recursive Flint](./runtime/flint.md) |
| `MereRunUnlimitedOCRModel` | Unlimited-OCR-2 SAM/CLIP vision, DeepSeek MoE, and page-aware DASH attention | [Unlimited-OCR-2](./runtime/unlimited-ocr-2.md) |
| `MereRunNemotronBoltModel` | Nemotron Bolt packed IQ2, Q8, and NVFP4 weights, hybrid blocks, and C-RADIO vision | [Nemotron Bolt](./runtime/nemotron-bolt.md) |
| `MereRunLayaModel` | Native ModernBERT and Laya decision heads | [Laya decisions](./runtime/laya.md) |
| `MereRunGLiNERModel` | Native DeBERTa and the GLiNER2.5 classification, count, and span heads | [GLiNER2.5 Decide](./runtime/gliner25-decide.md) |
| `MereRunD1Model` | LiquidAI D1 causal and bidirectional trunks, SigLIP2 NaFlex vision, option heads, and FastConformer audio | [Native D1 decisions](./runtime/d1.md) |
| `MereRunCoreMLModel` | EmbeddingGemma 2 Core ML graph execution, BF16 token lookup, and tokenizer and media preprocessing | [EmbeddingGemma 2](./runtime/embeddinggemma2.md#apple-core-ml-export) |

`MereRunCoreMLModel` runs on Apple Silicon macOS 15 or later. Core owns its
installation and ordered input routing.

## `Sources/MereRunCore`

The shared inference library: catalog assembly, runtime-dependent manifest
templates, downloads, family validation, and runtime orchestration. It
re-exports ModelKit types and delegates installed lookup through the
`ModelResolver` API. New model runtimes belong in an owned runtime target, not
in Core.

Each model family has its own subdirectory, usually with a `README.md`. The
families include:

- Image generation and editing: `Flux1/`, `Flux2Klein/`, `Flux3/`,
  `ZImageTurbo/`, `ZImageI2L/`, `HiDreamO1/`, `Ideogram4/`, `Krea2/`,
  `QwenImage21/`, `QwenImageEdit/`, and `SenseNovaU15/`.
- Text and vision-language chat: `Gemma4/`, `Q35/`, `LFM2/`, `Psi/`,
  `Inkling/`, `Laguna/`, `Kolibri/`, `Flint/`, `NemotronH/`, `NemotronBolt/`,
  `NemotronOmni/`, `MuseGlimmer/`, `DiffusionGemma/`, `DeepseekV4Flash/`, and
  `MeBot/`.
- Code generation over the vendored llama.cpp runtime: `CodeGen/`.
- Embeddings, decisions, and text analysis: `Embeddings/`,
  `MultimodalEmbeddings/`, `Laya/`, `GLiNER/`, `D1/`, `Clef/`, and
  `PrivacyFilter/`.
- Vision: `LightOnOCR/`, `VLM/`, `SAM3/`, `FalconPerception/`,
  `FaceAnalysis/`, `Pose/`, `OpticalFlow/`, `DepthAnything3/`, `VideoDepth/`,
  `MarigoldV2/`, `VisionGeometry/`, and `Geometry/`.
- 3D reconstruction: `Asset3D/` (the shared mesh contract), `TripoSR/`,
  `InstantMesh/`, and `Trellis2/`.
- Music and audio: `ACEStep/`, `YuE2/`, `MiniMaxMusic3/`, `MagentaRT2/`,
  `MuScriptor/`, `RoFormer/`, `APBWE/`, `UniverSR/`, `Woosh/`, `MMAudio/`, and
  `Drama3/`.
- Video and world simulation: `LTX/`, `MiniMaxH3/`, `Wan2/`, `SCAIL2/`, and
  `Cosmos3/`.
- Earth observation: `TESSERA/`, `OlmoEarth/`, `TerraMindFire/`, and
  `TerraMindFlood/`.
- Shared infrastructure: `Support/` (model sources, Hub snapshots, and model
  resolution), `LoRA/`, `Training/`, `Quantization/`, `MLX/`, `ONNX/`,
  `PyTorchCheckpoint/`, `Decode/`, and `Resources/`.

## Audio and speech

### `Sources/AudioCore`

Portable audio-domain operations and contracts shared by synthesis and
transcription. `SpeechTranscriptionOperation.swift` owns file transcription
plans, validation, events, and outcomes.

### `Sources/AudioCodecs`

Audio codecs and spectral features used by speech runtimes and tests. Linux
media support uses `ffmpeg` and `ffprobe` discovery, including the
`MERERUN_FFMPEG` and `MERERUN_FFPROBE` overrides, instead of Apple media
frameworks.

### `Sources/MediaIO`

Portable media reading and writing. `Sources/MediaIOSmoke` is a test
executable that the Linux gate runs against `ffmpeg` and `ffprobe`; it isn't a
shipped product.

### `Sources/AudioSTT`

Speech recognition orchestration and compatibility exports.
`SpeechTranscriptionResolver.swift` resolves native backend and model inputs
for CLI and API adapters. Each backend has its own subdirectory:
`Qwen3ASR/`, `Parakeet/`, `Whistle/`, and `NemotronTranscribe/`.

### `Sources/AudioTTS`

Speech synthesis orchestration, audio-input adapters, and compatibility
exports:

- `Qwen3TTS/`: native Qwen TTS loading, prompts, token generation, and audio
  output.
- `BreezeTTS/`: native Breeze TTS 2 loading, reference audio, and streaming.
- `Drama3/`: native DRAMA-3 loading, reference-code prompts, and causal audio
  streaming.
- `Phonon/`: adapts native Phonon 31m to shared synthesis, streaming, model
  resolution, and WAV publication.
- `TTS/`: shared TTS support types.

### Speech model libraries

These libraries own configuration and tensor computation. They build without
`MereRunCore`, audio codecs, or model downloads. See
[Speech runtime boundaries](./internals/speech-runtime-boundaries.md).

| Target | Owns |
| --- | --- |
| `AudioQwen3ASRModel` | Qwen speech recognition, using `MereRunKVCache` attention caches |
| `AudioQwen3TTSModel` | Qwen speech synthesis and the speech tokenizer |
| `AudioBreezeTTSModel` | Breeze TTS 2 synthesis |
| `AudioDrama3Model` | DRAMA-3 prompts, codec, audio tower, FP8 ingestion, and packed inference |
| `AudioPhononModel` | Phonon flow sampling and streaming Mimi synthesis |
| `AudioParakeetModel` | Parakeet transcription |
| `AudioWhistleModel` | Whistle encoder, cached decoder, checkpoint geometry, and vocabulary. See [Native Whistle](./runtime/whistle.md). |
| `AudioNemotronTranscribeModel` | Nemotron 3.5 Transcribe audio encoders, speaker fusion, and hybrid Mamba, attention, and MoE decoder. See [Nemotron Transcribe](./runtime/nemotron-transcribe.md). |
| `AudioSortformer` | Sortformer speaker diarization, using `MereRunModelKit` for directory traversal |

`AudioNemotronTranscribeModel` depends on `AudioSortformer` for its
speaker-aware branch. `AudioSTT/NemotronTranscribe` owns file decoding,
tokenization, installed-model resolution, and transcription results.

## Tests

SwiftPM test targets live under `Tests/`:

- `MereRunCoreTests`: Core runtime, resolver, model, and subsystem tests. Most
  behavioral validation lives here.
- `MereRunCLITests`: command parsing, CLI behavior, serving, and the
  documentation contract. When you change the public surface or command
  semantics, add or update coverage here.
- Library tests: `MereRunContractTests`, `MereRunEvaluationTests`,
  `MereRunRelayKitTests`, `MereRunModelKitTests`, `MereRunExecutionTests`,
  `MereRunAdmissionTests`, `MereRunResidencyTests`, `AudioCoreTests`, and
  `AudioTTSTests`.
- Runtime numerical tests: `DecodeRuntimeTests`, `QwenRuntimeTests`,
  `GemmaRuntimeTests`, `LTXRuntimeTests`, `H3RuntimeTests`,
  `LagunaRuntimeTests`, `ImageRuntimeTests`, `SpeechRuntimeTests`,
  `KolibriRuntimeTests`, `FlintRuntimeTests`, `NemotronBoltRuntimeTests`,
  `NemotronTranscribeRuntimeTests`, `Drama3RuntimeTests`, and
  `AudioPhononModelTests`.
- `MereRunMLXTestSupport`: shared MLX fixtures and device preparation.

The macOS Studio tests (`StudioKitTests`, `StudioUITests`, and
`MereRunAppTests`) live under `apps/macos/`. The iOS tests live under
`apps/ios/`. For what to run when, see the [testing guide](./testing.md).

## Scripts

### `scripts/check.sh`

The main local gate. On macOS it runs SwiftLint, the repository hygiene and
policy checks, `swift build`, the MLX shader verification, `swift test`, CLI
help smoke for the public command tree, and `model list` and `status` output
checks. Set `MERERUN_RUN_E2E` to add end-to-end smoke runs. On Linux it runs
`scripts/check-linux.sh` instead.

### `scripts/e2e_smoke.sh`

Sequential real-world smoke tests for installed models. Use it to validate
runtime paths beyond build and parse coverage.

### `scripts/package-linux.sh`

Builds Linux release artifacts for the headless CLI:

- `dist/linux/mere-run-<version>-linux-<arch>.tar.gz`
- `dist/linux/mere-run_<version>_<deb-arch>.deb`
- `dist/linux/SHA256SUMS`

With `--artifact-suffix cuda`, artifact names carry the suffix, such as
`mere-run-<version>-linux-x86_64-cuda.tar.gz`. Run the script on the Linux
host class you intend to validate. CUDA artifacts need matching CUDA hardware
for meaningful runtime smoke coverage.

## Vendor artifacts

Treat `vendor/` as an external artifact boundary. If you change anything in it,
update [`THIRD_PARTY_NOTICES.md`](https://github.com/sawfwair/mere-run/blob/main/THIRD_PARTY_NOTICES.md),
which records provenance and license notices for vendored and bundled
third-party material.

- `vendor/llama.xcframework`: the vendored llama.cpp runtime used by
  `text code` and API serving on macOS.
- `vendor/mlx-swift_Cmlx.bundle`: the MLX Metal shader library used by the
  macOS runtime, with a provenance stamp that the build verifies. See
  [mlx-swift fork policy](./mlx-swift-fork.md).
- `vendor/magentart.xcframework`: the optional Magenta RealTime 2 runtime for
  realtime music on macOS.
- `vendor/ds4`: prebuilt DwarfStar binaries that serve DeepSeek V4 Flash.

Hosted Linux CI stays CPU MLX-oriented and fixture-sized. CUDA runtime smokes
run on real CUDA hosts, not in the pull-request gate.

## Docs

The documentation has three layers:

- User-facing setup and reference in `docs/`.
- Runtime-family guides in `docs/runtime/`.
- Contributor and source-reading material in `docs/internals/`.

Start at the [mere.run documentation home](/) to navigate by audience.

## Recommended code reading order

1. Read the [architecture reading map](./architecture.md).
2. Open `Sources/MereRunCLI/MereRunCLI.swift`.
3. Pick one command family in `Sources/MereRunCLI/Commands`.
4. Jump to the matching runtime family in `Sources/MereRunCore`,
   `Sources/AudioSTT`, or `Sources/AudioTTS`.
5. Use the runtime family documentation to follow the load, prepare, generate,
   decode, and output path.
