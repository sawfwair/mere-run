# Source layout reference

Use this reference to find the target, test suite, or operational file that
owns your change.

## Package targets

As of v0.67.1, `Package.swift` exports 28 library products plus the
`mere.run` and `mere.run.app` executables. `scripts/package-policy.json` is the
registry of record for every target and product: it holds the role, owner,
purpose, and allowed dependencies used on this page, and `./scripts/check.sh`
fails on a missing or stale entry. Before you move behavior between targets,
read [package responsibilities](./package-policy.md). The
[repository tour](../repository-tour.md) lists the same targets in product
order.

### Contracts, model management, and relay

- `Sources/MereRunContract/`: portable CLI capability and result contracts
- `Sources/MereRunModelKit/`: portable model identity and storage metadata
- `Sources/MereRunEvaluation/`: versioned external evaluation pack format
- `Sources/MereRunRelayKit/`: portable relay client, workflow transport, and
  shared preflight diagnostics

### Execution, admission, and residency

- `Sources/MereRunExecution/`: durable operation storage and terminal state
- `Sources/MereRunAdmission/`: machine inference admission and reservations
- `Sources/MereRunResidency/`: resource leases and runtime eviction

### Shared computation

- `Sources/MereRunTensor/`: tensor kernels and checkpoint loading
- `Sources/MereRunDecode/`: shared autoregressive sampling and decoding
- `Sources/MereRunKVCache/`: attention cache storage and quantization
- `Sources/MereRunTextEncoder/`: shared text conditioning encoders

### Model runtimes

Text and decision models:

- `Sources/MereRunGemmaModel/`: Gemma text and vision computation
- `Sources/MereRunQwenModel/`: Qwen text and vision computation
- `Sources/MereRunLagunaModel/`: Laguna text computation
- `Sources/MereRunFlintModel/`: Flint attention, routing, and expert
  projections
- `Sources/MereRunKolibriModel/`: Kolibri attention, routing, and expert
  projections
- `Sources/MereRunNemotronBoltModel/`: Nemotron Bolt packed hybrid blocks and
  C-RADIO computation
- `Sources/MereRunGLiNERModel/`: DeBERTa and GLiNER classification
- `Sources/MereRunLayaModel/`: ModernBERT and Laya decisions
- `Sources/MereRunD1Model/`: D1 decision, vision, and audio computation
- `Sources/MereRunCoreMLModel/`: Core ML embedding graphs with native tokenizer
  and media preprocessing

Image, video, and shared audio models:

- `Sources/MereRunImageModels/`: image model computation
- `Sources/MereRunLTXModel/`: LTX transformer and VAE computation
- `Sources/MereRunH3Model/`: H3 transformer and VAE computation
- `Sources/MereRunAudioModels/`: shared vocoder computation

Speech models:

- `Sources/AudioParakeetModel/`: Parakeet recognition
- `Sources/AudioQwen3ASRModel/`: Qwen speech recognition
- `Sources/AudioWhistleModel/`: Whistle recognition
- `Sources/AudioNemotronTranscribeModel/`: Nemotron 3.5 Transcribe recognition
  and translation
- `Sources/AudioSortformer/`: speaker diarization
- `Sources/AudioQwen3TTSModel/`: Qwen speech synthesis
- `Sources/AudioBreezeTTSModel/`: Breeze speech synthesis
- `Sources/AudioDrama3Model/`: DRAMA-3 speech synthesis
- `Sources/AudioPhononModel/`: Phonon speech synthesis

A new model runtime belongs in one of these targets, or in a new target that
owns it, not in `MereRunCore`. Some of these targets also have published
products kept for compatibility; in-repository callers import the owning target
directly. For the boundaries of the larger runtimes, see the
[decode](./decode-runtime-boundaries.md), [Gemma](./gemma-runtime-boundaries.md),
[Qwen](./qwen-runtime-boundaries.md), [H3 and Laguna](./h3-laguna-runtime-boundaries.md),
[image](./image-runtime-boundaries.md), [LTX](./ltx-runtime-boundaries.md), and
[speech](./speech-runtime-boundaries.md) runtime pages.

### Audio and media

- `Sources/AudioCore/`: portable audio-domain operations and contracts
- `Sources/AudioCodecs/`: audio codecs and spectral features
- `Sources/AudioSTT/`: speech recognition orchestration and compatibility
  facade, over `NemotronTranscribe/`, `Parakeet/`, `Qwen3ASR/`, and `Whistle/`
- `Sources/AudioTTS/`: speech synthesis orchestration and compatibility facade,
  over `BreezeTTS/`, `Drama3/`, `Phonon/`, `Qwen3TTS/`, and `TTS/`
- `Sources/MediaIO/`: portable media reading and writing

Keep Linux audio and video probing fixture-sized: use `ffmpeg` and `ffprobe`
stubs or tiny generated files in tests, not real model checkpoints.

### `MereRunCore`

Public runtime orchestration and the compatibility facade over the libraries
listed earlier. Some of its areas:

- `ACEStep/`: music generation
- `CodeGen/`: code-generation support
- `Embeddings/`: embedding support
- `Flux2Klein/`: Klein image family
- `LTX/`: video generation
- `LightOnOCR/`: OCR
- `LoRA/`: LoRA support, including image LoRA training
- `MLX/`: MLX utilities
- `Gemma4/`, `LFM2/`, `MeBot/`, `Psi/`, `Q35/`: text model families
- `QwenImageEdit/`: image editing
- `Support/`: model paths, manifests, resolver, and configuration helpers
- `Training/`: training-specific support
- `VLM/`: vision-language model support
- `ZImageI2L/`, `ZImageTurbo/`: ZImage family support

Implement Linux CLI compatibility through reusable Core surfaces, not app
bundle code. Keep media-tool discovery behind typed APIs, and resolve
`MERERUN_FFMPEG`, then `MERERUN_FFPROBE`, and then `PATH`.

### `MereRunCLI`

The public executable target behind the `mere.run` product: CLI parsing,
transport, and presentation. `Commands/` holds the command types, `Support/`
holds shared helpers, and `Guides/` holds the offline cookbooks that
`mere.run guide` prints. Exercise Linux compatibility of the product through
this target.

### `StudioKit`, `StudioUI`, and `MereRunApp`

The optional macOS Studio has three targets under `apps/macos/`: `StudioKit` is
the SwiftUI-free model layer, `StudioUI` holds the views, and `MereRunApp`
(sources in `apps/macos/MereRunStudio/`) is the `mere.run.app` executable that
composes them. All three are macOS-only and must not become Linux compatibility
dependencies. For Studio prompt tasks, see
[Studio prompt workspace ownership](./studio-prompt-workspace.md).

## Tests

| Location | Use it for |
| --- | --- |
| `Tests/MereRunCoreTests` | Runtime behavior, model resolution, and subsystem integration |
| `Tests/MereRunCLITests` | Command parsing, public CLI contracts, help and command-tree expectations, and API serving integration |
| `Tests/*RuntimeTests` | Isolated numerical fixtures for one model library, such as `GemmaRuntimeTests` or `SpeechRuntimeTests` |
| `Tests/MereRunAdmissionTests`, `Tests/MereRunResidencyTests`, `Tests/MereRunExecutionTests` | Admission, residency, and durable-run libraries without inference dependencies |
| `Tests/AudioCoreTests`, `Tests/AudioTTSTests` | Audio operations and synthesis |
| `Tests/MereRunMLXTestSupport` | Shared MLX fixtures and device setup; test support only |
| `apps/macos/StudioKitTests`, `apps/macos/StudioUITests`, `apps/macos/MereRunStudioTests` | Studio model, view, and app tests |

## Operational files

- `Package.swift`: public package definition
- `scripts/check.sh`: main validation gate
- `scripts/package-policy.json` and `scripts/check-package-policy.sh`: target
  and product registry and its checker
- `scripts/check-model-boundaries.sh`: narrower model and portable-operation
  dependency guard
- `scripts/e2e_smoke.sh`: installed-model smoke runner
- `vendor/llama.xcframework`: vendored native dependency for code and API paths
- `vendor/mlx-swift_Cmlx.bundle`: macOS MLX shader resources. Hosted Linux CI
  uses CPU MLX-sized fixtures. Linux arm64 package validation requires
  the CUDA path on real arm64 CUDA hardware.
