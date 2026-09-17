# Source layout reference

Use this reference to find the public target, test suite, or operational file
that owns your change.

## Public package targets

`Package.swift` exports 28 library products plus the `mere.run` and
`mere.run.app` executables. `scripts/package-policy.json` is the registry of
record for every target and product: it holds the role, owner, purpose, and
allowed dependencies used below, and `./scripts/check.sh` fails on an entry that
is missing or stale. See [package responsibilities](./package-policy.md) before
you move behavior between targets, and the
[repository tour](../repository-tour.md) for the same list in product order.

### Contracts, model management, and relay

- `Sources/MereRunContract/`: portable CLI capability and result contracts
- `Sources/MereRunModelKit/`: portable model identity and storage metadata
- `Sources/MereRunEvaluation/`: versioned external evaluation pack format
- `Sources/MereRunRelayKit/`: portable relay client and workflow transport

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

- `Sources/MereRunGemmaModel/`: Gemma text and vision computation
- `Sources/MereRunQwenModel/`: Qwen text and vision computation
- `Sources/MereRunLagunaModel/`: Laguna text computation
- `Sources/MereRunLTXModel/`: LTX transformer and VAE computation
- `Sources/MereRunH3Model/`: H3 transformer and VAE computation
- `Sources/MereRunImageModels/`: image model computation
- `Sources/MereRunAudioModels/`: shared vocoder computation
- `Sources/AudioParakeetModel/`: Parakeet model computation
- `Sources/AudioQwen3ASRModel/`: Qwen speech recognition computation
- `Sources/AudioQwen3TTSModel/`: Qwen speech synthesis computation
- `Sources/AudioSortformer/`: speaker diarization model computation

A new model runtime belongs in one of these targets, or in a new target that
owns it, not in `MereRunCore`. Their products are published imports kept for
compatibility; in-repo callers import the owning target directly.

### Audio and media

- `Sources/AudioCore/`: portable audio-domain operations and contracts
- `Sources/AudioCodecs/`: audio codecs and spectral features
- `Sources/AudioSTT/`: speech recognition orchestration and compatibility
  facade, over `Parakeet/` and `Qwen3ASR/`; diarization moved to
  `AudioSortformer`
- `Sources/AudioTTS/`: speech synthesis orchestration and compatibility facade,
  over `Qwen3TTS/` and `TTS/`
- `Sources/MediaIO/`: portable media reading and writing

Keep Linux audio/video probing fixture-sized: use `ffmpeg` and `ffprobe`
stubs or tiny generated files in tests, not real model checkpoints.

### `MereRunCore`

Public runtime orchestration and the compatibility facade over the libraries
above. Some of its areas:

- `ACEStep/`: music generation
- `CodeGen/`: code-generation support
- `Embeddings/`: embedding support
- `Flux2Klein/`: Klein image family
- `LTX/`: video generation
- `LightOnOCR/`: OCR
- `LoRA/`: LoRA support
- `MLX/`: MLX utilities
- `Gemma4/`, `LFM2/`, `MeBot/`, `Psi/`, `Q35/`: text model families
- `QwenImageEdit/`: image editing
- `Support/`: model paths, manifests, resolver, config helpers
- `Training/`: training-specific support retained in the package
- `VLM/`: vision-language model support
- `ZImageI2L/`, `ZImageTurbo/`: image-family support

Implement Linux CLI compatibility through reusable core surfaces, not app
bundle code. Keep media-tool discovery behind typed APIs, and resolve
`MERERUN_FFMPEG`, then `MERERUN_FFPROBE`, and then `PATH`.

### `MereRunCLI`

The public executable target behind the `mere.run` product: CLI parsing,
transport, and presentation.

- `Commands/`
- `Support/`

Exercise product Linux compatibility through this target.

### `StudioKit`, `StudioUI`, `MereRunApp`

The optional Studio, in three targets under `apps/macos/`: `StudioKit` is the
SwiftUI-free model layer, `StudioUI` holds every view, and `MereRunApp` is the
`mere.run.app` executable that composes them. All three stay macOS-only and must
not become Linux compatibility dependencies.

## Tests

### `Tests/MereRunCoreTests`

Use this for:

- runtime behavior
- model resolution
- subsystem-level integration

### `Tests/MereRunCLITests`

Use this for:

- command parsing
- public CLI UX contracts
- help and command-tree expectations

## Operational files

- `Package.swift`: public package definition
- `scripts/check.sh`: main validation gate
- `scripts/e2e_smoke.sh`: installed-model smoke runner
- `vendor/llama.xcframework`: vendored native dependency for code and API paths
- `vendor/mlx-swift_Cmlx.bundle`: macOS MLX shader resources. Hosted Linux CI
  uses CPU MLX-sized fixtures, while Linux arm64 package validation requires
  the CUDA path on real arm64 CUDA hardware
