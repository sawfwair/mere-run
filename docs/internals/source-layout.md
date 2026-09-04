# Source layout reference

Use this reference to find the public target, test suite, or operational file
that owns your change.

## Public package targets

### `MereRunCore`

Shared inference and model-management code.

Main areas:

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

### `AudioCore`

Shared audio types and utilities.

### `AudioCodecs`

Audio codecs and conversion helpers.

Keep Linux audio/video probing fixture-sized: use `ffmpeg` and `ffprobe`
stubs or tiny generated files in tests, not real model checkpoints.

### `AudioSTT`

Speech-to-text and speaker-diarization backends.

- `Parakeet/`
- `Qwen3ASR/`
- `Sortformer/`

### `AudioTTS`

Text-to-speech backends.

- `Qwen3TTS/`
- `TTS/`

### `MereRunCLI`

The public executable target.

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
