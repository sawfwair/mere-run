# Source Layout Reference

This page is a concise reference for where each major subsystem lives.

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

Linux CLI compatibility should enter through reusable core surfaces here, not
through app bundle code. Media-tool discovery belongs behind typed APIs and
should honor `MERERUN_FFMPEG`, `MERERUN_FFPROBE`, and then `PATH`.

### `AudioCore`

Shared audio types and utilities.

### `AudioCodecs`

Audio codecs and conversion helpers.

Keep Linux audio/video probing fixture-sized: use `ffmpeg` and `ffprobe`
stubs or tiny generated files in tests, not real model checkpoints.

### `AudioSTT`

Speech-to-text backends.

- `Parakeet/`
- `Qwen3ASR/`

### `AudioTTS`

Text-to-speech backends.

- `Qwen3TTS/`
- `TTS/`

### `MereRunCLI`

The public executable target.

- `Commands/`
- `Support/`

This is the product Linux compatibility work should exercise.

### `MereRunApp`

The optional SwiftUI studio target. It stays macOS-only and should not become a
Linux compatibility dependency.

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
- `vendor/mlx-swift_Cmlx.bundle`: macOS MLX shader resources; hosted Linux CI
  should use CPU MLX-sized fixtures, while Linux arm64 package validation needs
  the CUDA path on real arm64 CUDA hardware
