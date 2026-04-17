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
- `Gemma4/`, `MeBot/`, `Psi/`, `Q35/`: text model families
- `QwenImageEdit/`: image editing
- `SHARP/`: SHARP-related support
- `Support/`: model paths, manifests, resolver, config helpers
- `Training/`: training-specific support retained in the package
- `VLM/`: vision-language model support
- `ZImageI2L/`, `ZImageTurbo/`: image-family support

### `AudioCore`

Shared audio types and utilities.

### `AudioCodecs`

Audio codecs and conversion helpers.

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
- `scripts/migrate_model_store.sh`: one-time model-store migration
- `vendor/llama.xcframework`: vendored native dependency for code and API paths
