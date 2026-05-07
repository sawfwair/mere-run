# Repository Tour

This page explains the shape of the repo and what each major directory owns.
Use it as the bridge between the end-user docs and the source tree.

## Top-level layout

```text
mere-run/
  Package.swift
  Sources/
  Tests/
  docs/
  scripts/
  vendor/
```

## SwiftPM products

`Package.swift` defines seven public products:

- `MereRunCore`
- `AudioCore`
- `AudioCodecs`
- `AudioSTT`
- `AudioTTS`
- `mere.run` (the executable product backed by the `MereRunCLI` target)
- `mere.run.app` (the optional SwiftUI studio backed by the `MereRunApp` target)

Those map cleanly to the main runtime families exposed by the CLI and the
optional macOS studio.

## Source tree

### `Sources/MereRunCLI`

The public command-line surface.

- `MereRunCLI.swift`: top-level command tree
- `Commands/`: modality-scoped subcommands
- `Support/`: shared CLI bootstrap, model registry, output helpers, and local
  API support

If you want to understand the CLI end to end, start here.

### `Sources/MereRunApp`

The optional macOS studio. It does not own runtime behavior; it turns
user-facing studio requests into `mere.run` arguments, launches the CLI as a
child process, streams stdout and stderr, saves local library metadata, and keeps
the raw command surface available in Advanced details.

### `Sources/MereRunCore`

The shared inference and model-management library.

Key subdirectories:

- `Flux2Klein/`: Klein image-family runtime
- `ZImageTurbo/`: ZImage image-family runtime
- `QwenImageEdit/`: image editing flow
- `Gemma4/`, `Q35/`, `Psi/`, `MeBot/`: text/chat model families
- `Embeddings/`: embedding-generation support
- `LightOnOCR/`: OCR runtime
- `VLM/`: vision-language model helpers
- `ACEStep/`: music generation pipeline
- `LTX/`: video generation pipeline
- `LoRA/`: LoRA loading and application support
- `Support/`: manifests, model resolution, model paths, archive/signing helpers
- `Training/`, `Quantization/`: advanced model-training and quantization
  utilities kept in the package tree

### `Sources/AudioCore`

Audio-oriented shared types and common utilities used by both speech synthesis
and transcription.

### `Sources/AudioCodecs`

Audio conversion and low-level codec support used by speech runtimes and some
tests.

### `Sources/AudioSTT`

Speech-to-text backends.

- `Qwen3ASR/`: native Qwen3 ASR path
- `Parakeet/`: Parakeet transcription path

### `Sources/AudioTTS`

Text-to-speech backends.

- `Qwen3TTS/`: native Qwen3-TTS generator and tokenizer
- `TTS/`: shared TTS support types

## Tests

### `Tests/MereRunCoreTests`

Core runtime, resolver, model, and subsystem tests. This is where most
behavioral validation lives.

### `Tests/MereRunCLITests`

Command parsing and CLI-facing behavior tests. When you change the public
surface or command semantics, this is where you should add or update coverage.

## Scripts

### `scripts/check.sh`

The main repo validation entrypoint. It runs:

- build and unit tests
- CLI help smoke for the public command tree
- output-format and hygiene checks
- optional e2e smoke runs

### `scripts/e2e_smoke.sh`

Sequential real-world smoke tests for installed models. Use this when you want
to validate actual runtime paths instead of just build and parse coverage.

## Vendor artifacts

### `vendor/llama.xcframework`

Vendored `llama.cpp` runtime used by the local code path and API serving.

The repo expects this artifact to live inside `vendor/`; it is intentionally
part of the standalone package layout so the package does not depend on the old
monorepo payload structure.

### `vendor/mlx-swift_Cmlx.bundle`

Vendored MLX shader resources used by the macOS runtime.

### `THIRD_PARTY_NOTICES.md`

Tracks provenance and license notices for vendored artifacts and other bundled
third-party materials that ship with the public repo.

## Docs

The docs are layered intentionally:

- user-facing setup and reference in `docs/`
- runtime-family guides in `docs/runtime/`
- contributor/source-reading material in `docs/internals/`

Start at [mere.run Documentation](/) to navigate by audience.

## Recommended code reading order

1. Read [Architecture Reading Map](./architecture.md)
2. Open `Sources/MereRunCLI/MereRunCLI.swift`
3. Pick one command family in `Sources/MereRunCLI/Commands`
4. Jump to the matching runtime family in `Sources/MereRunCore`,
   `Sources/AudioSTT`, or `Sources/AudioTTS`
5. Use the runtime family docs to follow the load -> prepare -> generate ->
   decode -> output path
