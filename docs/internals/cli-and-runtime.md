# CLI and Runtime Internals

This page explains how the public `mere.run` command tree maps to the package
internals.

## Top-level flow

1. `Sources/MereRunCLI/MereRunCLI.swift` defines the public modality-first tree
2. each command in `Sources/MereRunCLI/Commands` parses arguments and performs
   small amounts of orchestration
3. shared CLI helpers in `Sources/MereRunCLI/Support` bootstrap the model store,
   expose the managed model registry, and provide output helpers
4. runtime code in `MereRunCore`, `AudioSTT`, and `AudioTTS` does the actual work

The package is intentionally organized so public command concerns do not live in
the deep model code.

## Model resolution path

Almost every command eventually passes through the same model-resolution layer:

- `MereRunModelPaths.swift`
- `MereRunModelManifest.swift`
- `ModelResolver.swift`

That shared path is what keeps the command surface coherent across image, text,
speech, vision, music, and video.

## CLI design conventions

The public surface follows three rules:

### 1. Modality-first commands

Commands are grouped by user intent:

- image
- text
- speech
- vision
- music
- video
- model
- status
- api

### 2. Canonical public model IDs

The runtime uses explicit public IDs such as:

- `image-klein-max`
- `text-chat-gemma4`
- `text-chat-q35`
- `text-agent-deepseek-v4-flash`
- `speech-tts-qwen3-nano`

Docs, tests, and examples should use the canonical IDs reported by
`mere.run model list`.

### 3. Quiet default output

Normal success-path runs should emit only intended user-facing output. Internal
bring-up logging belongs behind the shared debug helper, not in default stdout.

## Runtime family pattern

The runtime families now follow a consistent reading pattern:

- root file owns the public type and high-level orchestration
- companion files own loading, generation, decoding, support, or diagnostics

That pattern is used heavily in:

- `Qwen3TTSGenerator`
- `LightOnOCRGenerator`
- `ZImageTurboGenerator`
- `QwenImageEditGenerator`
- `ACEStepPipeline`
- `ImageValidateCommand`

## When to start in the CLI vs the runtime

### Start in the CLI when

- you are changing flags, help text, or output shape
- you are changing which runtime family a command dispatches to
- you are debugging how user input becomes a runtime request

### Start in the runtime when

- you are changing inference behavior
- you are debugging loading, decoding, or generation internals
- you are working inside one modality family already

## Recommended reading sequence for contributors

1. [Repository Tour](../repository-tour.md)
2. [Architecture Reading Map](../architecture.md)
3. one command file in `Sources/MereRunCLI/Commands`
4. the matching runtime family entrypoint
5. the related tests in `Tests/`
