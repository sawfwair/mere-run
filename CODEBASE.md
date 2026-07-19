# mere.run Codebase Map

mere.run is a Swift package, CLI, and optional macOS GUI for local-first inference on Apple Silicon. The repo exposes the public `mere.run` executable plus a thin `mere.run.app` SwiftUI wrapper that runs that CLI. Runtime code lives in `MereRunCore`, shared audio primitives live in `AudioCore` and `AudioCodecs`, speech runtimes live in `AudioSTT` and `AudioTTS`, `MereRunCLI` owns the modality-first command surface, and `MereRunApp` owns the GUI shell.

## Read This First

1. `Package.swift` for target and dependency flow
2. `Sources/MereRunCLI/MereRunCLI.swift` for the public command tree
3. `Sources/MereRunApp` for the optional SwiftUI wrapper
4. `docs/repository-tour.md` for top-level ownership
5. `docs/architecture.md` for runtime reading order
6. the module README inside the subsystem you are editing

## Key Modules

- `Sources/MereRunCLI/Commands/`: one file per public command family or large subcommand cluster
- `Sources/MereRunApp/`: macOS GUI forms, command templates, and CLI process launching
- `Sources/MereRunCore/`: model paths, manifests, source config, shared runtime helpers, and modality runtime implementations
- `Sources/MereRunCore/LTX/`: native video generation and MP4 output
- `Sources/MereRunCore/SCAIL2/`: native SCAIL-2 transformer, OpenCLIP, masks,
  Wan 2.1 VAE loading, segmented generation, and MP4 orchestration
- `Sources/MereRunCore/LoRA/`: LoRA checkpoint, artifact, and compatibility logic
- `Sources/AudioSTT/Qwen3ASR/`: Qwen3 ASR config, tokenizer, model, and generator path
- `Sources/AudioTTS/Qwen3TTS/`: Qwen3 TTS tokenizer, model, and generation path
- `Tests/MereRunCoreTests/`: most behavior and compatibility coverage
- `Tests/MereRunCLITests/`: parsing and CLI contract coverage

## Validation

- Fast loop: `swiftlint --strict && swift build && swift test`
- Main gate: `./scripts/check.sh`
- Agent-readiness gate: `bash ./scripts/agent_readiness_check.sh`
- Runtime smoke: `MERERUN_RUN_E2E=core ./scripts/check.sh`
- Installed-model smoke: `MERERUN_RUN_E2E=installed ./scripts/check.sh`

## Do Not Touch Blindly

- `vendor/`: vendored runtime artifacts
- giant model-definition files in `Sources/MereRunCore/` without first reading the local module README
- canonical model IDs, migration vocabulary, or hosted-default hygiene checks without updating docs and tests together

## Editing Rules

- Prefer typed decoding at config/tokenizer boundaries over `[String: Any]`
- Keep stdout machine-readable and stderr diagnostic in CLI commands
- Add or update the closest test file for command parsing, model resolution, or compatibility behavior
- After changing the command tree or command abstracts, run `./scripts/update-docs-command-reference.sh`; the docs contract owns every top-level command and fails `swift test` on drift
- Add a module README before a source directory grows past 500 direct Swift lines
- If a change requires real checkpoint assets or GPU-only validation, stop after the local gate and report the remaining validation gap explicitly
