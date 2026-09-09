# mere.run codebase map

`mere.run` is a Swift package, CLI, and optional macOS GUI for local inference
on Apple Silicon. This map identifies source owners. `mere.run.app` runs the
CLI. `MereRunCore` owns inference; `MereRunModelKit` owns metadata and installed
lookup. `AudioCore` and `AudioCodecs` own audio primitives; `AudioSTT` and
`AudioTTS` own speech orchestration. `MereRunCLI` owns command presentation.

## Read this first

1. `Package.swift` for target and dependency flow
2. `Sources/MereRunCLI/MereRunCLI.swift` for the public command tree
3. `apps/macos/StudioKit` and `apps/macos/StudioUI` for the optional SwiftUI wrapper
4. `docs/repository-tour.md` for top-level ownership
5. `docs/architecture.md` for runtime reading order
6. the module README inside the subsystem you are editing

## Key modules

- `Sources/MereRunCLI/Commands/`: public command families
- `apps/macos/`: macOS Studio sources, tests, assets, templates, and CLI launching
- `Sources/MereRunEvaluation/`: runtime-neutral external evaluation-pack schema, validation, and content hashing
- `Sources/MereRunRelayKit/`: portable relay client, executor profiles/auth, and workflow wire types shared by the CLI and app shells
- `apps/ios/`: the iOS Studio app, a relay client over `MereRunRelayKit` (see `docs/ios-studio.md`)
- `Sources/MereRunAdmission/` and `Sources/MereRunResidency/`: reservations, queues, model generations, leases, and eviction
- `Sources/MereRunModelKit/`: model identities, manifests, storage paths, registered locations, artifact pins, and installed lookup
- `Sources/MereRunCore/`: catalog assembly, downloads, and runtime orchestration
- `Sources/MereRunQwenModel/` and `Sources/MereRunGemmaModel/`: text/vision layers, caches, and draft state
- `Sources/MereRunDecode/`: shared sampling, pipelined token decoding, streaming, and logprob diagnostics
- `Sources/MereRunTensor/`, `Sources/MereRunTextEncoder/`, and `Sources/MereRunImageModels/`: checkpoint loading, tensor kernels, and image model layers
- `Sources/MereRunCore/ImageGeneration*.swift` and `Sources/AudioCore/SpeechTranscriptionOperation.swift`: shared operation plans, validation, execution, and outcomes
- `Sources/MereRunCore/LTX/`: native video generation and MP4 output
- `Sources/MereRunCore/Cosmos3/`: omnimodal generation and world runtime
- `Sources/MereRunCore/SCAIL2/`: native SCAIL-2 transformer, OpenCLIP, masks,
  Wan 2.1 VAE loading, segmented generation, and MP4 orchestration
- `Sources/MereRunCore/LoRA/`: LoRA checkpoint, artifact, and compatibility logic
- `Sources/AudioQwen3ASRModel/`, `Sources/AudioQwen3TTSModel/`, `Sources/AudioParakeetModel/`, `Sources/AudioSortformer/`, and `Sources/MereRunKVCache/`: speech models, diarization, and attention caches
- `Sources/AudioTTS/Qwen3TTS/`: Qwen TTS loading, prompts, token generation, and audio output
- `Tests/MereRunCoreTests/`: most behavior and compatibility coverage
- `Tests/MereRunCLITests/`: parsing and CLI contract coverage

## Validation

- Fast loop: `swiftlint --strict && swift build && swift test`
- Main gate: `./scripts/check.sh`
- Agent-readiness gate: `bash ./scripts/agent_readiness_check.sh`
- Runtime smoke: `MERERUN_RUN_E2E=core ./scripts/check.sh`
- Installed-model smoke: `MERERUN_RUN_E2E=installed ./scripts/check.sh`

## Review before editing

- Do not modify `vendor/` without reviewing the vendored runtime requirements.
- Before editing a large model-definition file in `Sources/MereRunCore/`, read
  the local module README.
- When you change canonical model IDs, migration vocabulary, or hosted-default
  hygiene checks, update the documentation and tests together.

## Editing rules

- Prefer typed decoding at configuration and tokenizer boundaries over
  `[String: Any]`.
- Keep stdout machine-readable and stderr diagnostic in CLI commands.
- Add or update the closest test file for command parsing, model resolution, or
  compatibility behavior.
- After changing the command tree or command abstracts, run
  `./scripts/update-docs-command-reference.sh`. The documentation contract owns
  every top-level command and fails `swift test` on drift.
- Add a module README before a source directory grows past 500 direct Swift
  lines.
- If a change requires real checkpoint assets or GPU-only validation, stop
  after the local gate and report the remaining validation gap explicitly.
