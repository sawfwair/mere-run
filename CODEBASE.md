# mere.run codebase map

`mere.run` is a Swift package, command-line interface (CLI), and optional macOS
graphical user interface (GUI) for local-first inference on Apple Silicon. Use
this map to find the module that owns your change. The repository exposes the
public `mere.run` executable and a thin `mere.run.app` SwiftUI wrapper that runs
the CLI. `MereRunCore` owns runtime code. `AudioCore` and `AudioCodecs` own
shared audio primitives. `AudioSTT` and `AudioTTS` own speech runtimes.
`MereRunCLI` owns the modality-first command surface, and `MereRunApp` owns the
GUI shell.

## Read this first

1. `Package.swift` for target and dependency flow
2. `Sources/MereRunCLI/MereRunCLI.swift` for the public command tree
3. `apps/macos/StudioKit` and `apps/macos/StudioUI` for the optional SwiftUI wrapper
4. `docs/repository-tour.md` for top-level ownership
5. `docs/architecture.md` for runtime reading order
6. the module README inside the subsystem you are editing

## Key modules

- `Sources/MereRunCLI/Commands/`: one file per public command family or large subcommand cluster
- `apps/macos/`: macOS Studio sources, tests, assets, command templates, and CLI process launching
- `Sources/MereRunEvaluation/`: runtime-neutral external evaluation-pack schema, validation, and content hashing
- `Sources/MereRunRelayKit/`: portable relay client, executor profiles/auth, and workflow wire types shared by the CLI and app shells
- `apps/ios/`: the iOS Studio app, a relay client over `MereRunRelayKit` (see `docs/ios-studio.md`)
- `Sources/MereRunCore/`: model paths, manifests, source config, shared runtime helpers, and modality runtime implementations
- `Sources/MereRunCore/LTX/`: native video generation and MP4 output
- `Sources/MereRunCore/Cosmos3/`: native Cosmos3-Edge omnimodal generation,
  reasoner, learned action, and persistent world runtime
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
