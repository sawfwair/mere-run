# mere.run codebase map

`mere.run` runs local inference on Apple Silicon, and the optional
`mere.run.app` Studio runs that CLI. `MereRunCore` owns orchestration and the
compatibility facade; owned runtime targets hold model computation.
`MereRunModelKit` owns model metadata and installed lookup. `AudioSTT` and
`AudioTTS` own speech orchestration. `MereRunCLI` owns command presentation.

## Read this first

1. `Package.swift` and `scripts/package-policy.json` for targets and dependencies
2. `Sources/MereRunCLI/MereRunCLI.swift` for the public command tree
3. `docs/repository-tour.md` for ownership
4. `docs/architecture.md` for runtime reading order
5. The subsystem README

## Key modules

- `Sources/MereRunCLI/Commands/`: public command families
- `Sources/MereRunContract/`: CLI capability and result contracts
- `Sources/MereRunEvaluation/`: evaluation-pack schema and hashing
- `Sources/MereRunRelayKit/`: relay client and workflow wire types
- `Sources/MereRunModelKit/`: model identities, manifests, paths, and installed lookup
- `Sources/MereRunExecution/`, `Sources/MereRunAdmission/`, `Sources/MereRunResidency/`: durable runs, reservations, and eviction
- `Sources/MereRunTensor/`, `Sources/MereRunDecode/`, `Sources/MereRunKVCache/`, `Sources/MereRunTextEncoder/`: checkpoint loading, sampling, attention caches, and text encoding
- `Sources/MereRun*Model/`, `Sources/MereRunImageModels/`, `Sources/MereRunAudioModels/`: model computation per family
- `Sources/MereRunCore/`: catalog, downloads, and per-family loading and generation
- `Sources/MereRunCore/LoRA/`: LoRA checkpoints and compatibility
- `Sources/AudioCore/`, `Sources/AudioCodecs/`: audio operations and codecs
- `Sources/Audio*Model/`, `Sources/AudioSortformer/`: speech model computation and diarization
- `Sources/AudioSTT/`, `Sources/AudioTTS/`: transcription and synthesis orchestration
- `apps/macos/`: Studio (`StudioKit`, `StudioUI`, `MereRunStudio`)
- `apps/ios/`: iOS Studio, a relay client with on-device image and chat
- `Tests/MereRunCoreTests/`: most behavior and compatibility coverage
- `Tests/MereRunCLITests/`: parsing, CLI contract, and documentation contract coverage

## Validation

- Fast loop: `swiftlint --strict && swift build && swift test`
- Main gate: `./scripts/check.sh`
- Agent-readiness gate: `bash ./scripts/agent_readiness_check.sh`
- Package policy: `bash ./scripts/check-package-policy.sh`
- Runtime smoke: `MERERUN_RUN_E2E=core ./scripts/check.sh`
- Installed-model smoke: `MERERUN_RUN_E2E=installed ./scripts/check.sh`

## Review before editing

- Don't modify `vendor/` without updating `THIRD_PARTY_NOTICES.md`.
- Read the subsystem README before editing model definitions.
- When you change canonical model IDs, migration vocabulary, or hygiene checks,
  update the documentation and tests together.

## Editing rules

- Follow [package policy](docs/internals/package-policy.md) when changing
  targets, products, dependencies, or re-exports. Register every new target.
- Put new model runtimes in an owned runtime target, not in `MereRunCore`.
- Use typed decoding at configuration and tokenizer boundaries, not
  `[String: Any]`.
- Keep stdout machine-readable and stderr diagnostic in CLI commands.
- Add or update the closest test for command parsing, model resolution, or
  compatibility behavior.
- After changing the command tree or command abstracts, run
  `./scripts/update-docs-command-reference.sh`. The documentation contract
  fails `swift test` on drift.
- Add a module README before a source directory reaches 500 direct Swift lines.
- If a change needs real checkpoints or GPU-only validation, stop after the
  local gate and report the remaining validation gap.
