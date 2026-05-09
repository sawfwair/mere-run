---
name: mere-run
description: Develop, inspect, test, or modify the public `mere.run` Swift CLI in this OSS repository. Use when Codex needs to run source-checkout commands such as `swift run mere.run`, inspect the modality-first command tree as code, change CLI parsing, model resolution, local API behavior, or command output, update CLI docs/tests, validate with `./scripts/check.sh`, or distinguish the public `mere.run` inference CLI from the separate root `mere` portfolio command plane. For helping an end user operate the CLI without editing this repo, use the `use-mere-run` skill instead.
---

# mere.run CLI

## Overview

Use this skill for developing the public OSS `mere.run` CLI built by the Swift package in this repo. Keep it separate from the root `mere` portfolio CLI; this repo's command is `swift run mere.run ...` during development and `mere.run ...` only when testing an installed binary. For newcomer/operator help, use the companion `use-mere-run` skill.

## First Moves

- Work from the repo root.
- Read `AGENTS.md` for the repo contract when it is not already loaded.
- For code changes, read `Package.swift`, `CODEBASE.md`, `Sources/MereRunCLI/MereRunCLI.swift`, the closest command file under `Sources/MereRunCLI/Commands/`, and the closest tests under `Tests/MereRunCLITests/` or `Tests/MereRunCoreTests/`.
- For CLI usage or command discovery, start with `swift run mere.run --help`, then inspect the relevant subcommand help.
- Treat `README.md`, `docs/cli.md`, `docs/model-sources.md`, and the managed model registry as the canonical public surface.

## Command Surface

Prefer the source checkout command while developing:

```bash
swift run mere.run --help
swift run mere.run guide --list
swift run mere.run model capabilities
swift run mere.run model list
```

Keep the public command tree modality-first:

- `image generate`, `image validate`
- `guide`, `guide --list`, `guide <command path>`, `guide <command path> --model <model-id>`, `guide <command path> --json`
- `text chat`, `text code`, `text embed`, `text anonymize`
- `speech synthesize`, `speech transcribe`, `speech profile`
- `vision caption`, `vision inspect`, `vision ground`, `vision segment`, `vision track`, `vision track-live`, `vision ocr`
- `music generate`
- `video generate`, `video export-latents`
- `model list`, `model capabilities`, `model info`, `model pull`, `model remove`, `model repair-manifests`
- `api serve`
- `setup`
- `agent onboard`, `agent install-pi`, `agent start`

Use managed model IDs from `docs/cli.md`, `docs/model-sources.md`, or `Sources/MereRunCore/ManagedModelCatalog.swift`; avoid guessing retired or private identifiers.

## Runtime State

Use the default model store unless isolation matters:

```text
~/Library/Application Support/MereRun/models
```

For isolated tests or reproductions, pass a model root explicitly:

```bash
swift run mere.run --models-root /tmp/mererun-models model list
```

Respect the related environment overrides: `MERERUN_MODELS_DIR`, `MERERUN_HUB_CACHE`, and `MERERUN_MODEL_CACHE_HOME`.

## Implementation Rules

- Keep stdout machine-readable and stderr diagnostic/progress-oriented in CLI implementations.
- Use typed decoding at config and tokenizer boundaries. Keep dynamic compatibility shims narrow and close to the ingestion point.
- Update the closest CLI or core test when changing command parsing, model resolution, compatibility behavior, API behavior, or output shape.
- Update `README.md`, `docs/`, or `CHANGELOG.md` when changing public CLI behavior, setup, model guidance, or security-sensitive defaults.
- Preserve the public OSS boundary: no hosted-service, billing, app-store, or private-deployment surfaces.
- Leave `vendor/` unchanged unless explicitly required; update `THIRD_PARTY_NOTICES.md` in the same change if vendor artifacts change.
- Preserve API safety defaults. `api serve` may bind loopback without auth; non-loopback hosts require `--api-key` or `MERERUN_API_KEY`.
- Preserve tool-loop safety. `text chat` tool execution requires interactive approval unless the user opts into supported auto-approval behavior.

## Validation

Use the repo gate before treating a change as ready:

```bash
./scripts/check.sh
```

That gate runs SwiftLint strict mode, build, tests, CLI `--help` smoke coverage, and hygiene scans for legacy names, old model IDs, retired command vocabulary, and debug prints. If the hygiene scan fails, inspect the rejected pattern and use current docs, tests, or the managed model registry for the canonical replacement.

Use narrower checks while iterating:

```bash
swift build
swift test
swift run mere.run --help
```

Add runtime smoke only when the task needs it and the machine has the needed assets:

```bash
MERERUN_RUN_E2E=core ./scripts/check.sh
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

When real checkpoint assets, GPU-only behavior, or non-loopback network validation is required, run the local gate first and call out the remaining validation gap explicitly.
