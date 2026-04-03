# Testing Guide

This repo has three layers of validation:

1. Swift build and unit tests
2. CLI help and hygiene checks
3. end-to-end smoke runs against real installed models

Use the smallest layer that covers your change, then scale up before opening a
PR.

## Fast validation

```bash
swift build
swift test
```

Use this for:

- pure library changes
- parser changes that already have test coverage
- refactors that should not change public behavior

## Repo-wide validation

```bash
./scripts/check.sh
```

This script is the main gate for contributors. It runs:

- `swift build`
- `swift test`
- help smoke for the public command tree
- `mere.run model list` output sanity checks
- docs and source hygiene sweeps

Use this for almost every change before you stop.

## End-to-end smoke tests

### Core sweep

```bash
./scripts/e2e_smoke.sh --core
```

This runs a smaller, stable subset of real workflows. Use it when you touched:

- model resolution
- runtime orchestration
- output writing
- common CLI paths

### Installed sweep

```bash
./scripts/migrate_model_store.sh
./scripts/e2e_smoke.sh --installed
```

This runs the full installed-model matrix against the local model store. Use it
when you changed:

- image generation
- speech generation or transcription
- OCR
- music or video generation
- SAM segmentation or tracking behavior
- manifest handling or installed model discovery

## What each layer catches

| Layer | Catches |
| --- | --- |
| `swift build` | package graph issues, compile failures |
| `swift test` | unit and integration regressions covered by tests |
| `./scripts/check.sh` | public CLI regressions, docs hygiene issues, command-tree drift |
| `./scripts/e2e_smoke.sh --core` | common runtime-path failures against real models |
| `./scripts/e2e_smoke.sh --installed` | installed-model breakage across the full local matrix |

## Recommended combinations

### Docs or packaging change

```bash
./scripts/check.sh
```

### CLI parsing or output change

```bash
swift test
./scripts/check.sh
```

### Runtime inference change

```bash
swift test
MERERUN_RUN_E2E=core ./scripts/check.sh
```

If you touched the SAM runtime, also run at least one real local smoke like:

```bash
swift run mere.run model pull vision-segment-sam31
swift run mere.run vision segment ./image.png --prompt "a person"
swift run mere.run vision track ./clip.mp4 --prompt "a person"
```

### Model-store, manifest, or multi-family runtime change

```bash
swift test
./scripts/check.sh
./scripts/e2e_smoke.sh --installed
```

## Troubleshooting

### A managed model is “missing”

Check:

```bash
swift run mere.run model list
swift run mere.run model info image-klein-max
```

If you migrated from an older install, run:

```bash
./scripts/migrate_model_store.sh
```

### `mere.run model pull` fails immediately

That usually means model-source configuration is missing. Set one of:

- `MERERUN_MODEL_SOURCE_BASE_URL`
- `MERERUN_R2_SIGNED_URL_ENDPOINT`
- direct R2 credential variables

See [Model Sources](./model-sources.md) and [Configuration](./configuration.md).

### A smoke test fails only in `--installed`

That usually points to:

- an incomplete local model directory
- stale pre-migration directory names
- family-specific runtime expectations that the core sweep does not exercise

Use `mere.run model info <id>` to inspect the resolved install and compare it
against the expected canonical path.
