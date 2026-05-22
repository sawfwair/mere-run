# Testing Guide

This repo has three layers of validation:

1. Swift build and unit tests
2. CLI help and hygiene checks
3. end-to-end smoke runs against real installed models

Use the smallest layer that covers your change, then scale up before opening a
PR.

Linux CLI compatibility uses a narrower fixture boundary: headless CLI behavior,
media-tool discovery, and CPU MLX-sized checks. CUDA should stay out of default
CI and be documented as an optional local acceleration path when a runtime needs
it.

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
- `mere.run status` output sanity checks
- docs and source hygiene sweeps

Use this for almost every change before you stop.

## Linux CLI compatibility fixture

The Linux path is CLI-only. It must not require `mere.run.app`, SwiftUI, the
macOS installer, or DMG packaging.

Baseline runner packages:

```bash
sudo apt-get update
sudo apt-get install -y clang cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
```

Media probing and conversion should use `ffmpeg` and `ffprobe` from `PATH`.
When tests or runners need explicit binaries, set:

```bash
export MERERUN_FFMPEG=/usr/bin/ffmpeg
export MERERUN_FFPROBE=/usr/bin/ffprobe
```

The pull-request fixture should stay CPU MLX-compatible. CUDA machines can run
additional local smoke tests, but CUDA installation, driver selection, and GPU
availability are not assumptions in the shared CI contract. To use the optional
Linux CUDA bridge, prepare native artifacts and export the environment that the
script prints:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/prepare-linux-native.sh
# then export the printed PKG_CONFIG_PATH, LIBRARY_PATH, LD_LIBRARY_PATH,
# MERERUN_MLX_SWIFT_LINKAGE, MERERUN_MLX_SWIFT_BUILD_DIR,
# MERERUN_MLX_SWIFT_SOURCE_DIR, and MERERUN_MLX_SWIFT_LINK_FLAGS values.
swift build --target mere.run
```

In this mode `Package.swift` imports the CMake-built `mlx-swift` Swift modules
and static libraries instead of rebuilding `mlx-swift` through SwiftPM's
CPU-oriented Linux manifest path.

Linux release packaging has its own artifact check:

```bash
scripts/package-linux.sh --version 0.7.1
test -s dist/linux/SHA256SUMS
tar -tzf dist/linux/mere-run-*-linux-*.tar.gz | grep '/mere.run$'
tar -tzf dist/linux/mere-run-*-linux-*.tar.gz | grep '/install.sh$'
dpkg-deb --info dist/linux/mere-run_*_*.deb
dpkg-deb --contents dist/linux/mere-run_*_*.deb | grep 'usr/bin/mere.run'
```

The `linux-release` workflow runs the same package and manifest boundary on
Ubuntu 22.04 in the Swift 6.0 container. Manual workflow runs upload Actions
artifacts only; published GitHub Release events also upload the Linux assets to
the release.

MediaIO coverage has two layers. `Tests/MereRunCoreTests/MediaIOTests.swift`
covers pure Swift image, WAV, and FFT behavior in the normal test suite.
`scripts/check-linux.sh` also runs the hidden `MediaIOSmoke` SwiftPM executable
to exercise Linux `ffmpeg`/`ffprobe` image, audio, MP4, mux, and frame
extraction paths without requiring host media files or model checkpoints.

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
| Linux CLI fixture | headless CLI/media compatibility without macOS app or CUDA assumptions |

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

### Linux CLI compatibility docs or fixtures

```bash
swift run mere.run --help
```

If the change only updates Linux documentation, the CI docs fixture is enough.
If the change adds Linux-compatible code, include the focused unit test or stub
fixture result that exercises the new behavior.

If the change touches Linux release packaging or `.github/workflows/linux-release.yml`,
run the package script on Linux or dispatch the `linux-release` workflow on
`main` with a test version.

## Troubleshooting

### A managed model is “missing”

Check:

```bash
swift run mere.run status
swift run mere.run model list
swift run mere.run model info image-klein-max
```

`status` shows the active model store first; `model info` is better when you
need manifest and component details for one model.

### `mere.run model pull` fails immediately

That usually means the requested model is local-path-only in the public build or
the current Mac does not pass the capability check.

Run:

```bash
swift run mere.run model capabilities --all
```

See [Model Sources](./model-sources.md) and [Configuration](./configuration.md).

### A smoke test fails only in `--installed`

That usually points to:

- an incomplete local model directory
- missing or stale manifest metadata
- family-specific runtime expectations that the core sweep does not exercise

Use `mere.run model info <id>` to inspect the resolved install and compare it
against the expected canonical path.
