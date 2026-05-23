# Development Workflow

This page describes the expected local workflow for editing `mere-run`.

Before using `./scripts/check.sh`, install SwiftLint and ripgrep once:

```bash
brew install swiftlint ripgrep
```

For local docs and security checks, install Node.js, pnpm, and Gitleaks:

```bash
brew install node pnpm gitleaks
```

For Linux CLI compatibility work, use a headless toolchain. On Ubuntu-style
systems the baseline packages are:

```bash
sudo apt-get update
sudo apt-get install -y clang cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
```

`ffmpeg` packages normally include both `ffmpeg` and `ffprobe`. If a runner or
developer machine installs them somewhere else, use absolute executable
overrides:

```bash
export MERERUN_FFMPEG=/opt/ffmpeg/bin/ffmpeg
export MERERUN_FFPROBE=/opt/ffmpeg/bin/ffprobe
```

## The normal loop

For most changes:

1. make the code or docs change
2. run the smallest relevant local check
3. run `./scripts/check.sh`
4. run the installed smoke sweep if you touched real runtime behavior

That keeps the package healthy without forcing a full manual validation cycle
for every tiny edit.

## Which command to run

### Docs-only change

```bash
./scripts/check.sh
```

This catches docs hygiene regressions and verifies the CLI help surface still
matches the public tree.

### Command parsing or CLI UX change

```bash
swift test
./scripts/check.sh
```

### Runtime or model-resolution change

```bash
swift test
MERERUN_RUN_E2E=core ./scripts/check.sh
```

If the change affects installed-model behavior, also run:

```bash
MERERUN_RUN_E2E=installed ./scripts/check.sh
```

### Linux CLI compatibility change

Keep this scope to the headless CLI, local API, and test fixtures. Do not move
SwiftUI, app bundle, installer, or DMG behavior into the Linux target.

```bash
./scripts/check-linux.sh
swift run mere.run --help
```

Linux CI should use CPU MLX-compatible fixtures and mocked or tiny media I/O
inputs. The Linux gate runs the hidden `MediaIOSmoke` executable against
`ffmpeg`/`ffprobe` so image, audio, MP4, mux, and frame extraction paths stay
covered without model checkpoints. CUDA setup belongs in local runtime
documentation or manual smoke notes, not in the default pull-request gate.

### Linux release packaging change

Linux release packaging is a separate x86_64/amd64 path for distributable
headless CLI artifacts:

```bash
scripts/package-linux.sh --version 0.8.0
ls dist/linux/
```

The package script builds `dist/linux/mere-run-<version>-linux-x86_64.tar.gz`,
`dist/linux/mere-run_<version>_amd64.deb`, and `dist/linux/SHA256SUMS`. It must
run on Linux; arm64 package builds intentionally fail until upstream
`mlx-swift` Linux `bf16` support is available.

The GitHub `linux-release` workflow can be triggered manually with a version
input to validate the package build and upload an Actions artifact. On published
GitHub Release events, the workflow also uploads the Linux artifacts to the
release.

## Model-store expectations

The public runtime is hard-cut to the canonical OSS model IDs. That means:

- runtime code should use canonical public IDs only
- examples should use canonical public IDs only
- model-store and server troubleshooting should point readers at
  `mere.run status`, `mere.run model list`, `mere.run model info`, and
  `mere.run model repair-manifests`

When testing locally, inspect your current state with:

```bash
swift run mere.run status
swift run mere.run model list
swift run mere.run model info image-klein-max
```

## Editing guidance

### If you touch the public command tree

- keep the modality-first structure intact
- preserve stdout and stderr discipline
- update `docs/cli.md` if the user-facing behavior changes
- update quickstarts, cookbooks, and runtime docs when the command becomes part
  of setup or troubleshooting
- add or update `Tests/MereRunCLITests`

### If you touch runtime code

- keep debug output behind the internal debug helper
- avoid introducing new implicit hosted defaults
- keep model resolution explicit and canonical
- add or update the most local tests you can
- for Linux media I/O, test executable discovery through `MERERUN_FFMPEG`,
  `MERERUN_FFPROBE`, and `PATH` without requiring real model checkpoints

### If you touch docs

- teach the current public OSS surface
- prefer links between docs pages over repeating large blocks of reference text

## Contribution boundaries

This repo is intentionally scoped to the package and CLI. Changes should not
reintroduce:

- hosted-service assumptions
- billing or entitlement logic
- app-store-only release behavior
- relay/web/app-specific product layers

See the root `CONTRIBUTING.md` file for the short policy version.

## A good contributor checklist

Before opening a PR:

- `swift build` passes
- `swift test` passes
- `./scripts/check.sh` passes
- `pnpm docs:build` passes if you changed docs
- `gitleaks detect --source . --config .gitleaks.toml --redact --no-banner`
  passes before a public release
- the docs reflect any user-facing change
- the public command and model vocabulary remain canonical

If you changed runtime behavior against real models, also include the result of:

```bash
./scripts/e2e_smoke.sh --core
```

and, when relevant:

```bash
./scripts/e2e_smoke.sh --installed
```
