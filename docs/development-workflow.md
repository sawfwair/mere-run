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
CUDA validation is limited to the exact hosts that have run the CUDA package and
smoke path. Linux arm64 packages are only meaningful with CUDA; CPU-only arm64
packages are local smoke artifacts, not support claims.

### Linux release packaging change

Linux release packaging is a separate path for distributable headless CLI
artifacts. Build package artifacts locally on the Linux host class you intend to
validate:

```bash
scripts/package-linux.sh --version 0.22.0
ls dist/linux/
```

The package script builds
`dist/linux/mere-run-<version>-linux-<arch>.tar.gz`,
`dist/linux/mere-run_<version>_<deb-arch>.deb`, and
`dist/linux/SHA256SUMS`. It must run on Linux.

CUDA variants use a suffix so they can ship beside the CPU artifacts:

```bash
MERERUN_LINUX_ACCEL=cuda MERERUN_SKIP_MLX_CUDA_EXAMPLE=1 \
  scripts/package-linux.sh --version 0.22.0 --artifact-suffix cuda
```

That writes artifacts such as
`mere-run-<version>-linux-x86_64-cuda.tar.gz` and
`mere-run-cuda_<version>_amd64.deb`. Real CUDA runtime smoke belongs on an
actual GPU host. CUDA package builders need CMake 3.25 or newer for the
upstream `mlx-swift` CUDA bridge. On Ubuntu 22.04/Jammy images, install a current CMake with
`python3 -m pip install --upgrade "cmake>=3.25,<4"` before starting the CUDA
package build.

Arm64 CUDA package builds require a real arm64 CUDA host. CUDA `.deb` artifacts
are gated to binaries whose linked `libcudart` SONAME proves CUDA 12 or CUDA 13,
then add the matching runtime/JIT dependency family. CUDA 12 targets NVIDIA's
12.8 packages with Lambda Stack alternatives; CUDA 13 targets NVIDIA's 13.0
packages. For any other or unknown toolkit major, use `--skip-deb`; the tar
package path remains available and must be smoke-tested on a matching host.
An explicit `MERERUN_PACKAGE_LINUX_DEPS` value bypasses the automatic CUDA
major gate and is written verbatim, making dependency compatibility the
packager's responsibility.

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
