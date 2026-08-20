# Testing guide

This repository has three layers of validation:

1. Swift build and unit tests
2. CLI help and hygiene checks
3. End-to-end smoke runs against real installed models

Use the smallest layer that covers your change, then scale up before opening a
pull request.

Linux CLI compatibility uses a narrower fixture boundary for hosted CI:
headless CLI behavior and media-tool discovery. There is no CPU-only Linux
release. Every release-worthy Linux package must exercise its CUDA lane on
matching real CUDA hardware. Active release validation covers macOS and the
configured `tensor.local` x86_64 CUDA builder. The arm64 CUDA lane remains
paused until a matching host is available.

## Fast validation

```bash
swift build
swift test
```

Use this for:

- Pure library changes
- Parser changes that already have test coverage
- Refactors that should not change public behavior

## Repo-wide validation

```bash
./scripts/check.sh
```

This script is the main gate for contributors. It runs:

- `swift build`
- `swift test`
- Help smoke for the public command tree
- Generated CLI documentation and command-owner synchronization
- `mere.run model list` output sanity checks
- `mere.run status` output sanity checks
- Documentation and source hygiene sweeps

Run this gate for most changes before you open a pull request.

### CLI documentation contract

`DocumentationContractTests` derives the complete public command tree from
`MereRunCLI.configuration`. It compares that tree with the generated command
inventories, verifies that every top-level command owns a navigable docs page,
checks runtime-guide sidebar coverage, and rejects stale command paths in
Markdown examples.

When a command changes intentionally, regenerate the inventories before running
the main gate:

```bash
./scripts/update-docs-command-reference.sh
swift test --filter DocumentationContractTests
```

## Linux CLI compatibility fixture

The Linux path is CLI-only. It must not require `mere.run.app`, SwiftUI, the
macOS installer, or DMG packaging.

Baseline runner packages:

```bash
sudo apt-get update
sudo apt-get install -y clang cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
```

CUDA package builders need CMake 3.25 or newer for the upstream `mlx-swift`
CUDA bridge. Ubuntu 22.04/Jammy's default CMake is too old, so upgrade it
before CUDA packaging with `python3 -m pip install --upgrade "cmake>=3.25,<4"`.

Media probing and conversion should use `ffmpeg` and `ffprobe` from `PATH`.
When tests or runners need explicit binaries, set:

```bash
export MERERUN_FFMPEG=/usr/bin/ffmpeg
export MERERUN_FFPROBE=/usr/bin/ffprobe
```

The pull-request fixture must remain CPU MLX-compatible on hosted x86 runners.
CUDA machines can run additional package and smoke tests, but CUDA installation,
driver selection, and GPU availability are not assumptions in the shared hosted
CI contract. Do not describe a CUDA configuration as tested until that exact
host has run the CUDA path. To use the Linux CUDA bridge, prepare native
artifacts and export the environment that the script prints:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/prepare-linux-native.sh
# then export the printed PKG_CONFIG_PATH, LIBRARY_PATH, LD_LIBRARY_PATH,
# MERERUN_MLX_SWIFT_LINKAGE, MERERUN_MLX_SWIFT_BUILD_DIR,
# MERERUN_MLX_SWIFT_SOURCE_DIR, and MERERUN_MLX_SWIFT_LINK_FLAGS values.
swift build --target mere.run
```

In this mode `Package.swift` imports the CMake-built `mlx-swift` Swift modules
and static libraries instead of rebuilding `mlx-swift` through SwiftPM's
CPU-oriented Linux manifest path. The preparation script checks out the exact
`mlx-swift` revision selected by SwiftPM under the active Linux Swift toolchain,
keeping the CMake bridge and Swift package graph aligned even when that
toolchain resolves a compatibility revision different from a Mac checkout.
`MLX_SWIFT_CUDA_COMMIT` and `MLX_SWIFT_CUDA_URL` are deliberate diagnostic
overrides for maintainers, not part of normal setup. Normal builds use the
exact `sawfwair/mlx-swift` revision selected by SwiftPM and initialize its
pinned MLX and mlx-c submodules before CMake configuration. This pin removes
source drift; it does not count as CUDA validation on a host that has not run
the path.

Linux release packaging has its own artifact check:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/package-linux.sh --version 0.23.0
test -s dist/linux/SHA256SUMS
tar -tzf dist/linux/mere-run-*-linux-*.tar.gz | grep '/mere.run$'
tar -tzf dist/linux/mere-run-*-linux-*.tar.gz | grep '/install.sh$'
dpkg-deb --info dist/linux/mere-run_*_*.deb
dpkg-deb --contents dist/linux/mere-run_*_*.deb | grep 'usr/bin/mere.run'
```

These release checks are CUDA-only. CPU builds in the packaging test suite are
fixtures for archive and dependency logic, not publishable artifacts.

The x86_64 CUDA release artifact uses a suffix and can be built on a CPU-only
Linux builder with CUDA development packages:

```bash
MERERUN_LINUX_ACCEL=cuda MERERUN_SKIP_MLX_CUDA_EXAMPLE=1 \
  scripts/package-linux.sh --version 0.23.0 --artifact-suffix cuda
tar -tzf dist/linux/mere-run-*-linux-x86_64-cuda.tar.gz | grep '/.mererun-linux-cuda$'
dpkg-deb --info dist/linux/mere-run-cuda_*_amd64.deb
```

The `.deb` check requires the built binary to link one unambiguous CUDA 12 or
CUDA 13 `libcudart` SONAME. Packaging selects the matching dependency family
and fails instead of writing dependencies for an unknown toolkit major. The
selected family includes CUDA runtime development headers because MLX's NVRTC
kernels compile during inference, not only during package construction.
An explicit `MERERUN_PACKAGE_LINUX_DEPS` fixture separately verifies the
maintainer override; its value is written verbatim and bypasses the default
major gate.

On Linux arm64, use CUDA for the package check:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/package-linux.sh --version 0.23.0
```

Run Linux package and manifest checks on the affected Linux host class. CUDA
artifacts need a matching CUDA host for meaningful runtime smoke coverage.

For the speaker-diarization CUDA checkpoint, use a real recording whose first
speaker returns after a different middle speaker. The gate requires a visible
NVIDIA GPU, confirms that MLX reports its GPU device, validates timed segments,
and verifies the A-B-A speaker sequence:

```bash
MERERUN_SORTFORMER_BIN=/path/to/mere.run \
MERERUN_SORTFORMER_MODEL_DIR=/path/to/sortformer-model \
MERERUN_SORTFORMER_AUDIO=/path/to/two-speaker-a-b-a.wav \
MERERUN_SORTFORMER_OUTPUT=./sortformer-cuda-result.json \
  scripts/check-sortformer-diarization.sh --require-cuda
```

The model directory must come from a license-acknowledged managed installation
or an equivalent local checkpoint. A CUDA build without this real-audio GPU and
speaker-reidentification result is not diarization runtime proof.

### Historical GB10 and DGX Spark sweep

The GB10 and DGX Spark command remains as a historical validation recipe. These
machines are not active release builders. If an arm64 CUDA host becomes
available again, use the installed CUDA binary and record the quantized-kernel
choice alongside artifacts and throughput:

```bash
scripts/e2e_gb10.sh --bin /usr/bin/mere.run --quant-mode auto --out ./e2e-gb10-auto
```

`auto` probes `quantized_mm` and `GatherQMM` independently and records either
`backend=native` or `backend=dense` in the result detail. Use `--quant-mode
native` for a fail-loud kernel-availability run and `--quant-mode dense` for a
compatibility/throughput baseline. Do not treat the automatic selection as
GB10-validated until this sweep has run against the exact packaged binary and
host.

MediaIO coverage has two layers. `Tests/MereRunCoreTests/MediaIOTests.swift`
covers pure Swift image, WAV, and FFT behavior in the normal test suite.
`scripts/check-linux.sh` also runs the hidden `MediaIOSmoke` SwiftPM executable
to exercise Linux `ffmpeg`/`ffprobe` image, audio, MP4, mux, and frame
extraction paths without requiring host media files or model checkpoints.

### Unified audiovisual audio comparison

When unified audiovisual (AV) audio sounds too hot, bandwidth-limited, fluttery, or different
from the upstream LTX-2 reference, generate the same prompt/seed through both
paths and compare the finished MP4s:

```bash
scripts/compare-ltx-av-audio.py \
  --mere ./mere-unified-av.mp4 \
  --ltx ./ltx-reference.mp4 \
  --ltx-repo /path/to/LTX-2 \
  --model-root "$HOME/Library/Application Support/MereRun/models/video-ltx-av" \
  --json-out ./ltx-av-audio-report.json
```

The report combines `ffprobe` stream metadata, decoded PCM measurements
(LUFS, true peak, RMS, crest factor, near-clipping, and high-frequency energy),
and source-reference checks against local media assembly and the optional LTX-2
checkout. When `--model-root` is provided it also scans model JSON and
safetensors headers for vocoder BWE config/weights, so support in source code
is kept separate from BWE actually being present in the installed checkpoint.
Use it to separate waveform/vocoder problems from mux or AAC encoding
differences.

`video-ltx23-full-mlx` is the managed LTX 2.3 dev + distilled-LoRA checkpoint
for the high-quality two-stage `--quality final` and A2Vid paths. Generated
audio is independently selected with `--output-mode audio-video`; leaving it
off does not turn the full checkpoint into the fast lane. The standalone
`video-ltx23-av-mlx` distilled split remains the fast `--quality draft` lane.
Both can be scanned by the comparison script after generating a sample. The
Unsloth LTX 2.3 GGUF checkpoints are a separate quantized runner shape, not a
drop-in native MLX model root.

## End-to-end smoke tests

### Core sweep

```bash
./scripts/e2e_smoke.sh --core
```

This runs a smaller, stable subset of real workflows. Use it when you touched:

- Model resolution
- Runtime orchestration
- Output writing
- Common CLI paths

### Installed sweep

```bash
./scripts/e2e_smoke.sh --installed
```

This runs the established cross-modality installed-model subset against the
local model store. It is useful for broad development feedback, but it is not a
claim that every installed checkpoint ran. Use it when you changed:

- Image generation
- Speech generation or transcription
- OCR
- music or video generation
- SAM segmentation or tracking behavior
- Manifest handling or installed model discovery

### Strict pre-release video generation

```bash
/path/to/extracted/mere.run gate \
  --suite video \
  --require-all \
  --json-output ./video-gate.json
```

This is the release-blocking video contract. It runs true native generation for
the LTX 2.3 draft, full generated-audio, and source-audio A2Vid routes; decodes
the written MP4; and requires non-silent audio for the two audio-bearing paths.
Run it from the exact packaged binary. A missing required model is a failure,
not a skip.

The final packaged candidate runs the exhaustive installed-model matrix:

```bash
MERERUN_SORTFORMER_AUDIO=/path/to/two-speaker-a-b-a.wav \
  /path/to/extracted/mere.run gate \
  --all-installed \
  --require-all \
  --json-output ./release-gate.json
```

That produces one result per installed model ID, including every image model;
TripoSR, InstantMesh, and TRELLIS.2; music and SFX; OCR, SAM, grounding, face,
geometry, and depth; speech, embeddings, privacy, text, and every video/world
backend. Component-only entries must be consumed by a named true companion run.
An installed model with no recipe fails closed. When Sortformer is installed,
`MERERUN_SORTFORMER_AUDIO` is required and must identify a real fixture whose
first speaker returns after a different middle speaker.

If a release owner explicitly quarantines a known-broken installed model, add
`--skip-model <installed-id>`. The evidence retains it as a `skipped` row
rather than claiming a pass; invalid quarantine IDs fail closed.

## What each layer catches

| Layer | Catches |
| --- | --- |
| `swift build` | package graph issues, compile failures |
| `swift test` | unit and integration regressions covered by tests |
| `./scripts/check.sh` | public CLI regressions, docs hygiene issues, command-tree drift |
| `./scripts/e2e_smoke.sh --core` | common runtime-path failures against real models |
| `./scripts/e2e_smoke.sh --installed` | broad installed-model subset across common local workflows |
| `mere.run gate --suite video --require-all` | packaged LTX draft/full/A2Vid generation, MP4 decode, and promised audio |
| `mere.run gate --all-installed --require-all` | exhaustive packaged true-inference matrix with one result per installed model ID |
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

If you changed the Segment Anything Model (SAM) runtime, also run at least one
real local smoke test:

```bash
swift run mere.run model pull vision-segment-sam31 --accept-model-license
swift run mere.run vision segment ./image.png --prompt "a person"
swift run mere.run vision track ./clip.mp4 --prompt "a person"
```

### Model-store, manifest, or multi-family runtime change

```bash
swift test
./scripts/check.sh
./scripts/e2e_smoke.sh --installed
```

### Video runtime, MLX stream, or release candidate change

```bash
./scripts/check.sh
/path/to/extracted/mere.run gate --suite video --require-all \
  --json-output ./video-gate.json
```

Do not substitute `--preflight`, model validation, or a source-built binary for
the packaged true-generation command.

### Linux CLI compatibility docs or fixtures

```bash
swift run mere.run --help
```

If the change only updates Linux documentation, the CI docs fixture is enough.
If the change adds Linux-compatible code, include the focused unit test or stub
fixture result that exercises the new behavior.

If the change touches Linux release packaging, run the package script on the
affected Linux architecture. For arm64 CUDA changes, that means a
CUDA-provisioned arm64 host, not a hosted CPU arm64 runner.

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
the Mac does not pass the capability check.

Run:

```bash
swift run mere.run model capabilities --all
```

See [Model sources](./model-sources.md) and [Configuration](./configuration.md).

### A smoke test fails only in `--installed`

That usually points to:

- an incomplete local model directory
- missing or stale manifest metadata
- family-specific runtime expectations that the core sweep does not exercise

Use `mere.run model info <id>` to inspect the resolved install and compare it
against the expected canonical path.
