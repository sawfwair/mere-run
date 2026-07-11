# Linux QuickStart

This page is for installing or building the headless `mere.run` CLI on Linux.
Linux package artifacts are CLI-only and must be validated on the host class
they target. CUDA artifacts must be built and smoke-tested on matching CUDA
hardware before being treated as supported.

The macOS path remains the primary hands-on development and runtime validation
environment for this repo. The Linux package path is real, but intentionally
narrow: CLI-only packages, Ubuntu-style hosts, x86_64 CPU and CUDA package
manifests, and CUDA-gated arm64 package work.

## Current validation boundary

- Linux package artifacts are headless CLI-only.
- Package validation must cover the portable tarball, Debian package, runtime
  library bundling, and package manifests on the target Linux host class.
- CUDA package validation must include a real CUDA GPU host.
- Linux packages do not include `MereRun.app`, the SwiftUI studio, the macOS
  installer UI, or the DMG layout.
- Linux arm64 package builds must use `MERERUN_LINUX_ACCEL=cuda`; CPU arm64
  packages are only local smoke-test artifacts.
- NVIDIA GB10/DGX Spark is a valid arm64 CUDA target when the package is built
  and smoke-tested on matching hardware.
- Current CUDA validation should be treated as limited to the exact hosts that
  have run the CUDA package and smoke path.

## Install with apt

Use this path on Debian or Ubuntu-style systems after building a matching
`.deb`, or when a matching `.deb` is attached to a release:

```bash
version=0.20.0
case "$(uname -m)" in
  x86_64|amd64) deb_arch=amd64 ;;
  *) echo "use the arm64 CUDA package path for Linux arm64" >&2; exit 1 ;;
esac

sudo apt install "./dist/linux/mere-run_${version}_${deb_arch}.deb"
mere.run --version
mere.run status
```

The Debian package installs the `mere.run` command and the colocated runtime
assets that the CLI needs at runtime.

## Install the portable tarball

Use this path after building a matching tarball, or when a matching tarball is
attached to a release:

```bash
version=0.20.0
case "$(uname -m)" in
  x86_64|amd64) linux_arch=x86_64 ;;
  *) echo "use the arm64 CUDA package path for Linux arm64" >&2; exit 1 ;;
esac

tar -xzf "./dist/linux/mere-run-${version}-linux-${linux_arch}.tar.gz"
cd "mere-run-${version}-linux-${linux_arch}"
./install.sh
mere.run --version
mere.run status
```

The tarball installer copies the CLI and its runtime assets together. The
packaged launcher resolves its own install location and sets `LD_LIBRARY_PATH`
for the bundled runtime libraries before it starts the real `mere.run` binary.
On CUDA 13 hosts it also resolves CUDA CCCL from CUDA home, versioned Toolkit
roots, and target-specific include directories, then adds that `cuda/std` root
to `CPATH` and `MERERUN_CUDA_CCCL_INCLUDE_PATH`. MLX CUDA kernels use that
explicit path while JIT-compiling in the installed package.

## Install the x86_64 CUDA tarball

Use this artifact for GPU worker images, remote trainers, and other x86_64 CUDA
hosts after building it, or when a matching CUDA tarball is attached to a
release:

```bash
version=0.20.0
tar -xzf "./dist/linux/mere-run-${version}-linux-x86_64-cuda.tar.gz"
cd "mere-run-${version}-linux-x86_64-cuda"
./install.sh
mere.run --version
mere.run status
```

The CUDA tarball is the right build pack for companion remote-runner plugins.
It is not a source/bootstrap archive and should not compile on paid GPU time.

## First commands

After installing:

```bash
mere.run --help
mere.run guide --list
mere.run model list
mere.run model capabilities
mere.run model capabilities --recommended
mere.run status
```

Use `mere.run model capabilities` before pulling large checkpoints. It gives a
machine-local view of the recommended public model IDs for the current host.

## Media commands

Linux media paths expect `ffmpeg` and `ffprobe`:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

If the binaries are not on `PATH`, point the CLI at explicit locations:

```bash
export MERERUN_FFMPEG=/usr/bin/ffmpeg
export MERERUN_FFPROBE=/usr/bin/ffprobe
```

## Linux x86_64 CUDA package path

The package script can build x86_64 CUDA artifacts with CUDA development
packages and `MERERUN_SKIP_MLX_CUDA_EXAMPLE=1`. To build locally on a Linux
x86_64 CUDA development host:

```bash
MERERUN_LINUX_ACCEL=cuda MERERUN_SKIP_MLX_CUDA_EXAMPLE=1 \
  scripts/package-linux.sh --version 0.20.0 --artifact-suffix cuda
ls dist/linux/
```

On rented GPU builders, keep paid compile time down by setting
`MERERUN_NATIVE_BUILD_JOBS` to the worker's real vCPU count and
`MERERUN_CUDA_ARCHITECTURES` to the target build policy. For example, an RTX
A4000 builder that should also ship forward-compatible H100 PTX can use:

```bash
MERERUN_LINUX_ACCEL=cuda \
MERERUN_SKIP_MLX_CUDA_EXAMPLE=1 \
MERERUN_NATIVE_BUILD_JOBS=14 \
MERERUN_CUDA_ARCHITECTURES="86-real;90-virtual" \
  scripts/package-linux.sh --version 0.20.0 --artifact-suffix cuda
```

Run a real GPU smoke on the target runtime class before widening support claims.

## Linux arm64 CUDA package path

Linux arm64 is only useful for this project when the CUDA lane works. Do not
publish or recommend CPU-only arm64 packages as release artifacts.

Build arm64 packages on a native arm64 Linux host with an NVIDIA GPU, driver,
CUDA Toolkit, `nvcc`, CUDA CCCL headers, cuDNN, NCCL, Swift,
OpenBLAS/LAPACK headers, and `ffmpeg` available. On NVIDIA's Ubuntu 24.04 SBSA
repo this includes `cuda-cccl-13-0`, `libcudnn9-dev-cuda-13`, and
`libnccl-dev`:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/package-linux.sh --version 0.20.0
ls dist/linux/
```

CUDA `.deb` packages declare the CUDA 13 runtime/JIT packages they need:
`cuda-cccl-13-0`, `cuda-cudart-13-0`, `cuda-nvrtc-13-0`,
`libcublas-13-0`, `libcufft-13-0`, `libcudnn9-cuda-13`, and `libnccl2`.

For source-only CUDA checks, prepare native artifacts and then export the
environment printed by the script before building:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/prepare-linux-native.sh
swift build --product mere.run
```

The preparation script first lets SwiftPM resolve dependencies under the active
Linux Swift toolchain, then builds the CMake CUDA bridge from that exact
`mlx-swift` checkout. This keeps the bridge aligned when an older Swift
toolchain selects a compatibility revision different from a Mac checkout.
`MLX_SWIFT_CUDA_COMMIT` is available only as a deliberate maintainer diagnostic
override; normal source builds should use the SwiftPM-selected revision.

Do not describe a CUDA configuration as supported unless it has been run on that
matching host.

### Spark CUDA smoke status

The arm64 CUDA smoke path has been validated on NVIDIA GB10/DGX Spark after
installing the package and pulling public managed models. MLX-backed image,
speech, vision, music, SFX, video, embedding, anonymization, and dense Gemma
chat paths are expected to run there with the packaged CUDA setup. The packaged
Linux CUDA build also carries the matching `llama-cli`; `mere.run text code`
uses that subprocess on Linux so GGUF coding models do not share a process with
MLX CUDA. If a future CUDA run fails inside an upstream MLX or llama.cpp kernel,
keep that failure in the release notes or PR description until the exact package
has been rebuilt and rerun on matching hardware.

Quantized MLX paths default to `MERERUN_MLX_CUDA_NATIVE_QUANT=auto`: each
process probes native `quantized_mm` and `GatherQMM` separately, reports the
selected backend on stderr, and falls back to dense compatibility if the linked
runtime rejects a kernel. Run `scripts/e2e_gb10.sh --quant-mode auto` after each
CUDA package rebuild; `native` is the explicit fail-loud validation mode.

## Build from source on Linux

Install the package layer first:

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
```

CUDA package builds also require CMake 3.25 or newer because the CUDA bridge
build follows upstream `mlx-swift`. On Ubuntu 22.04/Jammy images, install a
newer CMake before running the CUDA package path:

```bash
sudo apt-get install -y python3-pip
sudo python3 -m pip install --upgrade "cmake>=3.25,<4"
cmake --version
```

On Linux arm64, older distro Clang packages can shadow Swift's bundled Clang and
fail MLX bf16 header compilation. The Linux scripts probe for this and select a
bf16-capable C++ driver, including a local `clang++` shim backed by Swift's
`clang-17` when needed; if not, set `CXX` to the Swift toolchain `clang++` path
before building.

Then run the headless Linux checks:

```bash
swift build
swift test
./scripts/check-linux.sh
swift run mere.run --help
```

Do not run the macOS app bundle commands on Linux. `mere.run.app`, the SwiftUI
studio, and DMG packaging are macOS-only.

## Verify release assets

When a Linux release includes `SHA256SUMS`, place it beside the matching
tarball or `.deb` before checking it:

```bash
tag=v0.20.0

curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/SHA256SUMS" -o SHA256SUMS
sha256sum -c SHA256SUMS
```

If you only downloaded one asset, `sha256sum -c` will report the missing file as
not found. That is expected; download the matching tarball or `.deb` before
running the full manifest check.
