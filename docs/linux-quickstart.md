# Linux QuickStart

This page is for installing the released headless `mere.run` CLI on Linux.
The default published release path is x86_64/amd64 Linux packages on GitHub
Releases. Linux arm64 packages are CUDA-only and must be built on a real arm64
CUDA host or self-hosted runner.

The macOS path remains the primary hands-on development and runtime validation
environment for this repo. The Linux release path is real, but intentionally
narrow: CLI-only packages, Ubuntu-style hosts, CPU-oriented x86 CI fixtures,
and CUDA-gated arm64 package work.

## Current validation boundary

- Published default Linux release packages are built for x86_64/amd64 hosts.
- The release workflow validates the portable tarball, Debian package, runtime
  library bundling, and package manifests on Ubuntu x86_64 in GitHub Actions.
- Linux packages do not include `MereRun.app`, the SwiftUI studio, the macOS
  installer UI, or the DMG layout.
- Linux arm64 package builds must use `MERERUN_LINUX_ACCEL=cuda`; CPU arm64
  packages are only local smoke-test artifacts.
- The optional arm64 CUDA workflow lane targets self-hosted runners labeled
  `self-hosted`, `linux`, `arm64`, and `cuda`.
- Current CUDA validation should be treated as limited to the exact hosts that
  have run the CUDA package and smoke path.

## Install with apt

Use this path on Debian or Ubuntu-style systems:

```bash
tag=v0.13.0
version="${tag#v}"
case "$(uname -m)" in
  x86_64|amd64) deb_arch=amd64 ;;
  *) echo "use the arm64 CUDA package path for Linux arm64" >&2; exit 1 ;;
esac

curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/mere-run_${version}_${deb_arch}.deb" -o mere-run.deb
sudo apt install ./mere-run.deb
mere.run --version
mere.run status
```

The Debian package installs the `mere.run` command and the colocated runtime
assets that the CLI needs at runtime.

## Install the portable tarball

Use this path when you do not want to install a Debian package:

```bash
tag=v0.13.0
case "$(uname -m)" in
  x86_64|amd64) linux_arch=x86_64 ;;
  *) echo "use the arm64 CUDA package path for Linux arm64" >&2; exit 1 ;;
esac

curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/mere-run-${tag}-linux-${linux_arch}.tar.gz" -o mere-run-linux.tar.gz
tar -xzf mere-run-linux.tar.gz
cd "mere-run-${tag}-linux-${linux_arch}"
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

## Linux arm64 CUDA package path

Linux arm64 is only useful for this project when the CUDA lane works. Do not
publish or recommend CPU-only arm64 packages as release artifacts.

Build arm64 packages on a native arm64 Linux host with an NVIDIA GPU, driver,
CUDA Toolkit, `nvcc`, CUDA CCCL headers, cuDNN, NCCL, Swift,
OpenBLAS/LAPACK headers, and `ffmpeg` available. On NVIDIA's Ubuntu 24.04 SBSA
repo this includes `cuda-cccl-13-0`, `libcudnn9-dev-cuda-13`, and
`libnccl-dev`:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/package-linux.sh --version 0.13.0
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

The GitHub-hosted `ubuntu-22.04-arm` runner is CPU-only for this purpose. The
arm64 CUDA workflow lane is intentionally self-hosted so the package build runs
against a real CUDA-capable arm64 machine.

Do not describe a CUDA configuration as supported unless it has been run on that
matching host.

### Spark CUDA smoke status

The current arm64 CUDA smoke pass has been validated on an NVIDIA GB10 Spark
host after installing the package and pulling public managed models. MLX-backed
image, speech, vision, music, video, embedding, anonymization, and dense Gemma
chat paths are expected to run there with the packaged CUDA setup. The packaged
Linux CUDA build also carries the matching `llama-cli`; `mere.run text code`
uses that subprocess on Linux so GGUF coding models do not share a process with
MLX CUDA. If a future CUDA run fails inside an upstream MLX or llama.cpp kernel,
keep that failure in the release notes or PR description until the exact package
has been rebuilt and rerun on matching hardware.

## Build from source on Linux

Install the package layer first:

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
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

Each Linux release includes `SHA256SUMS` beside the tarball and `.deb`:

```bash
tag=v0.13.0

curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/SHA256SUMS" -o SHA256SUMS
sha256sum -c SHA256SUMS
```

If you only downloaded one asset, `sha256sum -c` will report the missing file as
not found. That is expected; download the matching tarball or `.deb` before
running the full manifest check.
