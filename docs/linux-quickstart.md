# Linux QuickStart

This page is for installing the released headless `mere.run` CLI on Linux.
The current supported release path is x86_64/amd64 Linux packages published on
GitHub Releases.

The macOS path remains the primary hands-on development and runtime validation
environment for this repo. The Linux release path is real, but intentionally
narrow: CLI-only packages, Ubuntu-style hosts, CPU-oriented CI fixtures, and
optional local CUDA experiments.

## Current validation boundary

- Linux release packages are built for x86_64/amd64 hosts.
- The release workflow validates the portable tarball, Debian package, runtime
  library bundling, and package manifests on Ubuntu in GitHub Actions.
- Linux packages do not include `MereRun.app`, the SwiftUI studio, the macOS
  installer UI, or the DMG layout.
- Linux arm64 release packages are blocked for now by upstream `mlx-swift`
  Linux `bf16` support.
- CUDA is optional local acceleration work, not part of the default pull-request
  or release gate.
- Current x86 CUDA validation should be treated as limited to available hosts
  with up to 16 GB VRAM. Larger x86 CUDA systems and DGX-class machines are not
  claimed as tested until they are available for direct validation.

## Install with apt

Use this path on Debian or Ubuntu-style systems:

```bash
tag=v0.8.0
version="${tag#v}"

curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/mere-run_${version}_amd64.deb" -o mere-run.deb
sudo apt install ./mere-run.deb
mere.run --version
mere.run status
```

The Debian package installs the `mere.run` command and the colocated runtime
assets that the CLI needs at runtime.

## Install the portable tarball

Use this path when you do not want to install a Debian package:

```bash
tag=v0.8.0

curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/mere-run-${tag}-linux-x86_64.tar.gz" -o mere-run-linux.tar.gz
tar -xzf mere-run-linux.tar.gz
cd "mere-run-${tag}-linux-x86_64"
./install.sh
mere.run --version
mere.run status
```

The tarball installer copies the CLI and its runtime assets together. The
packaged launcher resolves its own install location and sets `LD_LIBRARY_PATH`
for the bundled runtime libraries before it starts the real `mere.run` binary.

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

## CUDA notes

CUDA is not required for the Linux release package quickstart. The shared Linux
CI path stays CPU-oriented and fixture-sized so package validation remains
repeatable.

For source builds that intentionally exercise the optional Linux CUDA bridge,
prepare the native artifacts and then export the environment printed by the
script:

```bash
MERERUN_LINUX_ACCEL=cuda scripts/prepare-linux-native.sh
# Export the printed PKG_CONFIG_PATH, LIBRARY_PATH, LD_LIBRARY_PATH,
# MERERUN_MLX_SWIFT_LINKAGE, MERERUN_MLX_SWIFT_BUILD_DIR,
# MERERUN_MLX_SWIFT_SOURCE_DIR, and MERERUN_MLX_SWIFT_LINK_FLAGS values.
swift build --target mere.run
```

Do not describe a CUDA configuration as supported unless it has been run on a
real matching host. Today that means x86 CUDA coverage is limited to available
machines with up to 16 GB VRAM.

## Build from source on Linux

Install the package layer first:

```bash
sudo apt-get update
sudo apt-get install -y clang cmake ninja-build pkg-config gfortran libcurl4-openssl-dev zlib1g-dev libopenblas-dev liblapacke-dev ffmpeg gzip unzip zip
```

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
tag=v0.8.0

curl -L "https://github.com/sawfwair/mere-run/releases/download/${tag}/SHA256SUMS" -o SHA256SUMS
sha256sum -c SHA256SUMS
```

If you only downloaded one asset, `sha256sum -c` will report the missing file as
not found. That is expected; download the matching tarball or `.deb` before
running the full manifest check.
