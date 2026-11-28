# IDOL orchestration

`IdolGenerator` runs local IDOL checkpoints through the native MLX runtime in
`MereRunImageModels/Idol`. The public command is `vision image-to-3d-idol`.

- `IdolResources` verifies the four released IDOL and BiRefNet weight hashes and records
  configuration and encoder hashes for the run manifest.
- `IdolConditioner` loads local DINOv3 H+ and FLUX.2 VAE weights. It samples
  the VAE posterior, applies packed BatchNorm, and prefixes five zero tokens.
- `IdolBackgroundRemover` runs the native Swin-L BiRefNet and transfers its
  alpha map back to the resized source image.
- `IdolPreprocessor` resizes the short side before background removal, erodes
  alpha, crops according to synthetic or real mode, and composites on black.
  `IdolImageResize` implements Pillow's fixed-point Lanczos RGB/RGBA resampling.
- `IdolGenerator` snapshots input bytes, releases image encoders before flow
  sampling, and publishes a staged directory after hashing its artifacts.

Opaque images use BiRefNet automatically. Transparent alpha bypasses inference;
`--already-framed` bypasses foreground detection and cropping. The command uses
local encoder directories because DINOv3 and FLUX.2-dev require separately
accepted access terms. It does not download gated assets or offer a managed
model ID. `--dry-run` checks checkpoint bytes and the bounded image header
without running background removal or inference.

See `docs/qualification/idol-native-20261030/README.md` for real-checkpoint
qualification, numerical metrics, and rendered outputs.
