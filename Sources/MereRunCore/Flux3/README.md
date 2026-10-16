# FLUX 3 Klein

This directory owns loading and orchestration for the original BF16 distilled
FLUX 3 Image Klein package. Model computation lives in
`Sources/MereRunImageModels/Flux3`.

- `Flux3Resources.swift`: pinned source identity and native package validation.
- `Flux3Conditioner.swift`: Qwen3-VL text-only conditioning, hidden-state indices
  4 through 32 at intervals of four, right padding to multiples of 80, and
  padding positions fixed to 1.
- `Flux3VAE.swift`: original LDM tensor mapping and shared Float32 VAE execution.
- `Flux3ImageIO.swift`: EXIF orientation, Lanczos resize, padding, and cropping.
- `Flux3Generator.swift`: sequential component residency, flow-map sampling,
  cancellation, progress, and PNG output.

The source reference is `black-forest-labs/flux-3-image-klein` at
`45e57c9fb06f9751cdced29a65c5d2171003be05`. The managed weight revision is
`8454017f9ebaada30c998fccd9f67a80c7580f68`.

Run `scripts/validation/flux3-klein-reference.py` in the documented test-only
environment to regenerate tiny upstream numerical fixtures. Checkpoint access,
VAE and text-encoder checkpoint parity, generated-image quality, and peak-memory
qualification remain separate from synthetic transformer parity.

Bounded released-checkpoint execution and visual checks pass 512/1024 generation
and single- and two-reference editing. See the
[dated qualification report](../../../docs/benchmarks/flux3-klein-native-qualification-2026-10-15.md)
for images, exact prompts, output hashes, process footprints, and remaining limits.
