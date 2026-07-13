# MMAudio Runtime

Swift-native MLX runtime for MMAudio text-to-audio and video-to-audio sound
effect generation.

## Managed model

- Canonical id: `sfx-mmaudio-large-44k-v2`
- Architecture source: `hkchengrex/MMAudio` at the pinned revision in
  `MMAudioResources.swift`
- Converted MLX-readable weights: `Kijai/MMAudio_safetensors`
- CLIP tokenizer/config: `apple/DFN5B-CLIP-ViT-H-14-378`
- Vocoder: `nvidia/bigvgan_v2_44khz_128band_512x`
- Serving engine: `mmaudio`
- Checkpoint license: CC-BY-NC-4.0 (non-commercial)

The architecture source is MIT-licensed. That does not change the separate
non-commercial license on the published MMAudio checkpoints. The mounted Apple
DFN5B CLIP model is separately research-only under the Apple Machine Learning
Research Model License Agreement. NVIDIA BigVGAN-v2 is MIT-licensed. Exact
component license files are installed beside the managed assets.

## Architecture

- `MMAudioResources.swift`: pinned sources, model dimensions, sequence-length
  contract, managed-root validation, and generation defaults.
- `MMAudioCLIP.swift`: DFN5B CLIP ViT-H/14 image and text towers.
- `MMAudioConditioner.swift`: text, CLIP-frame, and Synchformer conditioning.
- `MMAudioLayers.swift`, `MMAudioTransformer.swift`, and
  `MMAudioNetwork.swift`: the large-v2 joint/fused MMDiT and Euler flow sampler.
- `MMAudioVAE.swift`: 44.1 kHz magnitude-preserving latent decoder.
- `MMAudioBigVGAN.swift`: native BigVGAN-v2 generator and alias-free
  activations.
- `MMAudioBigVGANCheckpoint.swift`: restricted official PyTorch checkpoint
  mapping; it does not execute Python or arbitrary pickle globals.
- `MMAudioGenerator.swift`: staged conditioning, sampling, decoding, and PCM
  output with MLX cache release between large components.

The public surface remains under `mere.run sfx`. Keep stdout limited to the
output path and send progress or diagnostics to stderr.
