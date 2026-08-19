---
license: other
license_name: minimax-h3-community-license
license_link: LICENSE
library_name: mlx
pipeline_tag: text-to-video
base_model: MiniMaxAI/MiniMax-H3
tags:
  - mlx
  - minimax-h3
  - audio-video
  - mere-run
---

# MiniMax-H3 FL2VA MLX compact BF16

This is Mere's compact maximum-fidelity MLX runtime artifact for MiniMax-H3
FL2VA. Every weight input came from the pinned official checkpoint
`MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027`; no
third-party converted or quantized checkpoint was used.

The 20.11B-parameter active denoising core remains BF16. The Qwen3-VL
conditioner uses MLX affine INT8/group-64, the video VAE uses FP16, and the
audio VAE remains FP32. Schedule-only AdaLN projections, the timestep MLP, and
reconstructed RoPE tensors are omitted after generating a source-bound pack of
exact 5, 9, 12, 16, 21, and 31-point tables at video/audio shifts 12/3 plus the
LightX2V 5-point 6/3 table.

The cache pack was evaluated by `mere.run model optimize` on MLX Metal and is
bound to the official projection-tensor closure plus bit-identical 9- and
21-point real-generation receipts. The denoising core was reproduced on MLX
CUDA; CUDA-evaluated AdaLN tables are not accepted because backend BF16
reduction order is not bit-identical to Apple Silicon.

`SOURCE_MANIFEST.json`, `transformer.conversion.json`, `MODIFICATIONS.md`, and
`SHA256SUMS` provide the conversion and byte provenance. The artifact is
intended for `video-minimax-h3-fl2va-bf16-mlx` in mere.run.

## License

These weights are governed by the MiniMax H3 Community License Agreement in
`LICENSE`, not the mere.run source license. At the pinned source revision the
license excludes use, distribution, and display in the United States, European
Union, United Kingdom, and Republic of Korea and carries additional notice and
safeguard obligations. Review the complete license before downloading or using
the artifact.
