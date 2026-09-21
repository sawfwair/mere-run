# Qwen Image 2.1 model layers

This directory owns native Swift/MLX computation for Qwen Image 2.1.
`QwenImage21Transformer` implements the single-stream DiT, zero-centered text
RMS normalization, shared modulation, interleaved rotary embeddings, segmented
block-causal attention, and request-owned prefix KV reuse.

`QwenImage21VAE` implements the single-frame RGBA specialization of the residual
Wan-style encoder and decoder. It preserves the first-frame temporal shortcut
semantics. Temporal convolution weights are validated but are inactive for images.

Weights retain the official Diffusers names. Admission checks the complete key
set and tensor shapes before executing either model. Core owns loading, Qwen3-VL
conditioning, tokenization, scheduling, and PNG output.

The reference is the Apache-2.0 Diffusers implementation at commit
`8d3c30bfda9b511c00992f40cff4170a5502814d`. The model weights have separate
Qwen Research License terms. Numerical fixtures establish component contracts;
see the [trained-checkpoint qualification report](../../../docs/benchmarks/qwen-image-21-native-qualification-2026-09-20.md)
for bounded image-quality observations, measured memory, and remaining numerical
limitations.
