# Qwen Image 2.1 model layers

This directory owns shared native Swift/MLX computation for Qwen Image 2.1
and Qwen Image 2.1 Turbo. Both checkpoints use the same layer and tensor schemas;
Core selects the checkpoint and its sampling recipe.
`QwenImage21Transformer` implements the single-stream DiT, zero-centered text
RMS normalization, shared modulation, interleaved rotary embeddings, segmented
block-causal attention, and request-owned prefix KV reuse.

`QwenImage21VAE` implements the single-frame RGBA specialization of the residual
Wan-style encoder and decoder. It preserves the first-frame temporal shortcut
semantics. Temporal convolution weights are validated but are inactive for images.

Weights retain the official Diffusers names. Admission checks the complete key
set and tensor shapes before executing either model. Core owns loading, Qwen3-VL
conditioning, tokenization, scheduling, and PNG output.

`QwenImage21Quantization` validates optional component-local affine Q4/Q8
group-64 packing against logical tensor shapes. Transformer weights stay packed
through MLX quantized matrix multiplication; layers without scales remain dense.
The VAE retains its original dense format. Packed computation is checked against
explicit dequantization separately from trained-checkpoint quality qualification.

The reference is the Apache-2.0 Diffusers implementation at commit
`8d3c30bfda9b511c00992f40cff4170a5502814d`. The model weights have separate
Qwen Research License terms. Numerical fixtures establish component contracts;
see the [trained-checkpoint qualification report](../../../docs/benchmarks/qwen-image-21-native-qualification-2026-09-20.md)
for bounded image-quality observations, measured memory, and remaining numerical
limitations.
