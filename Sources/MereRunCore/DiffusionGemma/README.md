# DiffusionGemma

This module describes the native MLX text-generation runtime for contributors
working on DiffusionGemma checkpoints.
DiffusionGemma uses a causal encoder to cache the prompt and completed blocks,
then denoises a bidirectional token canvas in parallel. It is not compatible
with the autoregressive Gemma4 runtime.

The managed checkpoint is the revision-pinned OptiQ 4-bit conversion. Its
ordinary projections and shared embedding are loaded through mere.run's
portable affine quantization path. Routed experts keep their fused
`gate_up_proj` and `down_proj` layout and use gather-quantized matrix
multiplication directly.

The runtime supports text generation. The checkpoint's separate BF16 vision
sidecar is downloaded and validated, but the runtime rejects image input
because that path has no real-checkpoint qualification.
