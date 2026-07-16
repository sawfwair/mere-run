# Quantization

Runtime helpers for loading, writing, and executing quantized MLX weights.

- `QuantizationIO.swift`: typed quantization metadata shared by loaders and manifests.
- `QuantizedModelManifestWriter.swift`: model manifest emission for quantized checkpoints.
- `ResidualQuantizedLinear.swift`: residual-aware quantized projection support.
- `PortableQuantizedMatmul.swift`: CUDA-native quantized matmul/GatherQMM probing
  with a per-operation dense compatibility fallback.

On Linux CUDA, `MERERUN_MLX_CUDA_NATIVE_QUANT` accepts `auto` (the default),
`native`, or `dense`. Automatic mode evaluates each native quantized operation
once and keeps it enabled only when the linked MLX CUDA runtime supports it.
Every process reports the selected backend on stderr as `mlx_cuda_quant ...`.
Use `scripts/e2e_gb10.sh --quant-mode auto` on the target GPU to capture the
selection alongside the real model/artifact sweep.

Keep quantization formats explicit in typed metadata. Do not pass raw checkpoint
dictionaries past the loader boundary, and prefer narrow format-specific shims
when a checkpoint layout does not match upstream MLX primitives.
