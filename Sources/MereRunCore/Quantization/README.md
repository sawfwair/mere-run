# Quantization

Runtime helpers for loading, writing, and executing quantized MLX weights.

- `QuantizationIO.swift`: typed quantization metadata shared by loaders and manifests.
- `QuantizedModelManifestWriter.swift`: model manifest emission for quantized checkpoints.
- `ResidualQuantizedLinear.swift`: residual-aware quantized projection support.
- `PrismBinaryQuantizedLinear.swift`: Bonsai-compatible packed 1-bit affine projection support.

Keep quantization formats explicit in typed metadata. Do not pass raw checkpoint
dictionaries past the loader boundary, and prefer narrow format-specific shims
when a checkpoint layout does not match upstream MLX primitives.
