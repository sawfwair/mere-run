# MereRunTensor

Checkpoint loading, quantized modules, and shared tensor kernels.

- `ModelWeightsLoader.swift` selects indexed, single-file, or directory-shard loading.
- `HFSafetensorsWeightsLoader.swift` enforces index ownership, maps keys, and
  applies full-precision or quantized parameters.
- `DenseLayer.swift`, `PortableQuantizedMatmul.swift`, and
  `ResidualQuantizedLinear.swift` own shared projection and embedding modules.
- `SmallBatchAffineQMV.swift` and `DynamicSparseAttention.swift` own tensor kernels.
- `SafetensorsStreamingLoader.swift` reads typed headers and selected arrays.
- `FusedQuantizedProjection.swift` and `SmallBatchAffineGatherQMV.swift` share
  quantized projection fusion and expert-route kernels across model families.
- `MLXCheckpoint.swift` owns gradient recomputation.

This library depends on MLX and ModelKit. It does not resolve or download models.
Keep path selection and model-specific compatibility mappings in the caller.
Both quantized loading entry points use one array-application implementation.
Preserve array siblings and quantization metadata when replacing leaf modules.

`RoutedMoERouting` owns guarded routed quantization dispatch and kernels. Its
staging, sorted prefill, residual, and gather extensions retain the shared
implementation used by Laguna and other Core model families.
