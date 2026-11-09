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
- `MLXRequestStreams.swift` leases process-wide CPU/GPU execution contexts.
  Completed contexts survive generator replacement; concurrent operations and
  different selected default devices retain separate contexts.

`SmallBatchAffineQMV` retains serial QMV arithmetic while sharing weight reads.
Its bias sum promotes every BF16 input to FP32 before addition, matching MLX's
`load_vector`; adding in BF16 first can lose a cancellation residual.

`SmallBatchAffineQ8` shares packed affine-Q8 reads across two through eight
verification rows, with one, two, or four output rows per SIMD group. Its GPU
fixtures compare directly with native serial Q8 arithmetic, including BF16
activations beyond FP16 range. Callers select eligible projection shapes; a
component win does not establish a full-model speedup.

`M5AffineQ8Head` is an experimental M5 matrix projection for affine Q8/group-64
buffers with BF16 activations and metadata. It reads original packed weights,
stages BF16 tiles, and accumulates in FP32 without a retained dense copy.
One through eight input rows use the same padded matrix geometry. Split counts
of 1, 2, 4, and 8 divide the input dimension; multiple splits reduce FP32 partials
before the BF16 output cast. This arithmetic differs from native serial QMV.
Callers must qualify both quality and generation throughput before selecting it.

`FusedRMSNorm` prepares FP32 learned scales outside the parameter tree, including
an optional centered offset. It normalizes and multiplies in FP32 before casting
back to the input dtype. It retains no model-sized weight reconstruction.

This library depends on MLX and ModelKit. It does not resolve or download models.
Keep path selection and model-specific compatibility mappings in the caller.
Both quantized loading entry points use one array-application implementation.
Preserve array siblings and quantization metadata when replacing leaf modules.
Full-precision module loading completes selected CPU tensor reads per shard
before GPU casts or inference evaluate the parameters. This keeps external
storage I/O out of GPU command-buffer waits and leaves unmapped tensors lazy.

`RoutedMoERouting` owns guarded routed quantization dispatch and kernels. Its
staging, sorted prefill, residual, and gather extensions retain the shared
implementation used by Laguna and other Core model families.

`FusedAttentionPolicy` provides the shared `MERERUN_FUSED_SDPA` diagnostic
switch for native encoders and Core orchestration.
