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
- `DFlash2.swift` owns grouped dynamic causal convolution and sparse candidate
  selection shared by the Qwen and Muse Glimmer draft models.

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

The internal eight-row experiment can share each weight read across all eight
inputs with one, two, or four output rows per SIMD group. The default uses
two groups of four inputs. Set `MERERUN_Q38_EXACT_Q4_HEAD_OUTPUT_ROWS=4` to try
eight-input sharing only for the Qwen 27B vocabulary-head geometry
(`K=5120`, `N=248320`, eight input rows); `2` and `1` select smaller output tiles,
and `0` restores the default. Other geometries retain their existing dispatch.
`SmallBatchAffineQMVTests` checks the experimental
layouts against serial QMV. Its opt-in component benchmark rotates layout order
and reports synchronized wall times for Qwen-sized synthetic projections; set
`MERERUN_TEST_MLX_DEVICE=gpu` and `MERERUN_TEST_Q4_COMPONENT_BENCHMARK=1`.
These component results do not establish full-model decode throughput.

`M5AffineQ4Verification` is an experimental Metal 4 matrix-accelerator primitive
selected per projection by `QuantizedVerificationBackend.m5Matrix`. It reads existing affine Q4/group-64 buffers,
stages FP16 weight tiles, overlaps the next packed tile's loads with the matrix
operation, and keeps FP32 partial sums. One through eight input rows use the
same eight-row matrix geometry. Split counts are explicit until calibrated on
the selected GPU. Its FP16 weight rounding and matrix reduction define a new
arithmetic reference, which requires numerical and serial-versus-verification
qualification rather than equality with scalar QMV. `M5AffineQ4VerificationTests`
checks that reference and offers an opt-in projection sweep through
`MERERUN_TEST_M5_Q4_COMPONENT_BENCHMARK=1`.

The `SwiGLU` extension combines gate/up projection and the compiled activation's
BF16 operation boundaries in one dispatch without concatenating weights. Its
stable sigmoid follows MLX's BF16 math. `PortableQuantizedLinear` uses this path
only for compatible target projections under the runtime policy; unsupported
shapes and devices fall back to the existing kernels. Fused attention projections
carry the source backend policy and invalidate when it changes. Core selects the
target policy at load time; independent draft modules keep the reference policy.

`M5TiledAffineQ4` is the default layout under that target policy;
`MERERUN_Q38_M5_TILED_WEIGHTS=0` selects the earlier row-major path. It prepares contiguous
tile-major codes and metadata from immutable inference parameters, uses
32-channel stages, and interleaves gate/up matrix operations at each K step.
Packed buffers follow the projection's lifetime and retain original parameters
for prefill and fallback. This increases resident memory and adds preparation
work. `M5TiledAffineQ4Tests` checks dense-reference error, serial/verification
agreement, and exact interleaved-versus-separate SwiGLU arithmetic. Full-model
throughput qualification remains separate from these numerical tests.

`QuantizedVerificationContext.suppressM5` is task-local. Core uses it to retain
reference projections for request types outside the automatic DFlash route.
The stored backend and fused projection identities remain unchanged, so
concurrent requests do not change each other's arithmetic policy.

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

`M5TiledAffineQ8` keeps losslessly reordered affine-Q8 codes and metadata alongside
original parameters. Its matrix kernel stages 32 channels in BF16 and uses FP32
accumulation, with optional FP32 partial reduction. An interleaved gate/up kernel
preserves separate BF16 projection and activation boundaries. Packing costs
resident memory and startup time; callers own its lifetime, parameter identity,
model selection, and generation-level qualification.
