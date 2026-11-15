# Gemma model library

Use this library when you change Gemma 4 model computation. Core owns resources,
checkpoint mapping, tokenizer templates, request scheduling, and LoRA training.
The library depends on `MereRunTensor`, `MereRunDecode`, and MLX. Core re-exports
its public types through `GemmaRuntimeExports.swift`.

`EmbeddingGemma2Config.swift` and `EmbeddingGemma2TextModel.swift` own the
EmbeddingGemma 2 bidirectional text encoder. It uses projection-only per-layer
inputs, independent local/global attention, and the trained 512-to-768 projection.
It does not use causal masks, generation KV caches, or token-identity PLE tables.
Core's `Embeddings/` directory owns task formatting, tokenization, loading, and
bounded batching. `EmbeddingGemma2VisionModel.swift` owns the full bidirectional
Gemma 4 vision encoder with axial RoPE and spatial pooling;
`EmbeddingGemma2AudioModel.swift` owns the clipped conformer, relative chunked
attention, stride-four subsampling, and causal convolution. Their typed media
configs and original tensor shape validation live in this target. These towers
use the EmbeddingGemma 2 projections and are separate from the chat patch
embedder. See [the user guide](../../docs/runtime/embeddinggemma2.md).

## Reading order

- `Gemma4Config.swift`: typed text and unified-model configuration.
- `Gemma4Model.swift`: causal-model protocol, forward results, and RMS normalization.
- `Gemma4RoPE.swift`, `Gemma4Attention.swift`, `Gemma4MLP.swift`,
  `Gemma4Experts.swift`, and `Gemma4DecoderLayer.swift`: model stages.
- `Gemma4LanguageModel.swift`: embeddings, per-layer inputs, shared KV routing,
  hidden-state capture, and logits.
- `Gemma4TextCausalLM.swift`, `Gemma4VisionEmbeddings.swift`, and
  `Gemma4UnifiedCausalLM.swift`: text and multimodal forward paths.
- `Gemma4AttentionCache.swift`, `Gemma4FullKVCache.swift`, and
  `Gemma4SlidingKVCache.swift`: cache protocols, allocation, forks, and batching.
- `Gemma4ForwardAttentionCache.swift`: the query position and attention context
  retained for one forward call, including shared layers.
- `Gemma4KVQuantization.swift`: quantization configuration and defaults.
  `Gemma4QuantizedTensorState.swift` and `Gemma4PolarTensorState.swift` retain
  packed storage. Cache `+Attention` extensions own specialized attention.
- `Gemma4AffineFastKernels.swift`, `Gemma4PolarFastKernels.swift`, and
  `Gemma4DecodeFusedKernels.swift`: optional Metal kernels.
- `Gemma4FusedProjections.swift`: fusion policy and compiled-segment identities.
- `Gemma4MTP.swift`: assistant configuration and draft computation. Draft
  sampling uses `MereRunDecode`; Core loads the assistant and verifies drafts.

## Preserve these contracts

Full caches grow without dropping valid prompt rows. Sliding caches return
chronological state for multi-token attention and may return storage order for
single-token decode. A multi-token append must retain the preceding window for
early queries, even when resident storage advances to the next window.

Each forward call retains its starting cache offsets. Shared layers use those
offsets after a producer advances resident storage. For multi-token queries,
the forward cache view retains the producer's complete attention context until
all shared layers consume it. Single-token specialized attention does not
materialize dense quantized state.

Cache forks isolate subsequent mutations. Batched caches preserve row identity,
offsets, quantization settings, and valid token counts. Re-encoding keeps the
total position offset. Quantized snapshots exclude spare allocation capacity.
Make packed weights, scales, and biases contiguous before affine dequantization.
Trimming capacity or selecting an interior token range can leave gaps between
heads; the pinned Metal dequantizer can overwrite input bindings when it copies
a strided bias array.

Proportional RoPE uses the full head dimension to compute frequencies. Shared
KV layers reuse the correct preceding attention type. Keep key-equals-value,
per-layer inputs, RMS scales, softcapping, and MoE routing arithmetic unchanged.

Fusion and compiled segments must invalidate when source modules or captured
parameters change, including LoRA injection and requantization. Fused decode
kernels and compiled segments retain their explicit opt-in policies.

MTP reads target KV state without appending to it. Core verifies each emitted
token against the target model and restores a retained prefix after rejection.
Do not move installed-model resolution or tokenizer dependencies into this target.

`Gemma4TextCausalLM+Embeddings.swift` supplies external audio embeddings,
selected post-layer hidden states, raw token embeddings, and the tied output
head for SOV-Transcribe-1 and its DFlash companion. FP32 router checkpoints cast
router input to the router parameter dtype; expert outputs retain input dtype.

SOV's benchmark can repack loaded MoE experts to affine Q8 with fused gate/up
weights, or select BF16 GEMV for verification blocks of at most 16 tokens.
Preparation replaces expert modules before decoding; it does not cache source
parameter references. Routers, dense layers, attention, and output heads retain
their original parameters. These policies are explicit experiments and do not
change other Gemma runtimes or the default SOV path.
`preparePackedExpertWeights` installs lazy affine packed slots before loading.
It accepts fused gate/up codes, scales, and biases directly, avoiding a resident
BF16 copy during streamed conversion or native checkpoint reuse. Dispatch is
selected independently of the serialized quantization format.

The same packing API accepts Q4/Q6/Q8 schedules indexed by decoder layer.
Precision schedules are validated before module replacement; lower-bit layers
retain MLX arithmetic, and eligible Q8 layers can retain custom Metal dispatch.
The SOV `q6-matrix` experiment expands six-bit codes into eight-bit lanes once
during preparation, sharing Q6 scales and biases. Large sorted prefill batches
use this cached encoding; verification reads compact Q6 weights. The encoding
contains the same quantized values and refreshes after weight replacement.
It is private derived storage, not a checkpoint parameter or another quantizer.

`Gemma4MetalExperts` supplies the explicit `q8-metal` experiment. Its shared
Metal shader specializes affine Q8 GEMV and reuses each weight load across two
adjacent sorted routes for the same expert. It preserves the reference FP32
group sums and BF16 output rounding. Small, sparse BF16-activation batches use
the shader; larger or denser batches retain MLX's matrix path. Kernel arguments
read the current packed parameters on every call. The resource and wrapper
belong to this runtime target; installed-model policy stays in AudioSTT.
The paired benchmark switches dispatch without replacing packed parameters.
The target retains its compiled shader across switches; tests verify packed
weight identity, reduction tails, expert boundaries, and matrix fallback.
The byte-vector specialization reinterprets each packed word as four unsigned
bytes before conversion to FP32. Its multiply and reduction order is unchanged.
The optimized layout computes eight columns per SIMD group when the output
width permits it, and dispatches routes before output tiles. Adjacent workgroups
can reuse an expert's weight tile without changing the per-output arithmetic.
The benchmark's `--metal-baseline byte` selects the prior four-column layout;
`--metal-baseline reuse` also restores shift-based unpacking. Both controls use
the same resident parameters.

The opt-in SOV `--experimental-expert-fusion` experiment combines the sorted Q8
gate/up projection with GELU and multiplication. It rounds the projection and
each activation intermediate to BF16 in the native MLX order. Unsupported
shapes and dense prefill retain the existing path. `--metal-baseline layout`
compares fusion against the current optimized layout using the same weights.
Fusion remains disabled by default.

Eligible Q8 Metal projections automatically distribute the 16 output reductions
and stores across SIMD lanes on the qualified Apple M5 Max (`applegpu_g17s`).
Both device name and architecture must match. Other devices and four-column
projections retain the existing reduction. A nil dispatch override selects this
device default; explicit true/false overrides remain available for research.
The SOV benchmark's `--experimental-expert-reduction` flag forces the optimized
reduction, and `--metal-baseline layout` pins the original SIMD reduction as a
paired control. Measurement labels record the selected policy even without the
force flag. Automatic selection does not change expert quantization.
Sixteen retained policy identities keep compiled MoE switches distinct.

## Validation

Run `swift build --target MereRunGemmaModel`, then
`swift test --filter GemmaRuntimeTests`. The isolated tests exercise synthetic
models, cache isolation, batching, quantization, target verification, and draft
state. Core tests retain adapter, template, and generation-policy coverage.
Run `./scripts/check.sh` before opening a PR. CPU fixtures and Linux compilation
do not qualify published checkpoints or Metal performance.

SOV's separate attention experiment shares identical masks within a single
forward call and replaces proportional partial RoPE slicing and concatenation
with a full-width native RoPE call. Infinite frequencies give the untouched
parts of each half zero angular velocity. Position, window, context length and
dtype identify cached masks; padding and vision masks bypass reuse. The cache
expires with the forward call. `setExperimentalAttention` switches these paths
without changing parameters. Other runtimes retain their existing behavior.

`Gemma4HeadNormRoPE` is a separate SOV MLX experiment for small BF16 verification
blocks. Its Metal resource reproduces the native RMS reduction, BF16
normalization rounding, BF16 learned scaling, and rotary half layout in one
dispatch. Parameters are inputs to every call. Larger prefill and unsupported
head sizes or dtypes retain native operations. A target caches each layer's
compiled wrapper across benchmark switches; shared forward masks also retain
the dynamic integer position scalar for that call. The feature remains off by
default and does not replace ordinary Gemma decode kernels.

The fused head wrapper owns Cmlx kernel/configuration handles and caches
immutable launch metadata by batch and query length. Every call still passes
current projections, normalization weights and the dynamic position tensor.
Configuration reuse retains no model parameters. The optional
`SOVHeadDispatchProfileTests` diagnostic isolates host graph construction from
GPU execution.

The SOV `compiled` experiment compiles each MoE layer's complete segment after
attention. Normalization, dense MLP, routing, sparse experts, residuals, and layer
scaling share one MLX graph. Module parameters are observed runtime inputs and
the scalar is an explicit argument. Module replacements and Metal expert dispatch
changes invalidate the segment. This policy is separate from the older dense
single-token compiled segments and remains off in ordinary inference.
