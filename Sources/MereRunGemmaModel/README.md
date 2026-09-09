# Gemma model library

Use this library when you change Gemma 4 model computation. Core owns resources,
checkpoint mapping, tokenizer templates, request scheduling, and LoRA training.
The library depends on `MereRunTensor`, `MereRunDecode`, and MLX. Core re-exports
its public types through `GemmaRuntimeExports.swift`.

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

Proportional RoPE uses the full head dimension to compute frequencies. Shared
KV layers reuse the correct preceding attention type. Keep key-equals-value,
per-layer inputs, RMS scales, softcapping, and MoE routing arithmetic unchanged.

Fusion and compiled segments must invalidate when source modules or captured
parameters change, including LoRA injection and requantization. Fused decode
kernels and compiled segments retain their explicit opt-in policies.

MTP reads target KV state without appending to it. Core verifies each emitted
token against the target model and restores a retained prefix after rejection.
Do not move installed-model resolution or tokenizer dependencies into this target.

## Validation

Run `swift build --target MereRunGemmaModel`, then
`swift test --filter GemmaRuntimeTests`. The isolated tests exercise synthetic
models, cache isolation, batching, quantization, target verification, and draft
state. Core tests retain adapter, template, and generation-policy coverage.
Run `./scripts/check.sh` before opening a PR. CPU fixtures and Linux compilation
do not qualify published checkpoints or Metal performance.
