# LLMGeneration

Shared MLX text-generation helpers used by ZImage text encoders and native chat
engines that reuse the same primitives.

- `KVCache.swift` owns the reusable full-attention cache protocol and
  implementations. `KVCacheSimple` stores one scalar-offset row or equal-offset
  batches; `KVRaggedBatchCache` stores padded multi-row batches with per-row
  valid lengths so Qwen-family engines can prove variable-position decode
  batching without treating padding as real KV state.
- `QwenGeneration.swift` and `Sampling.swift` keep the legacy Qwen-compatible
  decode helpers and token sampling utilities.

When adding a cache implementation, keep the protocol typed and split/fork
semantics exact: a batched decode row must produce the same logits as the same
row decoded independently.
Cache forks use fresh array wrappers, not no-op dtype casts (which return the
same object). Test fresh cache reads after either branch writes, rather than
only comparing array views captured before those writes.
