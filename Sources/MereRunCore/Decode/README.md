# Shared Decode Runtime

Reusable autoregressive decoding and cache primitives shared by native MLX
model families.

- `AutoregressiveDecodeEngine.swift` owns the pipelined serial decode loop,
  device-side sampling and repetition history, EOS confirmation, and optional
  streamed token emission. Model-specific prompt construction, cache creation,
  and logits transforms remain with each model runtime.
- `AffineQuantizedKVCache.swift` provides the explicitly selected `affine8`
  cache implementation. It keeps keys and values quantized while resident,
  materializes them for attention, and preserves fork, batch, and row-split
  semantics used by serving schedulers.

## Invariants

- Keep sampling tensors and next-token inputs on the MLX device; host readback
  is only for token confirmation and emission.
- Preserve the one-step pipeline and final-token fast path when changing the
  decode loop. EOS may discard one already-scheduled speculative forward, but
  a completed token budget must not schedule an unused forward.
- Cache implementations must maintain `[batch, heads, tokens, head-width]`
  layout, matching key/value offsets, source dtype, and independent wrappers
  when forked or split into rows.
- Quantized caches are memory-oriented opt-ins. Do not silently replace a
  model's native cache mode; dequantization can trade decode speed for lower
  persistent memory.
- Add focused tests under `Tests/MereRunCoreTests` for decode behavior, cache
  storage, and batch/fork/row lifecycle changes.
