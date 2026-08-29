# Shared Decode Runtime

Reusable autoregressive decoding and cache primitives shared by native MLX
model families.

- `AutoregressiveDecodeEngine.swift` owns the pipelined serial decode loop,
  device-side sampling and repetition history, EOS confirmation, and optional
  streamed token emission. Model-specific prompt construction, cache creation,
  and logits transforms remain with each model runtime.
- `AffineQuantizedKVCache.swift` provides the explicitly selected `affine4` and `affine8`
  cache implementation. It keeps keys and values quantized while resident,
  materializes them for attention, and preserves fork, batch, and row-split
  semantics used by serving schedulers.

## Invariants

- Keep sampling tensors and next-token inputs on the MLX device; host readback
  is only for token confirmation and emission.
- Preserve deep steady-state queueing and the final-token boundary fast path
  when changing the decode loop. Confirm the first token before opening the
  deeper pipeline, and do not schedule an unused forward after a completed
  token budget. EOS may discard already-queued speculative work.
- Cache implementations must maintain `[batch, heads, tokens, head-width]`
  layout, matching key/value offsets, source dtype, and independent wrappers
  when forked or split into rows.
- Affine reconstruction makes packed data, scales, and biases contiguous before
  invoking Metal dequantization. Slicing unused token capacity leaves gaps
  between heads; copying those inputs inside the pinned dequantizer can clobber
  already-bound buffers. Cover padded multi-head tails, not only aligned or
  single-head caches, in round-trip and fork-content tests.
- Forks need fresh MLX array wrappers: a same-dtype `asType` returns the original
  object. Use same-shape reshaping to retain the immutable backing arrays while
  isolating subsequent subscript assignments. Read cache state again after
  parent/child mutations when testing this; an earlier array view can hide aliasing.
- Quantized caches are memory-oriented opt-ins. Do not silently replace a
  model's native cache mode; dequantization can trade decode speed for lower
  persistent memory.
- Add focused tests under `Tests/MereRunCoreTests` for decode behavior, cache
  storage, and batch/fork/row lifecycle changes.
