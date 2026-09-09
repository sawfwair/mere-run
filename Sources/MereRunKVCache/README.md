# Shared KV caches

This library owns the full-attention cache protocol and dynamic, static, and
ragged-batch implementations, plus the optional affine quantized cache. It depends on MLX, without model catalogs,
checkpoint downloaders, tokenizers, or inference families.

`MereRunCore` re-exports the cache types. `AudioQwen3ASRModel` uses this library
directly. Both paths share the same implementation and cache identity.

Preserve fork isolation, rollback, row offsets, masks, and batch split semantics.
A batched row must produce the same result as independent decoding. Forks must
use fresh array wrappers because writes rebind an `MLXArray` wrapper in place.
Read a fork after either branch writes when testing isolation.

## Affine quantized caches

`AffineQuantizedKVCache.swift` retains packed keys and values and materializes
attention inputs on demand. Model runtimes select four-bit or eight-bit caches
explicitly. Preserve source dtype, offsets, allocation headroom, and independent
array wrappers when forking or splitting rows.

Before Metal dequantization, make packed arrays, scales, and biases contiguous.
Trimming unused token capacity leaves gaps between heads; the pinned backend
can overwrite bound buffers when it copies those inputs internally.

`DecodeRuntimeTests` covers affine storage, incremental updates, fork content,
and batch/row isolation. Padded and long BF16 Metal fixtures remain opt-in;
CPU fixture success does not qualify those layouts on GPU.
