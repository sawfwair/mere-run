# Shared KV caches

This library owns the full-attention cache protocol and dynamic, static, and
ragged-batch implementations. It depends on MLX, without model catalogs,
checkpoint downloaders, tokenizers, or inference families.

`MereRunCore` re-exports the cache types. `AudioQwen3ASRModel` uses this library
directly. Both paths share the same implementation and cache identity.

Preserve fork isolation, rollback, row offsets, masks, and batch split semantics.
A batched row must produce the same result as independent decoding. Forks must
use fresh array wrappers because writes rebind an `MLXArray` wrapper in place.
Read a fork after either branch writes when testing isolation.
