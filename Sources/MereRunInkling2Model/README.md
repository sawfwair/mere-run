# Native Inkling 2 computation

This target owns the Inkling 2 Mini and Small decoders, vision transformer,
checkpoint reader, and original NVFP4/MXFP8 expert decoding. Core owns
tokenization, media loading, generation, and model discovery.

Mini restarts each pass through the middle layers from the same prelude.
Previous-pass activations provide messages through learned low-rank
controllers. Each pass holds its own attention and convolution cache.

On macOS, Mini keeps the original packed expert tensors resident and gathers
the routed experts in Metal matmuls. The fused short convolution reuses the
same cache layout and tap order as the reference.

The reference path retains ordinary weights in memory and expands selected
experts from disk one at a time. Both paths use BF16 expert activations, FP32
routing, and FP32 accumulation.

Mini's DSpark path mixes features from visited layers to build the
draft attention cache. The target verifies the proposed block in one pass.
Only the accepted prefix advances the target attention, short convolution,
and draft caches.

See [Native Inkling 2](../../docs/runtime/inkling2.md) for the reference
implementations and checkpoint revisions.
