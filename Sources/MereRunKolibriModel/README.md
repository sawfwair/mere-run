# Native Kolibri computation

This internal runtime owns Kolibri-1 configuration, GQA with Q/K norms,
RoPE only in sliding layers, bounded sliding caches, sandwich norms, and
sigmoid MoE routing with logit correction bias and an ungated shared expert.
Core owns checkpoint I/O, tokenization, chat scheduling, and qualification.

The native layout stacks routed expert projections in numeric expert order.
`mererun_quantization` explicitly identifies each packed projection's bits and
group size. Routers and embeddings stay dense. Official FP8 weights must be
converted before loading; they are never silently interpreted as affine Q8.

Architecture source: Aleph-Alpha/aleph-alpha-inference at
`049a6a7bd2405b27d6d280d256bd3d585191c7ae` (Apache-2.0).
