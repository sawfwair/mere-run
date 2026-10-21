# Native Flint computation

This target owns Flint attention with learned sinks, absolute-position sliding
caches, pre-norm decoder blocks, dense SwiGLU layers, and sigmoid MoE routing.
Core owns checkpoint loading, tokenization, generation, and model registration.

The architecture follows RecursiveSI/flint at
`2a247be0c0aec895ccb0ac7ca92229ec6df37288` and recursive-oss/rec at
`e53663b262af4cb1ca498e596ca8b1a2d79e0800` (Apache-2.0).
BF16 tensors load from their original names; routed projections stack in numeric
expert order. The published MLX INT4 export uses stacked `switch_mlp` tensors,
group-64 affine projections, and INT8 embeddings. Learned sinks and routers stay
dense. NVFP4 and GGUF files are separate formats and are not accepted here.
