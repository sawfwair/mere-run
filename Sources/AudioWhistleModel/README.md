# Whistle native model

This internal target owns the released Whistle encoder and cached decoder,
checkpoint geometry, and the `.cact` vocabulary/filterbank reader. AudioSTT owns
waveform preprocessing, model resolution, windowing, and transcription.

The runtime reads packed CQ2/CQ4 archives or original FP32 safetensors directly. It does not execute
Cactus binaries, Python, or ONNX. It uses all eight encoder layers and 2...8 decoder layers,
four mHC lanes, gated GQA, Monarch Hadamard MLPs, causal QKV convolution taps,
and n-gram engram lookups at decoder layers 3 and 7. Audio cross-K/V projections
are computed once per clip; self-attention and raw QKV taps are cached per token.

`Resources/layout.json` records the tensor shapes from the released checkpoint.
`Resources/permutations.json` records NumPy RandomState permutations at seeds 11
and 13, the upstream Hadamard MLP convention. Configuration and all tensor names,
shapes, and dtypes are validated before computation.

Keep the layer-major checkpoint layout intact. The stem flattens channel before
frequency. Encoder attention is bidirectional. Encoder positions are added after
the final normalization; cross-attention does not apply rotary positions.

See [runtime guide](../../docs/runtime/whistle.md) for scope and qualification.

`Resources/cact-layout.json` maps the pinned archive's nameless records to original
checkpoint names. Its mapping was derived by byte-exact comparison of every raw
or CQ segment against the public Whistle WebGPU pack and checked with independent
NumPy dequantization and graph outputs. Packed matmuls rotate activations into the
128-wide Walsh-Hadamard basis and reduce codebook indices with their FP16 norms;
embedding lookups expand only selected rows. FP32 activation/KV arithmetic is
retained. The managed manifest labels the mixed archive `int4` for its widest CQ
weight width; most matrix weights are CQ2.

Linux uses portable MLX operations for CQ projection and selected-row lookup.
The Metal kernels remain the Apple path; neither path expands the full Engram table.
