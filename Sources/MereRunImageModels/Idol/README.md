# Native IDOL computation

IDOL converts DINOv3 and FLUX.2 VAE image tokens into object Gaussian splats.
This directory owns the denoiser, learned RePo rotations, two decoders,
flow integration, octree sampling, Gaussian activations, and PLY serialization.
`BiRefNet*.swift` owns Swin-L shifted-window attention, multiscale fusion,
modulated deformable convolution, and the gradient-gated ASPP decoder. Its
7x7 im2col queries are chunked to bound temporary memory. It accepts the
publisher's FP16 checkpoint without conversion. Core owns checkpoint discovery,
image preprocessing, and artifact publication.

`IdolWeights.swift` checks the original FP32 tensor names and shapes before
casting transformer projections to FP16. Embeddings, rotary maps, norms,
and output heads retain FP32 arithmetic. Attention uses MLX SDPA.

`DINOv3Config.swift` and `DINOv3Encoder.swift` are shared with Z-Image-i2L
and TRELLIS.2. Their public names remain available through Core's existing
image-runtime exports. `FusedAttentionPolicy` lives in `MereRunTensor` so the
shared encoder preserves its diagnostic fallback without depending on Core.

The sampler uses `(1+w)*conditional-w*unconditional`, including the camera
stream. Octree children use x-fastest order and systematic resampling. Gaussian
self-attention sees every anchor; only independent octree queries are chunked.
PLY records contain SH DC coefficients, opacity logits, log scales, and
Hamilton quaternions rotated from z-up to y-up.

Reference: [speridlabs/idol-sdk](https://github.com/speridlabs/idol-sdk) at
`4702a7f00850e34b6e83a2170cc123c3e9570744` (Apache-2.0).
Tiny upstream CPU fixtures cover the denoiser, guided Euler trajectory,
decoders, Gaussian activation, PLY payload, and BF16 VAE token packing.
Real-checkpoint qualification is recorded in
`docs/qualification/idol-native-20261030/README.md`. The local gate also compares
BiRefNet's reduced graph with upstream and its FP32/FP16 deformable convolutions
with torchvision, including fractional offsets and out-of-bounds sampling.
