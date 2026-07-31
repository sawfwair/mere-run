# Inkling

Inkling-Small text generation runs in the native Swift/MLX runtime.

- `InklingResources.swift` pins the official BF16 checkpoint and mere.run's
  mixed MLX conversion: routed experts are affine 2-bit/group-128 and all
  non-routed weights remain BF16.
- `InklingModel.swift` implements Inkling's relative-position attention,
  short-convolution residuals, and sigmoid-routed shared-expert MoE.
- `InklingGenerator.swift` owns managed loading, prompt encoding, chunked
  prefill, and autoregressive decode.

The released checkpoint accepts text, images, and audio. This first mere.run
lane intentionally loads only `language_model.*` weights and exposes text chat;
the vision and audio towers remain out of scope.

The architecture advertises 1,048,576 tokens. mere.run defaults to 32,768
because KV-cache and attention-mask residency still grow with context on the
128 GB machines targeted by this conversion.
