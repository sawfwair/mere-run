# Ideogram4

Native runtime support for the SDNQ uint4 Ideogram 4 text-to-image stack.

This module owns:

- managed-resource layout for `image-ideogram4-sdnq-uint4`
- Diffusers transformer and Qwen3-VL config decoding
- SDNQ linear, embedding, and Conv2d loading through `SDNQWeightsLoader`
- Qwen3-VL raw activation feature concatenation
- packed text/image sample construction with Ideogram role indicators and MRoPE
  position ids
- the Ideogram 4 flow transformer topology
- experimental Apple-Silicon fused QKV/AdaLN kernels, opt-in with
  `MERERUN_IDEOGRAM4_FUSED_KERNELS=1`; the exact portable graph remains the
  default until checkpoint-level end-to-end performance clears the release gate
- uniform-segment attention that avoids the redundant quadratic mask emitted
  by the supported single-sample packer
- positive/unconditional CFG denoising, scheduler presets, latent normalization,
  and Flux2-style VAE decode for `image generate`

Image-to-image, reference inputs, and LoRA are intentionally unsupported for
this family until their conditioning paths are implemented.
