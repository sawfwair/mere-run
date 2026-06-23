# Krea2

Native runtime support for Krea 2 Turbo text-to-image generation.

This module owns:

- `Krea2Resources.swift`: managed `image-krea2-turbo` layout and component-only
  Hugging Face pull patterns.
- `Krea2Configs.swift`: typed Diffusers, Qwen3-VL text-encoder, transformer,
  scheduler, and VAE config decoding.
- `Krea2SampleBuilder.swift`: 16-pixel output alignment, latent patch packing,
  position ids, attention masks, and shifted 8-step FlowMatch timesteps.
- `Krea2Model.swift`: native Swift MLX single-stream MMDiT, text fusion, MRoPE,
  timestep modulation, and final latent projection.
- `Krea2ModelLoader.swift`: split-transformer, Qwen3-VL text encoder, and Qwen
  Image VAE weight loading.
- `Krea2Generator.swift`: prompt conditioning, denoising, VAE decode, and PNG
  output for `mere.run image generate`.

Krea publishes both split Diffusers component weights and a root-level
`turbo.safetensors` transformer copy. Managed pulls must keep using the
component layout so local installs do not fetch the same large transformer
twice.

The wired public mode is text-to-image with managed defaults of 8 steps, CFG
0.0, and FlowMatch shift/mu 1.15. Reference images, image-to-image, and LoRA are
intentionally unsupported until their conditioning paths are implemented.
