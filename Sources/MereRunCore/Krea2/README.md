# Krea2

Native runtime support for Krea 2 text-to-image generation and LoRA training.

This module owns:

- `Krea2Resources.swift`: managed `image-krea2-turbo` layout and
  component-only Hugging Face pull patterns.
- `Krea2RawResources.swift`: managed `image-krea2-raw` layout for LoRA
  training.
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
- `Krea2LoRAInjector.swift`: Krea transformer LoRA insertion and weight
  application.
- `Krea2LoRATrainer.swift`: Raw-model LoRA training, metrics, manifests, and
  adapter bundle output.

Krea publishes split Diffusers component weights and root-level `raw.safetensors`
or `turbo.safetensors` transformer copies. Managed pulls must keep using the
component layout so local installs do not fetch the same large transformer
twice.

The wired public generation mode is text-to-image with optional LoRA adapters.
Train LoRAs on `image-krea2-raw`, then run them on `image-krea2-turbo`.
Reference images and image-to-image are intentionally unsupported until their
conditioning paths are implemented.
