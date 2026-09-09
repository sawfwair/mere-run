# ZImageTurbo

ZImage Turbo image generation and LoRA training runtime.

- `ZImageTurboGenerator*.swift`: loading, inference, LoRA, and generation.
- `ZImageTurboLoRATrainer*.swift`: training, dataset encoding, optimization,
  and checkpointing.
- `ZImageTurboModelConfigs.swift`: typed model configuration.
- `Tokenizer/`: tokenizer compatibility boundary.
- `ZImageTurboGenerator+Denoising.swift`: guidance and scheduler updates.
- `ZImageTurboGenerator+Output.swift`: latent decoding and image writing.
- `MereRunImageModels`: transformer and VAE model components.
- `MereRunTextEncoder`: shared Qwen text and vision model layers.

Keep training progress and runtime debug output intentional; default generation
paths should remain quiet.
