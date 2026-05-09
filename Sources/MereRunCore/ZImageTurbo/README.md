# ZImageTurbo

ZImage Turbo image generation and LoRA training runtime.

- `ZImageTurboGenerator*.swift`: loading, inference, LoRA, and generation.
- `ZImageTurboLoRATrainer*.swift`: training, dataset encoding, optimization,
  and checkpointing.
- `ZImageTurboModelConfigs.swift`: typed model configuration.
- `Tokenizer/`: tokenizer compatibility boundary.
- `Model/`: native VAE, transformer, and text encoder components.

Keep training progress and runtime debug output intentional; default generation
paths should remain quiet.
