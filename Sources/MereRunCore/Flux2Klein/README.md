# Flux2Klein

FLUX.2 Dev and Klein image generation runtime, with Klein LoRA training.

- `Flux2KleinGenerator*.swift`: loading, prompt/reference encoding, denoising,
  decode, and generation support.
- `Flux2KleinLoRATrainer*.swift`: training, checkpointing, and optimization.
- `Flux2KleinConfigs.swift`: typed runtime configuration.
- `Model/Transformer/`: native transformer blocks.

Keep CLI behavior in `MereRunCLI`; this module should expose typed generation
and training primitives.
