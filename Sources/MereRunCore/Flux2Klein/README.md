# Flux2Klein

FLUX.2 Klein image generation and LoRA training runtime.

- `Flux2KleinGenerator*.swift`: loading, prompt/reference encoding, denoising,
  decode, and generation support.
- `Flux2KleinLoRATrainer*.swift`: training, checkpointing, and optimization.
- `Flux2KleinConfigs.swift`: typed runtime configuration.
- `Model/Transformer/`: native transformer blocks.

Keep CLI behavior in `MereRunCLI`; this module should expose typed generation
and training primitives.
