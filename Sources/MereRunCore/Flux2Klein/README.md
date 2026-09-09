# Flux2Klein

FLUX.2 Dev and Klein image generation runtime, with Klein LoRA training.

- `Flux2KleinGenerator*.swift`: loading, prompt/reference encoding, denoising,
  decode, and generation support.
- `Flux2KleinLoRATrainer*.swift`: training, checkpointing, and optimization.
- `Flux2KleinGenerator+Encoding.swift`: prompt and reference-image conditioning.
- `Flux2KleinGenerator+Denoising.swift`: guidance and scheduler updates.
- `Flux2KleinGenerator+Output.swift`: latent decoding and image writing.
- `MereRunImageModels/Flux2/`: typed configuration and native transformer blocks.

Keep CLI behavior in `MereRunCLI`; this module should expose typed generation
and training primitives.
