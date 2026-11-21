# Flux2Klein

FLUX.2 Dev and Klein image generation runtime, with Klein LoRA training.

- `Flux2KleinGenerator*.swift`: loading, prompt/reference encoding, denoising,
  decode, and generation support.
- `Flux2KleinLoRATrainer*.swift`: training, checkpointing, and optimization.
- `Flux2KleinGenerator+Encoding.swift`: prompt and reference-image conditioning.
- `Flux2KleinGenerator+Denoising.swift`: guidance and scheduler updates.
- `Flux2KleinGenerator+Output.swift`: latent decoding and image writing.
- `MereRunImageModels/Flux2/`: typed configuration and native transformer blocks.

Image generation, chat preparation, and chat inference lease task-local MLX CPU
and GPU streams for each active request. Completed requests synchronize their
streams before returning them to the shared `MLXRequestStreams` pool. Contexts
remain reusable across generator eviction and replacement.

Keep CLI behavior in `MereRunCLI`; this module should expose typed generation
and training primitives.
