# MereRunImageModels

FLUX.2 and ZImage transformer layers, typed configuration, and the shared VAE.

- `Flux2/Transformer/` owns FLUX.2 model computation and position embeddings.
- `Flux2/Flux2KleinConfigs.swift` owns typed checkpoint configuration.
- `ZImage/Transformer/` owns ZImage blocks, caches, and coordinate handling.
- `ZImage/` owns typed configuration and the sparse-attention admission policy.
- `VAE/` owns shared image encoding and decoding tensors.

This library depends on MLX and `MereRunTensor`. Core owns model loading,
tokenization, generation stages, LoRA orchestration, and image-file output.
Keep filesystem and CLI concerns outside the model layers.
