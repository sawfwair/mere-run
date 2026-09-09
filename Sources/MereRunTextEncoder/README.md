# MereRunTextEncoder

Shared Qwen text encoder and vision tower model layers.

- `TextEncoder.swift` owns configuration, hidden-state selection, and prompt padding.
- `TextEncoder+Blocks.swift` owns attention, MLP, and encoder layers.
- `TextEncoder+RoPE.swift` owns rotary embeddings.
- `Vision/` owns vision-tower computation and tensor preparation.

This library depends on MLX and the shared attention-cache library. Core owns
tokenizers, generation loops, checkpoint loading, and model resolution.
Preserve cached/full-pass parity and the selected intermediate hidden states.
