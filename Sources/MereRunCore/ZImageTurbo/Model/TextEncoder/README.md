# ZImageTurbo TextEncoder

Qwen-style text encoder used by ZImage Turbo.

- `TextEncoder.swift`: encoder entry point.
- `TextEncoder+Blocks.swift`: attention and MLP blocks.
- `TextEncoder+RoPE.swift`: rotary embedding.
- `LLMGeneration/`: generation helpers.
- `Vision/`: vision tower pieces.

Keep debugging hooks gated and avoid adding tokenizer/file parsing concerns to
model blocks.
