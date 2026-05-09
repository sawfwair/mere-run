# Qwen3 ASR Model

Native encoder/decoder layers for the Qwen3 speech-to-text backend.

- `Qwen3ASRModel.swift`: model entry point and decoding path.
- `Qwen3ASRAudioEncoder.swift`: audio encoder components.

Keep tokenizer and config compatibility in the parent `Qwen3ASR` boundary files.
This directory should stay focused on typed model math.
