# Qwen3 ASR model layers

This library owns typed configuration and native encoder/decoder layers for
Qwen3 ASR. It depends on MLX and `MereRunKVCache`; it does not import
`MereRunCore`, checkpoint downloaders, the CLI, or audio file decoding.

- `Qwen3ASRModel.swift`: model entry point and decoding path.
- `Qwen3ASRAudioEncoder.swift`: audio encoder components.
- `Qwen3ASRConfigs.swift`: typed model configuration.

`AudioSTT` re-exports these public types and owns model resolution, weight and
tokenizer loading, feature extraction, and transcription orchestration.
Preserve parameter names, grouped-query head counts, cache offsets, and
last-position projection semantics when editing the model layers.
