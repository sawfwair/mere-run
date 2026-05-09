# VLM

Vision-language model helpers.

- `QwenVLEncoder.swift`: multimodal encoder and prefill path.
- `QwenVLCaptioner.swift`: lightweight caption generator.
- `QwenVLImageLoader.swift`: image preparation.
- `Qwen3VLAutoCaptioner.swift`: automatic captioning support.

Keep debug output environment-gated and avoid letting image/tokenizer boundary
data leak into generation code as raw dictionaries.
