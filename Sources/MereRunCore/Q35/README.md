# Q35

Qwen 3.5/3.6 hybrid MoE text and vision-language runtime.

- `Q35Config.swift`: typed text/vision configuration.
- `Q35TokenizerAndTemplate.swift`: prompt rendering and tokenization.
- `Q35Model.swift`: native model entry point.
- Attention and MoE files own model math only.

Keep tokenizer/tool template compatibility isolated here; model layers should
not know about CLI or managed-model concerns.
