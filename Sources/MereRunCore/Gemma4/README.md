# Gemma4

Gemma 4 chat model runtime and tokenizer/template support.

- `Gemma4Config.swift`: typed text model configuration.
- `Gemma4TokenizerAndTemplate.swift`: chat-template and tokenizer boundary.
- `Gemma4Model.swift`: native model layers.
- `Gemma4Generator.swift`: `ChatGenerator` integration.
- `Gemma4ToolParser.swift`: tool-call parsing.

Keep OpenAI-style tool/message adaptation typed before passing into tokenizer
library boundaries.
