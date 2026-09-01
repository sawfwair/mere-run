# Gemma4

Gemma 4 text and vision-language runtime, tokenizer, and LoRA training support.

- `Gemma4Config.swift`: typed text model configuration.
- `Gemma4TokenizerAndTemplate.swift`: chat-template and tokenizer boundary.
- `Gemma4CanonicalChatTemplate.swift`: checksum-gated canonical-template overlay
  for known stale Google/MLX model packages. The E4B generation primer remains
  separate from the shared 12B/26B/31B template.
- `Gemma4Model.swift`: native model layers.
- `Gemma4Generator.swift`: `ChatGenerator` integration.
- `Gemma4UnifiedImageProcessor.swift`: image preprocessing and visual-token
  metadata for unified inference and training.
- `Gemma4UnifiedModelLoader.swift`: unified-model loading shared by inference
  and VLM training.
- `Gemma4VLMSFTTokenizer.swift`: image-conditioned SFT prompt expansion and
  assistant-only targets.
- `Gemma4VLMLoRATrainingPipeline.swift`: batch-one VLM LoRA orchestration with
  a frozen vision stack and language-attention adapters.
- `Gemma4ToolParser.swift`: tool-call parsing.

Keep OpenAI-style tool/message adaptation typed before passing into tokenizer
library boundaries.

Canonical templates are applied only when the package template has the exact
known-stale SHA-256 and the decoded model profile matches a released Gemma 4
shape. Current or custom package templates remain authoritative.
