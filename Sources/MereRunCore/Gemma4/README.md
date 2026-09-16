# Gemma4

Gemma 4 text and vision-language runtime, tokenizer, and LoRA training support.

Model configuration, layers, caches, and assistant computation live in
[`MereRunGemmaModel`](../../MereRunGemmaModel/README.md). Core retains loading,
templates, generation, and LoRA orchestration.

- `Gemma4TokenizerAndTemplate.swift`: chat-template and tokenizer boundary.
- `Gemma4CanonicalChatTemplate.swift`: checksum-gated canonical-template overlay
  for known stale Google/MLX model packages. The E4B generation primer remains
  separate from the shared 12B/26B/31B template.
- `Gemma4Generator.swift`: actor state and `ChatGenerator` integration.
  Extensions separate loading, request generation, policies, prefill, prefix
  snapshots, decode, MTP verification, prompt lookup, and batching.
- `Gemma4AssistantDraftModel+Loading.swift`: assistant checkpoint loading.
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

Chat, preparation, and the background batching loop lease separate execution
contexts while active. Completed contexts remain reusable after model unload.
The pool belongs to the generator; recreating generators creates more backend
streams, which MLX retains until process exit. Keep the generator resident when
serving repeated requests. Finish submitted work before returning a lease, and
do not let background tasks retain a request's stream after that request ends.

Prefix snapshots keep independent array wrappers so later prompts and cancelled
requests cannot overwrite a retained prefix. Token-limited replies report
`length`; replies that stop before the budget report `stop`.

Canonical templates are applied only when the package template has the exact
known-stale SHA-256 and the decoded model profile matches a released Gemma 4
shape. Current or custom package templates remain authoritative.
