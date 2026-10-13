# Core AI embedding runtime

This internal target owns the native Core AI EmbeddingGemma 2 graph adapter,
Gemma tokenizer, and image/audio preprocessing. The Swift implementation is
adapted from EverPaper/embeddinggemma-2-coreai at
`305fc2a96790d9343f38038aeca211dfca56d9a5`, under Apache-2.0. See the root
THIRD_PARTY_NOTICES.md for attribution. Core owns managed installation and CLI
request orchestration. No Python process or upstream package is required.

Core AI inference requires the macOS 27 SDK, macOS 27, and Apple Silicon.
`canImport(CoreAI)` excludes graph execution with older SDKs; the public
availability check refuses inference before model resolution on those builds.
The tokenizer and preprocessing compile and can be tested on macOS 26.

Read `Embedder.swift` for graph inputs and sequence assembly, then
`GemmaTokenizer.swift`, `ImagePreprocessor.swift`, and `AudioFrontend.swift`.
Keep preprocessing and the pinned graphs synchronized. The shipped text graph
has a 2048-token limit, unlike the original 8192-token MLX model.
