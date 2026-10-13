# Core ML embedding runtime

This internal target owns native Core ML EmbeddingGemma 2 graph execution,
bfloat16 token lookup, the exact Gemma tokenizer, and image/audio preprocessing.
Core owns managed installation, task formatting, ordered inputs, and CLI output.

The model artifact is FluidInference/embeddinggemma-2-coreml at
`3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`. The tokenizer, decoders, bicubic
resampler, and audio resampler adapt EverPaper's Apache-2.0 Swift implementation
at `305fc2a96790d9343f38038aeca211dfca56d9a5`. See LICENSE, NOTICE, and the root
THIRD_PARTY_NOTICES.md. No external inference process is required.

Multifunction Core ML requires macOS 15 or later; this runtime compiles and runs
with the macOS 26 SDK. Graph execution is excluded on Linux and iOS, while
portable metadata remains available. Text prefers CPU and Neural Engine; image
and audio encoders use CPU and GPU. Core ML compiles packages locally and may
specialize functions on first use. The class releases its temporary compiled
packages when inference finishes.

Read `RuntimeContract.swift` for typed geometry and BF16 lookup, `Embedder.swift`
for graph interfaces, and `ImagePreprocessor.swift` for normalized row-major
patches, explicit pooling, and position tables. The text limit is 512 tokens.
Audio includes its log-mel frontend in the graph and accepts explicit segments
longer than 10 milliseconds and at most 10 seconds.
