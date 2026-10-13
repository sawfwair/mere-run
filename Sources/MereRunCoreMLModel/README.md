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
specialize functions on first use. Compiled packages persist under
`~/Library/Caches/MereRun/CoreML/EmbeddingGemma2/`. Cache keys cover the graph,
manifest, resolved weight identity and metadata, operating system, and cache
format. Compilation is coalesced in process and published atomically across
processes. An artifact replacement or operating-system update selects a new key.
The actor owns loaded models and coalesces overlapping function loads.

`embed(sequences:)` packs eligible texts into the upstream `pack_256` graph,
using independent attention, reset positions, and per-record pooling. At most
eight short ANE predictions overlap per batch. Long GPU records run individually.
Core exposes `EmbeddingGemma2CoreMLSession` for caller-owned reuse across
requests, without a global resident model or an extra admission reservation.
The static convenience API creates one session per call. Media prepares and
executes in order; text preparation is bounded to 64 records at a time.
Packing adapts FluidUse at `d278e17df85e39e8c21bb64a467da81c39576fcd` under
Apache-2.0; see NOTICE.

Read `RuntimeContract.swift` for typed geometry and BF16 lookup, `Embedder.swift`
for graph interfaces, and `ImagePreprocessor.swift` for normalized row-major
patches, explicit pooling, and position tables. The downloaded bundle accepts
512 tokens and audio segments up to 10 seconds. The optional expanded bundle
accepts 8,192 tokens and audio segments up to 30 seconds. Its longer text
functions use fused GPU attention with the original bidirectional local mask
and float32 computation. Its audio function uses native Accelerate log-mel
features and a float32 Core ML encoder. Short inputs retain the downloaded
functions. Limits come from the validated typed bundle configuration.

`scripts/model-conversion/convert_embeddinggemma2_coreml.py` builds the expanded
bundle from the original pinned Google checkpoint and the downloaded Core ML
bundle. It verifies source hashes, preserves both inputs, and records graph
and file hashes. Python is conversion tooling only. The conversion architecture
adapts EverPaper's Apache-2.0 text and audio export sources; the text attention
uses Core ML's fused SDPA, and the host supplies token embeddings.
