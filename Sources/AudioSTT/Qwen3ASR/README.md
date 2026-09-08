# Qwen3 ASR execution

This directory owns Qwen3 speech-to-text loading and execution. Model layers
and typed configuration live in `AudioQwen3ASRModel` and are re-exported by
`AudioSTT`.

- `Qwen3ASRTokenizer.swift`: tokenizer loading and compatibility
- `Qwen3ASRGenerator.swift`: actor state and public transcription lifecycle
- `Qwen3ASRGenerator+Loading.swift`: resolution, checkpoint loading, and weight mapping
- `Qwen3ASRGenerator+Generation.swift`: features, prompts, and pipelined decoding
- `Qwen3ASRStreamingSession.swift`: streaming cadence, backpressure, and terminal events
- `Qwen3ASRLiveSession.swift`: live session adapter

Keep tokenizer and checkpoint compatibility typed and covered by focused tests.
Preserve the pipelined decode toggle and token-readback schedule when editing
generation. Streaming producers own their work until finish or cancellation.
