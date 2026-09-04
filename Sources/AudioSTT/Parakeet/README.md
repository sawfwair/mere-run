# Parakeet STT

Parakeet speech-to-text backend implementation.

- `ParakeetConfig.swift`: typed model configuration.
- `ParakeetTokenizer.swift`: tokenizer loading and text decoding.
- `ParakeetModel.swift`: native model layers.
- `ParakeetGenerator.swift`: transcription orchestration.
- `ParakeetExecutionProvider.swift`: explicit MLX or Core ML provider selection
  and standalone package discovery.
- `ParakeetCoreMLEncoder.swift`: verified Core ML/MLX artifact contract and
  encoder tensor bridge.
- `ParakeetCoreMLDecoder.swift`: batched TDT decoder, recurrent state, and
  embedding-table bridge.
- `ParakeetBenchmark.swift`: resident pipeline stage timings.
- `ParakeetCoreMLWindowing.swift`: overlapped long-file windows and global
  alignment offsets for the static-shape Core ML encoder.

Keep backend routing in `AudioCore`; this directory should only own Parakeet
loading, inference, and decoding behavior.
