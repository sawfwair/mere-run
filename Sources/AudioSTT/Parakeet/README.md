# Parakeet STT

Parakeet speech-to-text backend implementation.

- `ParakeetConfig.swift`: typed model configuration.
- `ParakeetTokenizer.swift`: tokenizer loading and text decoding.
- `ParakeetModel.swift`: native model layers.
- `ParakeetGenerator.swift`: transcription orchestration.

Keep backend routing in `AudioCore`; this directory should only own Parakeet
loading, inference, and decoding behavior.
