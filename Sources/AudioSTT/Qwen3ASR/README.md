# Qwen3 ASR Module

This directory contains the Qwen3 speech-to-text path.

- `Qwen3ASRConfigs.swift`: config ingestion and typed model settings
- `Qwen3ASRTokenizer.swift`: tokenizer loading and compatibility
- `Qwen3ASRGenerator.swift`: user-facing transcription path
- `Model/`: encoder and decoder model definitions

When editing here, keep the config and tokenizer boundary typed. If a model-format quirk forces compatibility logic, isolate it in the boundary file and cover it with a focused test.
