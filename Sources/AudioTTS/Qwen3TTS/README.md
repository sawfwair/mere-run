# Qwen3 TTS Module

This directory contains the Qwen3 text-to-speech runtime.

- `Qwen3TTSTokenizer.swift`: tokenizer loading and prompt encoding
- `Qwen3TTSGenerator*.swift`: load, generate, and streaming support
- `Qwen3TTSModel*.swift`: model definitions and config
- `Qwen3TTSSpeechTokenizer*.swift`: speech-token encoder and decoder path

Keep the tokenizer and config ingestion layer explicit and typed. Runtime generation code should not need to reason about raw external JSON shapes.
