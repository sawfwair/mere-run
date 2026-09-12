# Qwen TTS orchestration

Use this directory when changing model loading, prompt preparation, or speech
generation. `AudioQwen3TTSModel` owns neural layers and typed configuration.

- `Qwen3TTSGenerator.swift`: public entry points and loaded actor state.
- `SpeechSynthesisModelSelection.swift`: shared CLI/API model selection and the
  Qwen waveform executor for `AudioCore.SpeechSynthesisOperation`.
- `Qwen3TTSGenerator+Loading.swift`: model resolution and checkpoint loading.
- `Qwen3TTSGenerator+Generation.swift`: voice-design and voice-clone flows.
- `Qwen3TTSGenerator+PromptPreparation.swift`: text and reference-code prompts.
- `Qwen3TTSGenerator+TokenGeneration.swift`: pipelined talker and codec loops.
- `Qwen3TTSGenerator+StreamingAudio.swift`: waveform tails for streaming output.
- `Qwen3TTSTokenizer.swift`: text tokenization.
- `Qwen3TTSSpeechTokenizer+Audio.swift` and `Qwen3TTSSpeakerEncoder+Audio.swift`:
  audio-input adapters over the model library.

Keep typed ingestion at the tokenizer and config boundaries. Preserve the
pipelined loop's evaluation order, delayed token confirmation, and sampling
policy when reorganizing execution code. The SNAC helpers remain separate
from the Qwen speech tokenizer.

Offline and streaming synthesis share loading, clone validation, and native
generation. The actor retains its MLX streams and synchronizes before releasing
model state. Keep WAV publication in AudioCore: offline output uses PCM16, and
streaming output uses float32. The public generator's raw sample stream does
not publish a file; its file-generation entry point uses the shared operation.
