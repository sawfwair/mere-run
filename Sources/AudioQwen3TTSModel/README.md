# Qwen TTS model runtime

Use this library when changing Qwen TTS model layers or speech-token tensors.
It depends on MLX and `MereRunKVCache`, without Core or audio codecs.

- `Qwen3TTSModel.swift`: generic text/codec model and configuration adapter.
- `Qwen3TTSConfig.swift` and `Qwen3TTSModelConfig.swift`: typed root schemas.
- `Qwen3TTSTalker.swift` and `Qwen3TTSTalkerConfig.swift`: talker layers,
  code prediction, rotary embeddings, and their configuration.
- `Qwen3TTSSpeechTokenizer.swift`: speech-code decoding and weight sanitization.
- `Qwen3TTSSpeechTokenizer+Encoder.swift`: reference-audio tensor encoder.
- `Qwen3TTSSpeechTokenizer+Decoder.swift`: transformer and waveform assembly.
- `Qwen3TTSSpeechTokenizer+Convolutions.swift` and `+Quantization.swift`:
  convolution blocks and residual vector quantization.
- `Qwen3TTSSpeakerEncoder.swift`: speaker embeddings from mel tensors.

The talker and speech-tokenizer composition surfaces use package access.
`AudioTTS` owns generation, sampling, checkpoint resolution, text tokenization,
PCM resampling, and the public audio-input convenience methods. Existing
`AudioTTS` imports retain the exported model types and audio methods.

Keep checkpoint keys, grouped-query head counts, cache offsets, and causal
waveform context unchanged. `SpeechRuntimeTests` checks those boundaries
without loading the full runtime or downloading weights.
