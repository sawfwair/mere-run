# Speech runtime boundaries

Use this map when changing speech model layers, checkpoint loading, or
transcription orchestration. The model libraries let you compile and test
speech math without importing the full Core runtime.

## Choose the owning library

| Library | Responsibility | Dependencies |
| --- | --- | --- |
| `AudioCore` | Transcription requests, routing policy, operation events, and outcomes | Foundation |
| `AudioQwen3ASRModel` | Qwen ASR configuration, audio encoder, and decoder | MLX and `MereRunKVCache` |
| `AudioQwen3TTSModel` | Qwen TTS configuration, talker, speech-token tensors, and speaker layers | MLX and `MereRunKVCache` |
| `AudioParakeetModel` | Parakeet configuration, encoder, decoders, and alignment | MLX |
| `AudioSortformer` | Diarization features, model layers, checkpoint loading, and segments | MLX and `MereRunModelKit` |
| `MereRunKVCache` | Full-attention cache protocol, dynamic/static storage, and ragged batches | MLX |
| `AudioSTT` | Native transcription executors, model resolution, Core ML bridges, and sessions | Core, audio utilities, and the speech libraries |
| `AudioTTS` | Speech synthesis, prompt preparation, sampling, and audio-input adapters | Core, audio utilities, and `AudioQwen3TTSModel` |

`MereRunCore` re-exports the cache types. `AudioSTT` re-exports Qwen ASR,
Parakeet, and Sortformer types; `AudioTTS` re-exports Qwen TTS types. Existing
source imports and audio-input convenience methods continue to work. Neural
composition details use package access where only the orchestrator needs them.

## Follow the execution path

Qwen ASR starts in `AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift`:

1. `Qwen3ASRGenerator+Loading.swift` resolves the model and loads weights and
   tokenizer data.
2. `Qwen3ASRGenerator+Generation.swift` extracts features and runs prompt and
   token generation through `AudioQwen3ASRModel`.
3. `Qwen3ASRStreamingSession.swift` owns streaming cadence, backpressure, and
   terminal events.

Parakeet follows the same separation for lifecycle, loading, and measured
execution. `AudioParakeetModel` separates Conformer layers, recurrent prediction,
TDT/RNN-T/CTC decoding, and alignment. `AudioSTT` supplies Core ML implementations
through the model library's tensor interfaces. Keep the task-safe stream scopes
around preparation and decoding, including their suspension points.

Qwen TTS starts in `AudioTTS/Qwen3TTS/Qwen3TTSGenerator.swift`:

1. `+Loading.swift` resolves checkpoints and fills the model library's layers.
2. `+PromptPreparation.swift` combines text and reference-code embeddings.
3. `+Generation.swift` selects voice-design or cloning behavior.
4. `+TokenGeneration.swift` runs the pipelined talker and codec loops.
5. `+StreamingAudio.swift` emits the newly decoded waveform tail.

The speech-tokenizer convolution and quantization files belong to
`AudioQwen3TTSModel`. PCM resampling and speaker audio-input adapters belong to
`AudioTTS`. Preserve causal context and the one-step delayed confirmation in
the pipelined loop.

Sortformer owns its complete array-based runtime in `AudioSortformer`.
`SortformerModel.swift` contains model layers; `+Loading.swift` contains weight
sanitization and loading; `+Inference.swift` contains features-to-segments
execution. File decoding belongs to the caller.

## Validate the dependency boundary

Compile the isolated numerical test target:

```bash
swift build --target SpeechRuntimeTests
```

This target imports the model libraries and shared MLX test support. It does
not depend on Core, the CLI, Transformers, audio codecs, or model downloads.
The repository gate also checks transitive local and external dependencies.
It rejects a model dependency on Core, codecs, or Transformers, and any shipped
product dependency on MLX test support. The gate runs these tests with the CLI
and full runtime integration suite:

```bash
./scripts/check.sh
```

The tests cover cache fork isolation, ragged batch masks and splitting, Qwen
cached decoding and last-position projection, TTS waveform chunking and weight
sanitization, Parakeet recurrent-state continuity, batched windows, token timing,
and a generated Sortformer checkpoint loaded through a symlink. The generated fixture checks numerical
round-trip behavior; it does not measure recognition quality or throughput.

`MereRunMLXTestSupport` shares test resource setup across the speech, Core, and
TTS suites. It is a test dependency and is not linked into shipped products.
