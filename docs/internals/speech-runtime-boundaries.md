# Speech runtime boundaries

Use this map when you change speech model layers, checkpoint loading, or
recognition and synthesis orchestration. The model libraries let you compile
and test speech math without importing the full Core runtime.

## Ownership

| Owner | Responsibility | Dependencies |
| --- | --- | --- |
| `AudioCore` | Transcription and synthesis requests, routing policy, operation events, and outcomes | Foundation and `MereRunExecution` |
| `AudioQwen3ASRModel` | Qwen ASR configuration, audio encoder, and decoder | MLX and `MereRunKVCache` |
| `AudioParakeetModel` | Parakeet configuration, encoder, decoders, and alignment | MLX |
| `AudioWhistleModel` | Whistle encoder, cached decoder, checkpoint, and vocabulary loading | MLX |
| `AudioNemotronTranscribeModel` | Nemotron 3.5 Transcribe perception, speaker fusion, and hybrid language decoding | MLX and `AudioSortformer` |
| `AudioSortformer` | Diarization features, model layers, checkpoint loading, and segments | MLX and `MereRunModelKit` |
| `AudioQwen3TTSModel` | Qwen TTS configuration, talker, speech-token tensors, and speaker layers | MLX and `MereRunKVCache` |
| `AudioBreezeTTSModel` | Breeze speech synthesis computation | MLX, `AudioQwen3TTSModel`, the decode, cache, and tensor libraries, and Tokenizers |
| `AudioDrama3Model` | DRAMA-3 prompt, codec, audio tower, FP8 ingestion, and packed inference | MLX, `MereRunQwenModel`, `MereRunKVCache`, and `MereRunTensor` |
| `AudioPhononModel` | Phonon flow sampling and streaming Mimi synthesis | MLX |
| `MereRunKVCache` | Full-attention cache protocol, dynamic and static storage, and ragged batches | MLX |
| `AudioSTT` | Native transcription executors, model resolution, Core ML bridges, and sessions | Core, audio utilities, and the recognition libraries |
| `AudioTTS` | Speech synthesis, prompt preparation, sampling, and audio-input adapters | Core, audio utilities, and the synthesis libraries |

`MereRunCore` re-exports the cache types. `AudioSTT` re-exports the Qwen ASR,
Parakeet, and Sortformer types, and `AudioTTS` re-exports the Qwen TTS types,
so existing imports and audio-input convenience methods keep working. Libraries
added later, such as Whistle, Nemotron Transcribe, Breeze, DRAMA-3, and Phonon,
aren't re-exported; import them directly. Neural composition details that only
the orchestrator needs use package access.

## Reading order

### Qwen ASR

Qwen ASR starts in `AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift`:

1. `Qwen3ASRGenerator+Loading.swift` resolves the model and loads weights and
   tokenizer data.
2. `Qwen3ASRGenerator+Generation.swift` extracts features and runs prompt and
   token generation through `AudioQwen3ASRModel`.
3. `Qwen3ASRStreamingSession.swift` owns streaming cadence, backpressure, and
   terminal events.

### Parakeet

Parakeet follows the same split for lifecycle, loading, and measured
execution. `AudioParakeetModel` separates Conformer layers, recurrent
prediction, TDT, RNN-T, and CTC decoding, and alignment. `AudioSTT` supplies
Core ML implementations through the model library's tensor interfaces.

### Other recognizers

`AudioSTT/Whistle/WhistleGenerator.swift` and
`AudioSTT/NemotronTranscribe/NemotronTranscribeGenerator.swift` adapt their
model libraries to the shared transcription executor. For backend routing, see
the [shared speech transcription operation](./speech-transcription-operation.md).

### Qwen TTS

Qwen TTS starts in `AudioTTS/Qwen3TTS/Qwen3TTSGenerator.swift`:

1. `+Loading.swift` resolves checkpoints and fills the model library's layers.
2. `+PromptPreparation.swift` combines text and reference-code embeddings.
3. `+Generation.swift` selects voice-design or cloning behavior.
4. `+TokenGeneration.swift` runs the pipelined talker and codec loops.
5. `+StreamingAudio.swift` emits the newly decoded waveform tail.

The speech-tokenizer convolution and quantization files belong to
`AudioQwen3TTSModel`. PCM resampling and speaker audio-input adapters belong to
`AudioTTS`. The Breeze, DRAMA-3, and Phonon generators live in
`AudioTTS/BreezeTTS`, `AudioTTS/Drama3`, and `AudioTTS/Phonon`.

### Sortformer

`AudioSortformer` owns its complete array-based runtime. `SortformerModel.swift`
contains model layers, `+Loading.swift` contains weight sanitization and
loading, and `+Inference.swift` runs features-to-segments execution. The same
target holds the Nemotron 3 diarization model. File decoding belongs to the
caller.

## Contracts to preserve

- Parakeet keeps its task-safe stream scopes around preparation and decoding,
  including their suspension points.
- Qwen TTS keeps causal context and the one-step delayed confirmation in its
  pipelined loop.

## Validate a change

Compile the isolated numerical test target:

```bash
swift build --target SpeechRuntimeTests
```

This target imports the Qwen, Parakeet, and Sortformer model libraries and
shared MLX test support. It doesn't depend on Core, the CLI, Transformers,
audio codecs, or model downloads. Nemotron Transcribe, DRAMA-3, and Phonon have
their own isolated targets: `NemotronTranscribeRuntimeTests`,
`Drama3RuntimeTests`, and `AudioPhononModelTests`.

The repository gate checks transitive local and external dependencies. For the
Qwen ASR, Qwen TTS, Parakeet, Sortformer, and Whistle libraries,
`scripts/check-model-boundaries.swift` rejects any dependency on Core, codecs,
or Transformers. The gate also rejects any shipped product dependency on MLX
test support, and runs these tests with the CLI and full runtime integration
suite:

```bash
./scripts/check.sh
```

`SpeechRuntimeTests` covers cache fork isolation, ragged batch masks and
splitting, Qwen cached decoding and last-position projection, TTS waveform
chunking and weight sanitization, Parakeet recurrent-state continuity, batched
windows, token timing, and a generated Sortformer checkpoint loaded through a
symlink. The generated fixture checks numerical round-trip behavior; it doesn't
measure recognition quality or throughput.

`MereRunMLXTestSupport` shares test resource setup across the speech, Core, and
TTS suites. It's a test dependency and isn't linked into shipped products.
