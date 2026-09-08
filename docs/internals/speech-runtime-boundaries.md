# Speech runtime boundaries

Use this map when changing speech model layers, checkpoint loading, or
transcription orchestration. The model libraries let you compile and test
speech math without importing the full Core runtime.

## Choose the owning library

| Library | Responsibility | Dependencies |
| --- | --- | --- |
| `AudioCore` | Transcription requests, routing policy, operation events, and outcomes | Foundation |
| `AudioQwen3ASRModel` | Qwen ASR configuration, audio encoder, and decoder | MLX and `MereRunKVCache` |
| `AudioSortformer` | Diarization features, model layers, checkpoint loading, and segments | MLX and `MereRunModelKit` |
| `MereRunKVCache` | Full-attention cache protocol, dynamic/static storage, and ragged batches | MLX |
| `AudioSTT` | Native transcription executors, model resolution, tokenizers, and session orchestration | Core, audio utilities, and the speech libraries |

`MereRunCore` re-exports the cache types. `AudioSTT` re-exports Qwen model and
Sortformer types. Existing source imports continue to work. Direct consumers
can depend on the smaller libraries when they supply their own model loading
or audio inputs.

## Follow the execution path

Qwen ASR starts in `AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift`:

1. `Qwen3ASRGenerator+Loading.swift` resolves the model and loads weights and
   tokenizer data.
2. `Qwen3ASRGenerator+Generation.swift` extracts features and runs prompt and
   token generation through `AudioQwen3ASRModel`.
3. `Qwen3ASRStreamingSession.swift` owns streaming cadence, backpressure, and
   terminal events.

Parakeet follows the same file separation for lifecycle, loading, and measured
decoding. Its model layers and Core ML bridges remain in `AudioSTT`. Keep the
task-safe stream scopes around preparation and decoding, including their
suspension points.

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
The repository gate runs these tests with the CLI and full runtime integration
suite:

```bash
./scripts/check.sh
```

The tests cover cache fork isolation, ragged batch masks and splitting, Qwen
cached decoding and last-position projection, and a generated Sortformer
checkpoint loaded through a symlink. The generated fixture checks numerical
round-trip behavior; it does not measure recognition quality or throughput.

`MereRunMLXTestSupport` shares test resource setup across the speech, Core, and
TTS suites. It is a test dependency and is not linked into shipped products.
