# Shared speech synthesis operation

The CLI and the speech API use `SpeechSynthesisPlan` and
`SpeechSynthesisOperation` in `AudioCore`. The plan holds a validated copy of
the request and an explicit WAV export policy. The operation consumes waveform
samples from a `SpeechSynthesisExecutor` and publishes a completed artifact.
`AudioCore` has no MLX dependency.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `SpeechSynthesisPlan` | Validated request copy and WAV export policy |
| `SpeechSynthesisOperation` | Execution, streaming, and atomic artifact publication |
| `SpeechSynthesisModelSelection` (`AudioTTS`) | Resolve managed IDs and local model directories to a backend, without loading or downloading, and build that backend's generator and executor |
| Backend executors (`AudioTTS`) | `Qwen3TTSSynthesisExecutor`, `BreezeTTSSynthesisExecutor`, `Drama3SynthesisExecutor`, and `PhononSynthesisExecutor` adapt a caller-owned generator |
| CLI adapter | Flags, profile lookup, reference transcription, language hints, progress, and receipts |
| API adapter | Input trimming, model aliases, voice-name mapping, the 32 KiB prompt limit, and response transcoding |

## Plan and validate a request

Shared validation rejects empty text, non-finite or out-of-range temperature
and speed, invalid streaming intervals, and incomplete clone references. The
CLI runs scalar validation before profile or reference preparation. The API
still accepts speed values that the Qwen runtime doesn't apply.

Execution rechecks that reference files exist and that the destination type is
valid before it calls the executor.

## Execution and runtime lifetime

`SpeechSynthesisModelSelection` selects Qwen3-TTS, Breeze TTS 2, DRAMA-3, or
Phonon. The following details describe the Qwen3-TTS executor.

The `Qwen3TTSGenerator` actor shares model loading, clone-asset checks, prompt
preparation, and generation between offline and streaming synthesis. It keeps
MLX CPU and GPU streams, evaluates output before returning host samples, and
synchronizes on completion, failure, and unload. Token loops check for
cooperative task cancellation.

For every backend, the CLI unloads its generator after completion or failure.
The API keeps its generator in `APISidecarModelPool`, and its admission and residency leases
surround the shared operation. `AudioCore` doesn't acquire a second permit.

Public `Qwen3TTSGenerator.generate` still returns a file result through the
shared operation. Its public `generateStream` still emits raw sample events
without writing a file. `SNACAudioWriter` keeps its PCM16 entry point through
the shared export service.

## Artifact publication

Offline output is PCM16 with no normalization, fades, or dither. Streaming
output is float32 with finite headroom preserved. Both replace non-finite
samples with zero. `AudioExportStream` shares processing and WAV encoding with
offline export. It writes incremental payloads to a temporary sibling, fixes the
header, and then atomically publishes the destination.

The streaming operation requires a stable sample rate, audio samples, and one
matching final result. It waits for the producer to finish before it publishes
and emits completion. Failure or cancellation discards temporary output.
Consumer cancellation propagates to the operation and the native producer
tasks; cleanup completes when the producer reaches a cancellation boundary.

## Validate a change

Portable fixtures cover request equivalence, byte-level encoding, malformed
streams, final receipt metadata, producer cancellation, and preservation of
existing files. Native baseline comparisons cover deterministic style and clone
requests across offline, streaming, and repeated API synthesis.

Neither qualifies GPU cancellation recovery, long-running synthesis, or
perceptual voice quality.
