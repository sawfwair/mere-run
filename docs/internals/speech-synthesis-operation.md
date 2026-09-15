# Shared speech synthesis

The CLI and speech API use `SpeechSynthesisPlan` and
`SpeechSynthesisOperation` in AudioCore. The plan retains a validated copy of
the request and an explicit WAV export policy. The operation consumes waveform
samples from a `SpeechSynthesisExecutor` and publishes a completed artifact.
AudioCore has no MLX dependency.

## Request ownership

The CLI owns flags, profile lookup, reference transcription, language hints,
progress, and receipts. The API owns input trimming, model aliases, voice-name
mapping, its 32 KiB prompt limit, and response transcoding. Both use
`SpeechSynthesisModelSelection` in AudioTTS for managed IDs and local model
directories. Selection does not load or download a model.

Shared validation rejects empty text, nonfinite or out-of-range temperature and
speed, invalid streaming intervals, and incomplete clone references. Execution
rechecks reference-file existence and destination type before calling the
executor. CLI scalar validation runs before profile or reference preparation.
The API continues accepting speed values that the Qwen runtime does not apply.

## Native execution and lifetime

`Qwen3TTSSynthesisExecutor` adapts a caller-owned `Qwen3TTSGenerator`. The actor
shares model loading, clone-asset checks, prompt preparation, and generation
between offline and streaming synthesis. It retains MLX CPU/GPU streams,
evaluates output before returning host samples, and synchronizes on completion,
failure, and unloading. Token loops check for cooperative task cancellation.

The CLI unloads its generator after completion or failure. The API keeps its
generator in `APISidecarModelPool`; existing admission and residency leases
surround the shared operation. AudioCore does not acquire a second permit.

Public `Qwen3TTSGenerator.generate` still returns a file result through the
shared operation. Its public `generateStream` still emits raw sample events
without writing a file. `SNACAudioWriter` retains its PCM16 compatibility entry
point through the shared export service.

## Artifact publication

Offline output uses PCM16 with no normalization, fades, or dither. Streaming
output uses float32 with finite headroom preserved. Both replace nonfinite
samples with zero. `AudioExportStream` shares processing and WAV encoding with
offline export, writes incremental payloads to a temporary sibling, then fixes
the header and atomically publishes the destination.

The streaming operation requires a stable sample rate, audio samples, and one
matching final result. It waits for the producer to finish before publishing
and emitting completion. Failure or cancellation discards temporary output.
Consumer cancellation propagates to the operation and native producer tasks;
cleanup completes when the producer reaches a cancellation boundary.

## Validation boundaries

Portable fixtures cover request equivalence, byte-level encoding, malformed
streams, final receipt metadata, producer cancellation, and existing-file
preservation. Native baseline comparisons cover deterministic style and clone
requests across offline, streaming, and repeated API synthesis. These are
separate from qualification of GPU cancellation recovery, long-running
synthesis, or perceptual voice quality.
