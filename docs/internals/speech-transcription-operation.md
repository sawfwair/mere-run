# Shared speech transcription operation

Use the shared speech operation to transcribe or translate an audio file from
a command, API adapter, or another executor. `AudioCore` owns its typed plan,
validation, events, and outcome. `AudioSTT` supplies native model resolution and
execution.

## Resolve and execute a request

Resolve an `ASRRequest` with `SpeechTranscriptionResolver.resolve`. The resolver
selects a backend from the task, language hint, requested backend, and model
override. It resolves installed model paths without loading a checkpoint.

Translation selects Qwen. When backend routing changes a built-in model
selection, the plan uses the selected backend's compatible default model.
An incompatible explicit local model path produces a validation error. Core ML
execution requires a request that resolves to Parakeet.

Pass the resolved plan to `SpeechTranscriptionOperation.execute` with a
`SpeechTranscriptionExecutor`. The operation checks the token limit, task/backend
compatibility, model identity, and input file. It rechecks mutable file availability
at execution because a file can disappear after resolution.

`NativeSpeechTranscriptionExecutor` constructs and unloads its temporary
generator. A resident executor borrows a prepared runtime from its pool. The
operation does not unload a borrowed runtime or acquire another machine
reservation.

## Consume outcomes

The operation returns `SpeechTranscriptionOutcome`, including the run identity,
resolved plan, backend decision, and `ASRResult`. An optional event handler
receives started and progress events followed by one success, failure, or
cancellation event.

Cancellation is checked before execution and after the executor returns.
A cancelled request does not become a successful outcome when an executor
returns without throwing. Input and compatibility errors use
`SpeechTranscriptionIssue` with a stable code and message.

The CLI preserves transcript formatting, timestamps, receipts, and diagnostics.
The API preserves its managed-model allowlist, OpenAI model aliases, and JSON,
text, SRT, and VTT response formats. Its runtime service owns the request slot,
and the transport removes the temporary upload after processing.

Raw stdin and live streaming continue to use the existing ASR session protocol.
The shared file operation's events do not add a CLI event mode or durable
transcription run records.

## Validate the boundary

Compile the operation tests without inference dependencies:

```bash
swift build --target AudioCoreTests
```

The repository gate executes operation tests with fixture executors and
CLI/API plan-parity tests. These tests cover backend dispatch, effective inputs,
validation, terminal events, cancellation, and resident request cleanup.
Real-model decoding remains a separate runtime validation step.
