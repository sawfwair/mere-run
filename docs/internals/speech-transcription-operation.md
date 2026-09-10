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
The shared file operation's events do not add a CLI event mode.

## Retain and retry a file operation

Pass an optional `SpeechTranscriptionRunSession` to the operation to retain its
input and outcome. The session creates a private directory and takes a process
lease before writing a versioned `transcription-run.json` record. Existing
directories are rejected without rewriting their files.

`SpeechTranscriptionRunOptions` preserves supplied settings. The resolved plan
records the backend decision, model ID and local path, provider, language,
task, and token limit. Before inference, the session copies the input audio and
writes the effective plan. The executor reads that retained copy, so temporary
HTTP uploads can be removed after the response.

Success retains `result.json`, including available alignments, and
`transcript.txt`, containing plain text. Both files have content fingerprints.
Cancellation and failure retain their own terminal states. If the process ends
without a terminal write, inspection takes the abandoned lease and records an
interrupted outcome. Unknown schema versions and corrupt records remain untouched.

Retry verifies the retained audio and local metadata, then starts a new sibling
run with a parent ID. It uses the saved backend and provider without rerouting,
and leaves the parent unchanged. The CLI obtains one machine reservation for
the retry; API requests use their existing request slot and resident executor.

Metadata checks cover the captured configuration, tokenizer files, installation
manifest, and Core ML manifest when applicable. They record absent metadata
files too, so a newly added manifest counts as a change. These checks do not
hash tensor weights or establish identical numerical results. Native loading
still owns checkpoint validation. Runs without captured local model metadata
can be inspected but cannot be retried through `run retry`.

`MereRunExecution` owns the storage primitives shared with image history.
Operation families retain their typed schemas and retry policy. The CLI's
`RecordedOperationRun` adapter presents both families to inspection and listing;
existing `image_run` JSON fields and `image-run.json` records keep their format.

For commands and retention behavior, see
[Record and retry a transcription](../runtime/speech.md#record-and-retry-a-transcription).

## Validate the boundary

Compile the operation tests without inference dependencies:

```bash
swift build --target AudioCoreTests
```

The repository gate executes operation tests with fixture executors and
CLI/API plan-parity tests. These tests cover backend dispatch, effective inputs,
validation, terminal events, cancellation, and resident request cleanup. Durable
run tests cover retained uploads, output failures, metadata changes, abandoned
leases, future schemas, retry lineage, and existing receipt behavior.
Real-model decoding remains a separate runtime validation step.
