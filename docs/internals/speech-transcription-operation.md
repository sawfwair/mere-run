# Shared speech transcription operation

Use the shared speech operation to transcribe or translate an audio file from a
command, an API adapter, or another executor. `AudioCore` owns the typed plan,
validation, events, and outcome. `AudioSTT` supplies native model resolution
and execution.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `SpeechTranscriptionResolver` (`AudioSTT`) | Select a backend and resolve installed model paths without loading a checkpoint |
| `SpeechTranscriptionOperation` (`AudioCore`) | Validate the plan, run an executor, check cancellation, and report events and the outcome |
| `NativeSpeechTranscriptionExecutor` (`AudioSTT`) | Construct and unload a temporary native generator |
| `SpeechTranscriptionRunSession` (`AudioCore`) | Retain inputs, the effective plan, and outputs for inspection and retry |
| CLI adapter | Transcript formatting, timestamps, receipts, and diagnostics |
| API adapter | Managed-model allowlist, OpenAI model aliases, and JSON, text, SRT, and VTT responses |

## Plan and validate a request

Resolve an `ASRRequest` with `SpeechTranscriptionResolver.resolve`. The
resolver picks a backend from the task, language hint, requested backend, and
model override:

1. An explicit Nemotron Transcribe or Whistle backend wins, after that
   backend validates the task and language.
2. Otherwise, translation, and a language hint that Parakeet doesn't list,
   select Qwen.
3. Otherwise, an explicit backend wins, then the backend that the named model
   belongs to.

When routing changes a built-in model selection, the plan uses the selected
backend's compatible default model. An incompatible explicit local model path
produces a validation error. Core ML execution requires a request that resolves
to Parakeet.

## Execution and runtime lifetime

Pass the resolved plan and a `SpeechTranscriptionExecutor` to
`SpeechTranscriptionOperation.execute`. The operation checks the token limit,
task and backend compatibility, model identity, and input file. It rechecks
the file at execution because the file can disappear after resolution.

`NativeSpeechTranscriptionExecutor` constructs and unloads its own generator.
A resident executor borrows a prepared runtime from its pool. The operation
never unloads a borrowed runtime or acquires another machine reservation. In
the API, the runtime service owns the request slot and the transport removes
the temporary upload after processing.

Cancellation is checked before execution and after the executor returns, so a
cancelled request never becomes a successful outcome, even when the executor
returns without throwing.

Raw stdin and live streaming use the ASR session protocol instead. The file
operation's events don't add a CLI event mode.

## Results and run records

The operation returns `SpeechTranscriptionOutcome`, which includes the run
identity, resolved plan, backend decision, and `ASRResult`. An optional event
handler receives started and progress events followed by one success, failure,
or cancellation event. Input and compatibility errors use
`SpeechTranscriptionIssue` with a stable code and message.

### Retain a run

Pass an optional `SpeechTranscriptionRunSession` to retain a run's input and
outcome. The session creates a private directory and takes a process lease
before it writes a versioned `transcription-run.json` record. It rejects an
existing directory without rewriting its files.

`SpeechTranscriptionRunOptions` preserves the supplied settings. The resolved
plan records the backend decision, model ID and local path, provider, language,
task, and token limit. Before inference, the session copies the input audio and
writes the effective plan. The executor reads that retained copy, so the API
can remove a temporary upload after the response.

Success retains `result.json`, with any available alignments, and
`transcript.txt`, with plain text. Both files carry content fingerprints.
Cancellation and failure record their own terminal states. If the process ends
without a terminal write, inspection takes the abandoned lease and records an
interrupted outcome. Records with unknown schema versions and corrupt records
stay untouched.

### Retry a run

Retry verifies the retained audio and local metadata, then starts a new sibling
run with a parent ID. It reuses the saved backend and provider without
rerouting, and leaves the parent unchanged. The CLI takes one machine
reservation for the retry; API requests use their existing request slot and
resident executor.

Metadata checks cover the captured configuration, tokenizer files, installation
manifest, and, when applicable, the Core ML manifest. They also record absent
metadata files, so a newly added manifest counts as a change. These checks
don't hash tensor weights or guarantee identical numerical results; native
loading still validates the checkpoint. A run without captured local model
metadata can be inspected but not retried with `run retry`.

`MereRunExecution` owns the storage primitives shared with image run records.
Each operation family keeps its typed schema and retry policy. The CLI's
`RecordedOperationRun` adapter presents both families to `run inspect` and
`run list`; `image_run` JSON fields and `image-run.json` records keep their
format.

For commands and retention behavior, see
[Record and retry a transcription](../runtime/speech.md#record-and-retry-a-transcription).

## Validate a change

Compile the operation tests without inference dependencies:

```bash
swift build --target AudioCoreTests
```

The repository gate runs the operation tests with fixture executors and the
CLI and API plan-parity tests. These cover backend dispatch, effective inputs,
validation, terminal events, cancellation, and resident request cleanup.
Durable-run tests cover retained uploads, output failures, metadata changes,
abandoned leases, future schemas, retry lineage, and receipt behavior.

Real-model decoding is a separate runtime validation step.
