# AudioCore

Shared audio-domain types and streaming primitives used by speech synthesis,
speech transcription, and CLI streaming sessions.

- `AudioExportPlan.swift`, `AudioExportProcessor.swift`, and `AudioWAVEncoder.swift`: validated WAV export, explicit interleaved waveforms, fades, normalization, statistics, and atomic chunked file encoding. These types have no MLX dependency.
- `AudioGeneration.swift`: request, response, progress, and streaming protocols.
- `SpeechSynthesisPlan.swift` and `SpeechSynthesisOperation.swift`: validated synthesis requests, waveform executors, and completed audio artifacts.
- `AudioExportStream.swift`: incremental WAV encoding with explicit completion and atomic publication; cancellation preserves an existing destination.
- `ASRBackendRouting.swift`: speech-to-text backend selection policy.
- `SpeechTranscriptionOperation.swift`: typed file transcription plans, validation, executors, events, and outcomes.
- `SpeechTranscriptionRunRecord.swift` and `SpeechTranscriptionRunSession.swift`: retained file inputs, versioned outcomes, recovery, and retry.
- `ParakeetExecutionProvider.swift`: portable provider selection; AudioSTT resolves the assets.
- `StreamingSessionUtilities.swift`: cadence and partial/final emission helpers.

Keep this module backend-neutral. Runtime-specific model code belongs in
`AudioSTT` or `AudioTTS`.

Run storage uses `MereRunExecution` for process leases, atomic writes, file hashes,
and terminal states. `AudioCoreTests` and `MereRunExecutionTests` run without
model runtimes, Core, or CLI dependencies.
