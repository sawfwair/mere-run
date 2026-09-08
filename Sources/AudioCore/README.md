# AudioCore

Shared audio-domain types and streaming primitives used by speech synthesis,
speech transcription, and CLI streaming sessions.

- `AudioGeneration.swift`: request, response, progress, and streaming protocols.
- `ASRBackendRouting.swift`: speech-to-text backend selection policy.
- `SpeechTranscriptionOperation.swift`: typed file transcription plans, validation, executors, events, and outcomes.
- `StreamingSessionUtilities.swift`: cadence and partial/final emission helpers.

Keep this module backend-neutral. Runtime-specific model code belongs in
`AudioSTT` or `AudioTTS`.
