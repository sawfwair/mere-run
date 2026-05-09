# AudioCore

Shared audio-domain types and streaming primitives used by speech synthesis,
speech transcription, and CLI streaming sessions.

- `AudioGeneration.swift`: request, response, progress, and streaming protocols.
- `ASRBackendRouting.swift`: speech-to-text backend selection policy.
- `StreamingSessionUtilities.swift`: cadence and partial/final emission helpers.

Keep this module backend-neutral. Runtime-specific model code belongs in
`AudioSTT` or `AudioTTS`.
