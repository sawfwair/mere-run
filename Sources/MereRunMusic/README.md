# MereRunMusic

Music generation, sound-effects synthesis, stem separation, music transcription, and audio super-resolution runtimes carved out of `MereRunCore`.

- `ACEStep/`, `MiniMaxMusic3/`, `YuE2/`, and `MagentaRT2/`: music generation, adapter training, and realtime music synthesis (`magentart.xcframework` is macOS-only and scoped to this target).
- `MuScriptor/` and `RoFormer/`: music transcription and BS-RoFormer / MelBand-RoFormer stem separation.
- `MMAudio/` and `Woosh/`: video-to-audio and text-to-SFX generation.
- `APBWE/` and `UniverSR/`: audio bandwidth extension and restoration.
- `MusicWaveformAdapter.swift`: shared waveform adaptation over `AudioCore`.

Model catalog metadata, checkpoint pins, and bundled config definitions (`*Resources`, `*Configuration`, `ACEStepTask`) remain in `MereRunCore` so model discovery and validation stay centralized while `MereRunCore` no longer depends on `AudioCore`, `MereRunSheetSageModel`, or `magentart`.
