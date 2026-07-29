# Sortformer Speaker Diarization

This directory owns the native MLX offline speaker-diarization runtime.

- `SortformerConfig.swift`: typed checkpoint configuration.
- `SortformerDSP.swift` and `SortformerFeatures.swift`: MLX STFT and NeMo-compatible filterbank features.
- `SortformerModel.swift`: FastConformer, Transformer, checkpoint loading, inference, and segment post-processing.
- `SortformerDiarizer.swift`: array-based public entrypoint used by the CLI.
- `DiarizationOutput.swift`: backend-neutral segment and RTTM output types.

The implementation is adapted from `Blaizzy/mlx-audio-swift` commit
`4266f988d170a83017d1e82e2e4654602f277f1d` under the MIT License. Keep the
source attribution and `THIRD_PARTY_NOTICES.md` entry when changing this port.

Audio decoding stays in `AudioCodecs`; this directory must not import
AVFoundation so it remains available to the Linux CUDA package.
