# AudioCodecs

Low-level audio file, PCM, WAV, and spectrogram support shared by speech
runtime paths and tests.

- `AudioReader.swift`: audio loading.
- `PCMStreamConverter.swift`: PCM conversion helpers.
- `StreamingWAVWriter.swift`: incremental WAV output.
- `MelSpectrogram*.swift`: mel feature extraction.

Keep codec behavior deterministic and covered by focused tests before changing
runtime callers.
