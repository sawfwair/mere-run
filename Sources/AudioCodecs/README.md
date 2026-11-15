# AudioCodecs

Low-level audio file, PCM, WAV, and spectrogram support shared by speech
runtime paths and tests.

- `AudioReader.swift`: audio loading.
- `PCMStreamConverter.swift`: PCM conversion helpers.
- `StreamingWAVWriter.swift`: incremental WAV output.
- `MelSpectrogram*.swift`: mel feature extraction.

Keep codec behavior deterministic and covered by focused tests before changing
runtime callers.

The Whisper mel extractor stores each filter's nonzero frequency range and
preserves accumulation order while skipping zero coefficients. Its FFT input
buffer is reused between frames. Batch and incremental extraction share those
same coefficients and normalization rules.
