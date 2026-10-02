# Nemotron 3 precision and returning-speaker regressions

`checkpoint-precision.pt` is a synthetic state dictionary written with PyTorch
2.9.1. Its encoder values are FP32 and cannot survive conversion to BF16 unchanged;
its speaker head is stored in BF16, as in the released initializer. Both must
load into FP32 model weights, matching NeMo restore behavior. The fixture
checks the production checkpoint loader, including exclusion of preprocessing
buffers and the unused auxiliary activity head. Regenerate it with
`generate-precision-fixture.py`.

`returning-speaker.wav` is a 46.57-second, 16 kHz mono PCM A-B-A recording derived
from Mini LibriSpeech SLR31, by Vassil Panayotov, Guoguo Chen, Daniel Povey, and
Sanjeev Khudanpur. Source: https://www.openslr.org/31/ . License:
https://creativecommons.org/licenses/by/4.0/ . The publisher's `dev-clean-2.tar.gz`
MD5 is `6d7ab67ac6a1d2c993d050e16d61080d`.

The transformation concatenates these clips, inserting 0.5 seconds of silence
between turns and converting them to PCM WAV with FFmpeg:

- A: `5694/64038/5694-64038-0022.flac` (0–15.77 seconds).
- B: `1272/135031/1272-135031-0024.flac` (16.27–30.74 seconds).
- A: `5694/64038/5694-64038-0017.flac` (31.24–46.57 seconds).

WAV SHA-256: `042749dfa02b8f159228426d8aba1a7c3b6f8ee293a2c5349a770cd75b672189`.
The opt-in real-checkpoint test requires substantial speech coverage for each
known turn and preservation of A's identity across the default chunk boundary.
It does not claim general diarization accuracy.
