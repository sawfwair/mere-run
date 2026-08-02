# Audio Enhance

## Purpose

Extend narrowband speech or bandwidth-limited general audio to a hashed 48 kHz
mono WAV with native Swift/MLX inference. No audio is uploaded and no Python
runtime is launched.

`mere.run audio enhance` extends 16 kHz narrowband speech to 48 kHz with the
pinned AP-BWE checkpoint. Inference stays local and uses the native Swift/MLX
runtime. Select UniverSR for speech, music, or sound effects with an effective
8, 12, 16, or 24 kHz input bandwidth.

## Install

```bash
mere.run model pull audio-enhance-ap-bwe-16kto48k
mere.run model pull audio-enhance-universr-audio
```

AP-BWE's source code and published checkpoint are MIT-licensed. UniverSR's
source code is MIT-licensed; its official `universr-audio` checkpoint is CC BY
4.0. Managed snapshots pin the corresponding source revision, artifact
revision, model metadata, configuration, byte counts, and SHA-256 digests.

## Run

```bash
mere.run audio enhance ./speech.wav
mere.run audio enhance ./speech.mp3 --output ./speech-wideband.wav
mere.run audio enhance ./speech.wav --dtype float16 --overlap 4
mere.run audio enhance ./music-12k.wav \
  --model audio-enhance-universr-audio \
  --output ./music-super-resolved.wav
mere.run audio enhance ./limited-48k.wav \
  --model audio-enhance-universr-audio \
  --input-rate 16000 --seed 42
```

AP-BWE decodes input to 16 kHz mono, performs band-limited 3x interpolation,
and runs its generator at 48 kHz. UniverSR decodes at the declared effective
input rate, upsamples to 48 kHz, and reconstructs high-frequency content with
the published flow-matching defaults: midpoint integration, four steps,
guidance scale 1.5, and seed 42. Both routes write a mono float WAV plus a
sibling JSON provenance manifest. The same manifest is emitted on stdout;
progress is diagnostic stderr output.

## Notes

- AP-BWE is intended for speech bandwidth extension. UniverSR is a general
  audio super-resolution model, not a full mastering chain.
- UniverSR infers the effective input rate from native 8/12/16/24 kHz files.
  For bandwidth-limited audio already stored at 48 kHz, pass `--input-rate`.
- `--ode-method`, `--ode-steps`, `--guidance-scale`, `--seed`, and
  `--chunk-seconds` apply only to UniverSR; `--overlap` applies only to AP-BWE.
- `--model-path` accepts an explicit root only when every pinned artifact matches.
- `--dtype float32` is the default; `float16` reduces model memory use.
- Quality depends on the source. A valid output artifact proves runtime
  execution, not perceptual improvement for every recording.

## Sources

- https://github.com/yxlu-0102/AP-BWE
- https://huggingface.co/rsxdalv/AP-BWE
- https://github.com/woongzip1/UniverSR
- https://huggingface.co/woongzip1/universr-audio
