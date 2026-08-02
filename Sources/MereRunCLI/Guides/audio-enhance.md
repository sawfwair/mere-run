# Audio Enhance

## Purpose

Extend narrowband speech to a hashed 48 kHz mono WAV with native Swift/MLX
AP-BWE inference. No audio is uploaded and no Python runtime is launched.

`mere.run audio enhance` extends 16 kHz narrowband speech to 48 kHz with the
pinned AP-BWE checkpoint. Inference stays local and uses the native Swift/MLX
runtime.

## Install

```bash
mere.run model pull audio-enhance-ap-bwe-16kto48k
```

The source code and published checkpoint are MIT-licensed. The managed snapshot
contains the upstream code and weights license texts and exact, hash-verified
16→48 kHz checkpoint/configuration artifacts.

## Run

```bash
mere.run audio enhance ./speech.wav
mere.run audio enhance ./speech.mp3 --output ./speech-wideband.wav
mere.run audio enhance ./speech.wav --dtype float16 --overlap 4
```

The command decodes input to 16 kHz mono, performs band-limited 3x
interpolation, and runs the AP-BWE generator at 48 kHz. It writes a mono float
WAV plus a sibling JSON provenance manifest. The same manifest is emitted on
stdout; progress is diagnostic stderr output.

## Notes

- This profile is intended for speech bandwidth extension, not music mastering.
- `--model-path` accepts an explicit root only when every pinned artifact matches.
- `--dtype float32` is the default; `float16` reduces model memory use.
- Quality depends on the source. A valid output artifact proves runtime
  execution, not perceptual improvement for every recording.

## Sources

- https://github.com/yxlu-0102/AP-BWE
- https://huggingface.co/rsxdalv/AP-BWE
