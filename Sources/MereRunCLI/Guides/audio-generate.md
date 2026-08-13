# Audio Generate

## Purpose

Generate audio directly from text with the native LTX 2.5 audio diffusion
pipeline. No video tensor is constructed or denoised.

## Model

```bash
mere.run model pull video-ltx25-full-bf16 --accept-model-license
```

The complete LTX 2.5 root is required because text-to-audio uses the dev
transformer, Gemma 4 contexts, audio VAE/vocoder, and DurationHead.

## Examples

```bash
# DurationHead chooses a legal 8n+1 clock inside the official 1...20s range.
mere.run audio generate \
  "ocean surf, distant gulls, and a wooden buoy bell" \
  --output ./surf.wav

# Fixed duration and custom guidance.
mere.run audio generate \
  "a short cinematic impact with a deep metallic tail" \
  --duration 8 \
  --audio-cfg-guidance-scale 8 \
  --output ./impact.wav

# Explicit model-clock frames and fractional time base.
mere.run audio generate \
  "quiet room tone beneath sparse dialogue" \
  --num-frames 121 --fps 23.976 \
  --output ./room.wav
```

`--auto-duration MIN MAX` overrides the default prediction range. An explicit
`--num-frames` wins and causes `--auto-duration` to be ignored with a warning.
Use repeatable `--lora PATH[=STRENGTH]`, `--sigmas`, CFG/STG controls, or
`--enhance-prompt` for the corresponding official upstream features.

The command writes the WAV path to stdout and progress to stderr.
