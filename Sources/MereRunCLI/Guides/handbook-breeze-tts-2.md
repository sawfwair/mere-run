# Breeze TTS 2 (BreezeBlue)

## Purpose

Use native Breeze TTS 2 speech synthesis for voice design or voice cloning on
Apple Silicon.

## Start here

BreezeBlue restricts the checkpoint weights and self-hosted outputs to research
and noncommercial use. Review the checkpoint license before pulling the model.
Keep the words to speak in the text argument and delivery instructions in
`--voice`.

## Example to adapt

This original example shows the command shape. It is not a recorded model result.

```bash
mere.run model pull speech-tts-breeze-2 --accept-model-license
mere.run speech synthesize "Welcome to the lab." \
  --model speech-tts-breeze-2 \
  --voice "A clear, calm narrator." \
  --output welcome.wav
```

## Controls and variants

For voice cloning, pass `--mode clone`, `--ref-audio`, and `--ref-text` with the
exact words spoken in the reference recording. A saved speech profile can
supply the reference instead. Use `--voice` to add a delivery instruction.
The model does not provide named `--speaker` voices. Use `--stream` to receive
audio chunks during generation.

## Iterate and review

Listen to the output before using it. If a clone mispronounces words or drifts
from the reference, use a clean single-speaker recording with an accurate
transcript and try a shorter text segment.

## Read this guide offline

The guide is bundled with mere.run. Reading it does not load model weights.

```bash
mere.run guide --model speech-tts-breeze-2
mere.run speech synthesize --help
```

## Covered models

This guide covers the following managed model ID:

- `speech-tts-breeze-2`

## Sources and validation

The source and checkpoint are separate: the upstream source is Apache 2.0,
while the weights and self-hosted outputs have a research and noncommercial
restriction. The native runtime was checked with the pinned checkpoint for
style synthesis and reference cloning. Listen to each generated result before
relying on it.

Editorial review date: September 26, 2026.

- [Breeze TTS 2 checkpoint and license](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
- [Breeze TTS source](https://github.com/breezeblue-ai/breeze-tts)
- [Speech runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/speech.md)
