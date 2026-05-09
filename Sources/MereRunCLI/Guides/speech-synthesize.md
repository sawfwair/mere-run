# Speech Synthesize

## Purpose

Generate WAV speech from text with Qwen3-TTS. Use style mode for described voices and clone mode when a reference profile or audio file should guide timbre.

## Required Models

- `speech-tts-qwen3-nano` for voice design/style.
- `speech-tts-qwen3-customvoice` for clone workflows.

## Install And Check

```bash
mere.run model pull speech-tts-qwen3-nano
mere.run speech synthesize --help
```

## Parameters

- positional text: text to speak.
- `--output`, `-o`: required WAV path.
- `--model`, `-m`: model id or local path.
- `--voice`, `-v`: natural-language voice description.
- `--mode`: `style` or `clone`.
- `--profile`: saved clone profile id or name.
- `--ref-audio`: reference audio path for clone mode.
- `--ref-text`: transcript override for reference audio.
- `--language`: language hint, default `auto`.
- `--save-profile`: save clone reference for reuse.
- `--temperature`: sampling temperature, default `0.6`.
- `--stream`: stream audio chunks to the WAV writer.
- `--stream-chunk-tokens`: token interval for streaming chunks.
- `--quiet`, `-q`: suppress progress.

## Prompting Patterns

- Put delivery style in `--voice`: age range, accent, energy, pace, emotion, microphone feel.
- Keep synthesis text clean. Expand abbreviations and spell numbers the way they should be read.
- Use `--language` when the text mixes scripts or languages.
- For clone mode, provide a clear transcript with `--ref-text` if auto-transcription could be wrong.

## Examples

```bash
mere.run speech synthesize \
  "Welcome to mere.run." \
  --voice "A calm, warm narrator with clear pronunciation and gentle pacing" \
  --output ./welcome.wav
```

```bash
mere.run speech synthesize \
  "This line uses the saved narrator profile." \
  --mode clone \
  --profile narrator \
  --model speech-tts-qwen3-customvoice \
  --output ./narrator.wav
```

## Iteration Tips

- Change one voice descriptor at a time: pace, emotion, timbre, then accent.
- Lower temperature for steadier delivery; raise slightly for more expressive reads.
- Save strong clone references as profiles so later runs are repeatable.

## Troubleshooting

- Clone mode fails: pass `--profile` or `--ref-audio`.
- Pronunciation is off: rewrite the text phonetically or add `--language`.
- Streaming produces too many chunks: increase `--stream-chunk-tokens`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/SpeechSynthesizeCommand.swift
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
