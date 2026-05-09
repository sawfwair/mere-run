# Speech Profile

## Purpose

Manage reusable voice clone profiles for speech synthesis. Profiles store reference audio metadata and transcript text so clone mode can be repeated.

## Required Models

Profile creation may transcribe reference audio with ASR. Clone synthesis uses `speech-tts-qwen3-customvoice`.

## Install And Check

```bash
mere.run model pull speech-tts-qwen3-customvoice
mere.run speech profile --help
mere.run speech profile list
```

## Parameters

`speech profile list` has no options.

`speech profile create`:

- `--name`: profile name.
- `--audio`: reference audio file.
- `--text`: reference transcript override.
- `--language`: language code or `auto`.
- `--quiet`, `-q`: suppress progress.

`speech profile delete`:

- `--id`: profile UUID.

## Usage Patterns

- Use short, clean reference audio with one speaker and little background noise.
- Provide `--text` when the reference transcript matters.
- Use memorable names for profiles and keep UUIDs for automation.
- After creating a profile, test it with `speech synthesize --mode clone --profile <name>`.

## Examples

```bash
mere.run speech profile create \
  --name narrator \
  --audio ./reference.wav \
  --text "This is the exact transcript spoken in the reference audio." \
  --language en
```

```bash
mere.run speech synthesize \
  "A short profile test." \
  --mode clone \
  --profile narrator \
  --model speech-tts-qwen3-customvoice \
  --output ./profile-test.wav
```

## Iteration Tips

- Replace noisy profiles instead of fighting them in prompts.
- Keep one neutral profile and one expressive profile per speaker.
- Recreate profiles when the transcript or language hint was wrong.

## Troubleshooting

- Auto-transcription is empty: pass `--text`.
- Profile not found: run `mere.run speech profile list`.
- Delete fails: use the UUID printed by `list`, not the display name.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/SpeechProfileCommand.swift
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3
