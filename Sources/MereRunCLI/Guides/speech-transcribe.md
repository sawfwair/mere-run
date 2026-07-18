# Speech Transcribe

## Purpose

Transcribe or translate speech from a WAV file using native ASR backends. Auto mode prefers Parakeet for transcription and Qwen for translation.

## Required Models

- `speech-asr-parakeet` for fast transcription.
- `speech-asr-qwen3` for quality-first transcription and translation.

## Install And Check

```bash
mere.run model pull speech-asr-parakeet
mere.run model pull speech-asr-qwen3
mere.run speech transcribe --help
```

## Parameters

- positional audio: input WAV, 16 kHz recommended.
- `--output`, `-o`: optional transcript path.
- `--model`, `-m`: model id or local model path.
- `--backend`: `auto`, `parakeet`, or `qwen`.
- `--task`: `transcribe` or `translate`.
- `--language`: optional language hint.
- `--max-tokens`: generation cap, default `448`.
- `--stream`: streaming ASR mode using the selected backend.
- `--stream-chunk-ms`: audio feed chunk size for streaming.
- `--stream-decode-ms`: decode interval for streaming.
- `--timestamps`, `--no-timestamps`: include alignment lines when available.
- `--quiet`, `-q`: suppress progress.

## Usage Patterns

- Start with `--backend auto`.
- Use `--backend parakeet` for normal transcription where speed matters.
- Live transcription follows the same policy: `auto` selects Parakeet for
  transcription and Qwen for translation.
- Use `--task translate --backend auto` for translation.
- Use `--language en` or another language hint when the audio is short or ambiguous.

## Examples

```bash
mere.run speech transcribe ./meeting.wav --backend auto --output ./meeting.txt
```

```bash
cat ./audio.pcm | mere.run speech transcribe - \
  --stream \
  --backend parakeet \
  --input-format pcm-s16le \
  --sample-rate 16000 \
  --jsonl
```

```bash
mere.run speech transcribe ./spanish.wav \
  --task translate \
  --backend qwen \
  --model speech-asr-qwen3
```

## Iteration Tips

- Clean or normalize audio before changing model settings.
- Disable timestamps when you only need prose.
- For streaming demos, tune chunk and decode intervals together.

## Troubleshooting

- Streaming backpressure errors: pace PCM input in real time and keep no more
  than five seconds queued.
- Bad transcript on noisy audio: preprocess audio or try the other backend.
- Translation with Parakeet requested: use Qwen or `--backend auto`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/SpeechTranscribeCommand.swift
- https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3
- https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit
