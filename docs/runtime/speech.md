# Speech Runtime

Read text aloud, clone a voice from a short reference clip and save it for
reuse, and transcribe either a file or a live microphone. Two ASR backends are
available and the CLI picks between them by task — none of the audio leaves the
machine in either direction.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run speech synthesize` | Generate speech from text using Qwen3-TTS. |
| `mere.run speech transcribe` | Transcribe or translate speech to text using native ASR backends. |
| `mere.run speech listen` | Transcribe a macOS microphone with live Qwen ASR. |
| `mere.run speech profile create` | Create a reusable voice profile from reference audio. |
| `mere.run speech profile list` | List saved speech voice profiles. |
| `mere.run speech profile delete` | Delete a speech profile by id. |

## Model families

### Text-to-speech

- `speech-tts-qwen3-nano`
- `speech-tts-qwen3-customvoice`

### Speech-to-text

- `speech-asr-qwen3`
- `speech-asr-parakeet`

## Typical workflows

### Synthesize speech

```bash
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --output ./hello.wav
```

### Transcribe audio

```bash
swift run mere.run speech transcribe ./hello.wav --backend auto
```

In automatic mode, transcription prefers Parakeet while translation routes to
Qwen. Streaming transcription uses the selected backend and accepts raw
`pcm-s16le/16000/mono` on stdin. Use `--backend parakeet` or `--backend qwen`
to pin it explicitly:

```bash
audio-source | swift run mere.run speech transcribe - \
  --stream --input-format pcm-s16le --sample-rate 16000 --jsonl
audio-source | swift run mere.run speech transcribe - \
  --stream --backend parakeet \
  --input-format pcm-s16le --sample-rate 16000 --jsonl
swift run mere.run speech listen --list-devices
swift run mere.run speech listen --device <core-audio-uid>
```

`speech listen` remains the Qwen-backed macOS microphone convenience command.

### Manage voice profiles

```bash
swift run mere.run speech profile list
swift run mere.run speech profile create --name narrator --audio ./reference.wav
# `profile list` prints each profile's UUID; pass it to delete.
swift run mere.run speech profile delete --id <profile-uuid>
```

### Voice cloning

`speech synthesize` defaults to `--mode style`, which renders the `--voice`
description. Pass `--mode clone` with either a saved profile or ad-hoc
reference audio:

```bash
# Clone a saved profile (id or name).
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --mode clone --profile narrator \
  --output ./cloned.wav

# Clone ad-hoc reference audio and save it as a reusable profile.
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --mode clone --ref-audio ./ref.wav \
  --ref-text "Transcript of the reference audio." \
  --save-profile narrator \
  --output ./cloned.wav
```

If `--ref-text` is omitted, the reference audio is auto-transcribed with the
speech transcriber. `--language` hints the language (default `auto`). Add
`--stream` to emit audio incrementally while generating; `--stream-chunk-tokens`
sets the chunk interval (default 25).

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/SpeechSynthesizeCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechTranscribeCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechListenCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechProfileCommand.swift` (the `list`, `create`, and `delete` profile subcommands)

### TTS runtime

- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Loading.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Generation.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Support.swift`

Tokenizer internals:

- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSSpeechTokenizer.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSSpeechTokenizer+Encoder.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSSpeechTokenizer+Decoder.swift`

### STT runtime

- `Sources/AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift`
- `Sources/AudioSTT/Qwen3ASR/Qwen3ASRLiveSession.swift`
- `Sources/AudioSTT/Parakeet/ParakeetGenerator.swift`
- `Sources/AudioSTT/Parakeet/ParakeetASRLiveSession.swift`

## How speech synthesis flows

At a high level:

1. the CLI resolves the chosen TTS model
2. optional profile or voice configuration is loaded
3. text is converted into the model’s intermediate token representation
4. the generator produces waveform data
5. audio is written to disk through the codec/output helpers

`MediaAudioIO` can write float WAV output incrementally from chunk providers,
including device-backed producers, which avoids constructing a second
whole-file `Data` copy. Individual TTS generators may still produce a complete
waveform before handing it to the output helper; the streaming writer API does
not by itself make every synthesis model incremental.

## How transcription flows

1. audio input is decoded into the expected local format
2. the selected backend loads its model components
3. the backend produces a transcript
4. the CLI prints or writes the output

## Notes for contributors

- speech code spans both `AudioTTS` and `AudioSTT`; do not assume it all lives
  under `MereRunCore`
- profile management is CLI-facing but depends on the same canonical model and
  model-store conventions as the rest of the repo

See [Architecture Reading Map](../architecture.md) for a recommended reading order.
