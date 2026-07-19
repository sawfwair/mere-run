# Speech Runtime

This page covers speech synthesis, transcription, and voice-profile management.

## Public surface

- `mere.run speech synthesize`
- `mere.run speech transcribe`
- `mere.run speech listen`
- `mere.run speech profile list`
- `mere.run speech profile create`
- `mere.run speech profile delete`

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
swift run mere.run speech profile create ./reference.wav --name narrator
swift run mere.run speech profile delete narrator
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/SpeechSynthesizeCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechTranscribeCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechListenCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechProfileListCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechProfileCreateCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechProfileDeleteCommand.swift`

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
