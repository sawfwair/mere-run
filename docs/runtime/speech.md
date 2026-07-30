# Speech Runtime

Read text aloud, clone a voice from a short reference clip and save it for
reuse, transcribe either a file or a live microphone, and identify who spoke
when in a recording. Two ASR backends and one speaker-diarization backend are
available — none of the audio leaves the machine.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run speech synthesize` | Generate speech from text using Qwen3-TTS. |
| `mere.run speech transcribe` | Transcribe or translate speech to text using native ASR backends. |
| `mere.run speech diarize` | Identify speaker time ranges as versioned JSON or RTTM using native MLX Sortformer. |
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

### Speaker diarization

- `speech-diarization-sortformer` (NVIDIA Streaming Sortformer v2.1, up to four speakers)

## macOS Voice Studio

The Speak and Listen workspaces expose **Voice Studio** as the purpose-built
speech surface. It covers styled and reference/profile-cloned synthesis,
reference recording, reusable profile create/list/delete, streaming chunk
controls and feedback, A/B playback of recent renders, backend/task/language
selection, transcription, and an editable transcript that can be saved as a
durable artifact. Runs use the public CLI contract and stay in Library.

The packaged app declares `NSMicrophoneUsageDescription` and signs both the app
and embedded CLI with the audio-input entitlement. Recording is local; granting
microphone access does not enable a network upload path.

The app's Advanced command catalog also exposes **Diarize speakers** with an
audio picker, managed Sortformer default, and durable JSON output in Library.

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

### Identify speakers

Install the pinned public Sortformer checkpoint:

```bash
mere.run model pull speech-diarization-sortformer
mere.run speech diarize ./meeting.wav --output ./meeting.json
mere.run speech diarize ./meeting.wav --format rttm --output ./meeting.rttm
```

JSON output uses a versioned schema and anonymous, recording-local speaker
labels:

```json
{
  "schema_version": 1,
  "speaker_count": 2,
  "segments": [
    {
      "speaker": "speaker_0",
      "speaker_index": 0,
      "start_seconds": 0,
      "end_seconds": 4.8,
      "duration_seconds": 4.8
    }
  ]
}
```

Sortformer answers “who spoke when”; it does not transcribe the words or infer
real identities. Labels such as `speaker_0` are stable within one recording
but are not identities that carry between recordings. The v2.1 checkpoint has
four output channels, so recordings with more than four distinct speakers are
outside this model's supported range. This first runtime is offline/file-based;
streaming diarization is not exposed yet.

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
- `Sources/MereRunCLI/Commands/SpeechDiarizeCommand.swift`
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

### Diarization runtime

- `Sources/AudioSTT/Sortformer/SortformerDiarizer.swift`
- `Sources/AudioSTT/Sortformer/SortformerModel.swift`
- `Sources/AudioSTT/Sortformer/SortformerFeatures.swift`

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

## How diarization flows

1. audio input is decoded and resampled to 16 kHz mono
2. the MLX feature extractor produces NeMo-compatible mel features
3. Sortformer predicts activity independently across four speaker channels
4. thresholding, minimum-duration filtering, and same-speaker gap merging
   produce time segments
5. the CLI emits versioned JSON or standard RTTM text

## Notes for contributors

- speech code spans both `AudioTTS` and `AudioSTT`; do not assume it all lives
  under `MereRunCore`
- profile management is CLI-facing but depends on the same canonical model and
  model-store conventions as the rest of the repo

See [Architecture Reading Map](../architecture.md) for a recommended reading order.
