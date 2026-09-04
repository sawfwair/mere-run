# Speech runtime

Read text aloud, clone a voice from a short reference clip and save it for
reuse, transcribe either a file or a live microphone, and identify who spoke
when in a recording. Two automatic speech recognition (ASR) backends and one
speaker-diarization backend are available. These operations run locally, so
the audio does not leave the machine.

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

## macOS Studio

Speech is split across two domains. **Voice ▸ Speak** is the prompt task for
synthesis, with **Voice ▸ Clone** and **Voice ▸ Voices** covering styled and
reference-cloned synthesis, reference recording, reusable profile create, list,
and delete, streaming chunk controls and feedback, A/B playback of recent
renders, and backend, task, and language selection.

**Audio ▸ Transcribe** shows the timestamped transcript beside the recording's
waveform, with a Timeline and a raw JSON view, and saves the transcript as a
durable artifact. **Audio ▸ Who Spoke** is `speech diarize`, with an audio
picker, the managed Sortformer default, JSON and RTTM timelines, and
segment-tuning controls. **Audio ▸ Live** is `speech listen`: it enumerates
capture devices through the CLI and streams partial transcripts as the
recognizer emits them, with an operator-owned stop. Every run uses the public
CLI contract and stays in the Library.

The packaged app declares `NSMicrophoneUsageDescription` and signs both the app
and embedded CLI with the audio-input entitlement. Recording is local; granting
microphone access does not enable a network upload path.

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

#### Optional Parakeet Core ML/MLX package

Parakeet uses MLX by default. For controlled Apple-platform experiments, Mere
also provides a pinned conversion tool and a typed Core ML encoder provider:

```bash
uv run --script scripts/model-conversion/convert_parakeet_coreml.py --plan
uv run --script scripts/model-conversion/convert_parakeet_coreml.py \
  --workspace /path/to/parakeet-conversion-workspace \
  --output /path/to/parakeet-coreml

mere.run speech transcribe ./short.wav \
  --backend parakeet \
  --provider coreml \
  --coreml-encoder /path/to/parakeet-coreml
```

An explicit Core ML request fails if its language hint would route to Qwen.
To use Qwen, select `--backend qwen` and omit `--provider coreml` and
`--coreml-encoder`.

The converter downloads NVIDIA's exact
`nvidia/parakeet-tdt-0.6b-v3` revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, verifies the source file sizes and
SHA-256 values, exports the FastConformer encoder with pinned tool versions,
and exports the TDT decoder and joint network as a second Core ML model. The
package retains the 13 required decoder and joint tensors in a 69 MiB MLX
fallback checkpoint. `parakeet-coreml.json` records the complete Core ML
models, decoder embedding table, generated runtime configuration, vocabulary,
tokenizer, and notice closure. The
runtime verifies that closure before loading the model. The package uses Mere's
own runtime integration and does not require a third-party inference SDK,
telemetry, or an opaque model download.

This provider is limited to non-streaming files. Each Core ML encoder call has
batch size 1 and a maximum 15-second input. Longer files use 15-second windows
with two seconds of overlap. The runtime decodes as many as 16 windows in one
Core ML call, then reconciles matching token IDs and global timestamps at each
boundary. Mere's native Swift/Accelerate feature extractor remains in the path,
but the standalone artifact no longer requires `speech-asr-parakeet` or its
complete 2.3 GiB weight file.

Core ML is configured for CPU and Neural Engine execution. That preference
doesn't prove that every operation runs on the Neural Engine or that the path
is faster or more accurate. Treat performance, placement, boundary merging,
and transcript parity as unqualified until you measure them on the target Mac.
The conversion needs at least 10 GiB of free working space and the exact Xcode
version printed by `--plan`.

To report resident model-load, feature extraction, encoder, decoder, alignment,
merge, and total timings, use an optimized executable:

```bash
.build/release/mere.run model benchmark parakeet-coreml ./sample.wav \
  --artifact /path/to/parakeet-coreml \
  --warmups 2 \
  --repetitions 5 \
  --json
```

The benchmark records transcript consistency across repetitions. It doesn't
compare transcript quality with a reference corpus.

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
outside this model's supported range. This runtime is offline and file-based.
Streaming diarization is not supported.

### Manage voice profiles

```bash
swift run mere.run speech profile list
swift run mere.run speech profile create --name narrator --audio ./reference.wav
# `profile list` prints each profile's UUID; pass it to delete.
swift run mere.run speech profile delete --id <profile-uuid>
```

### Voice cloning

`speech synthesize` defaults to `--mode style`, which renders the `--voice`
description. Pass `--mode clone` with either a saved profile or one-time
reference audio:

```bash
# Clone a saved profile (id or name).
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --mode clone --profile narrator \
  --output ./cloned.wav

# Clone one-time reference audio and save it as a reusable profile.
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --mode clone --ref-audio ./ref.wav \
  --ref-text "Transcript of the reference audio." \
  --save-profile narrator \
  --output ./cloned.wav
```

If `--ref-text` is omitted, the speech transcriber automatically transcribes the
speech transcriber. `--language` hints the language (default `auto`). Add
`--stream` to emit audio incrementally while generating; `--stream-chunk-tokens`
sets the chunk interval (default 25).

## Runtime entry points

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
- `Sources/AudioSTT/Parakeet/ParakeetCoreMLEncoder.swift`
- `Sources/AudioSTT/Parakeet/ParakeetASRLiveSession.swift`

### Diarization runtime

- `Sources/AudioSTT/Sortformer/SortformerDiarizer.swift`
- `Sources/AudioSTT/Sortformer/SortformerModel.swift`
- `Sources/AudioSTT/Sortformer/SortformerFeatures.swift`

## How speech synthesis flows

At a high level:

1. The CLI resolves the selected text-to-speech (TTS) model.
2. The runtime loads the optional profile or voice configuration.
3. The runtime converts text into the model's intermediate token representation.
4. The generator produces waveform data.
5. The codec and output helpers write the audio to disk.

`MediaAudioIO` can write float WAV output incrementally from chunk providers,
including device-backed producers, which avoids constructing a second
whole-file `Data` copy. Individual TTS generators may still produce a complete
waveform before handing it to the output helper; the streaming writer API does
not by itself make every synthesis model incremental.

## How transcription flows

1. The runtime decodes audio input into the expected local format.
2. The selected backend loads its model components.
3. The backend produces a transcript.
4. The CLI prints or writes the output.

## How diarization flows

1. The runtime decodes and resamples audio input to 16 kHz mono.
2. The MLX feature extractor produces NeMo-compatible mel features.
3. Sortformer predicts activity independently across four speaker channels.
4. thresholding, minimum-duration filtering, and same-speaker gap merging
   produce time segments.
5. The CLI emits versioned JSON or standard RTTM text.

## Notes for contributors

- Speech code spans both `AudioTTS` and `AudioSTT`. Do not assume it all lives
  under `MereRunCore`.
- Profile management is CLI-facing but depends on the same canonical model and
  model-store conventions as the rest of the repository.

See the [architecture map](../architecture.md) for a recommended reading order.
