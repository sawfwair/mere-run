# Speech runtime

Read text aloud, clone a voice from a short reference clip and save it for
reuse, transcribe a file or a live microphone, and identify who spoke when in a
recording. Four automatic speech recognition (ASR) backends and two
speaker-diarization backends are available. Every operation runs locally, so
the audio doesn't leave the machine.

Two ASR backends have their own pages:

- [Native Whistle](./whistle.md) covers the compact seven-language model,
  beam search, keyword biasing, and live transcription.
- [Nemotron 3.5 Transcribe](./nemotron-transcribe.md) covers the large
  file and per-utterance live model, its INT4 default and BF16 option, and its
  measured tradeoffs.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run speech synthesize` | Generate speech from text using Qwen3-TTS, Breeze TTS 2, Phonon 31m, or DRAMA-3. |
| `mere.run speech transcribe` | Transcribe or translate speech to text using native ASR backends. |
| `mere.run speech diarize` | Identify speaker time ranges as versioned JSON or RTTM using native MLX diarization. |
| `mere.run speech diarize-live` | Emit incremental Nemotron 3 speaker activity from a microphone or PCM stdin. |
| `mere.run speech listen` | Transcribe a macOS microphone with live Qwen3-ASR, Parakeet, Whistle, or Nemotron Transcribe. |
| `mere.run speech profile create` | Create a reusable voice profile from reference audio. |
| `mere.run speech profile list` | List saved speech voice profiles. |
| `mere.run speech profile delete` | Delete a speech profile by id. |

## Model families

### Text-to-speech

- `speech-tts-qwen3-nano`
- `speech-tts-qwen3-customvoice`
- `speech-tts-phonon-31m` (English; bundled voice; Apache 2.0)
- `speech-tts-breeze-2` (English and Chinese; research and non-commercial use only)
- `speech-tts-drama-3` (CLI only; Fish Audio Community License; inline delivery and speaker tags)

### Speech-to-text

- `speech-asr-parakeet` (default for automatic transcription)
- `speech-asr-qwen3` (translation and languages Parakeet doesn't recognize)
- [`speech-asr-whistle`](./whistle.md) (packed CQ2/CQ4 transcription in seven
  languages; about 17 MB)
- [`speech-asr-nemotron-35-transcribe`](./nemotron-transcribe.md) (file and
  per-utterance live transcription and English translation with 4-bit routed
  experts; 25.09 GB;
  explicit installation)
- [`speech-asr-nemotron-35-transcribe-bf16`](./nemotron-transcribe.md) (the
  original BF16 checkpoint; 67.5 GB; explicit installation)

### Speaker diarization

- `speech-diarization-sortformer` (NVIDIA Streaming Sortformer v2.1, up to four speakers)
- `speech-diarization-nemotron3` (NVIDIA Nemotron 3 Diarization, up to eight speakers)

## macOS Studio

Speech is split across two domains. **Voice ▸ Speak** is the prompt task for
synthesis, styled or cloned: attaching a reference recording to its composer
well (or picking a saved voice in the inspector) is clone mode, with streaming
chunk controls and language selection. For Breeze TTS 2 cloning, enter the exact
reference transcript in the inspector. Breeze TTS 2 also accepts a voice
instruction with a reference recording for voice direction. With the CustomVoice
model selected, the inspector offers its named speakers. **Voice ▸ Voices** is the Manage page
for saved voices — a list, a detail that plays the reference and shows its
transcript, Delete behind a confirmation, and New voice, which records or
attaches a reference and runs `speech profile create`.

**Audio ▸ Transcribe** shows the timestamped transcript beside the recording's
waveform, with a Timeline and a raw JSON view, and saves the transcript as a
durable artifact. **Audio ▸ Who Spoke** is `speech diarize` on the same
surface: the recording in the well, Sortformer or Nemotron 3 from the model
chip, the output format and Nemotron 3 input buffer as chips, the segment
controls in the inspector, and a JSON or RTTM timeline drawn as speaker lanes
and turn rows. **Audio ▸ Live** is a Session page over `speech listen`
transcription and `speech diarize-live` speaker activity: it lists capture
devices through the CLI, streams results as they arrive, stops the CLI the way
Ctrl-C does, and files the session's text as the run's artifact. Every run
uses the public CLI contract and stays in the Library.

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

Breeze TTS 2 is a native Swift/MLX alternative for English and Chinese. Review
its [research and non-commercial license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)
before pulling the pinned checkpoint. Its weights and self-hosted outputs have
use restrictions. The CLI does not download this model automatically.

```bash
mere.run model pull speech-tts-breeze-2 --accept-model-license
mere.run speech synthesize "Welcome to mere.run." \
  --model speech-tts-breeze-2 --voice "A warm, steady narrator" \
  --output ./breeze.wav
mere.run speech synthesize "I am speaking with a new direction." \
  --model speech-tts-breeze-2 --mode clone \
  --ref-audio ./reference.wav --ref-text "The exact words in the reference." \
  --voice "Speak softly and slowly" --output ./directed.wav
```

Breeze TTS 2 needs the exact reference transcript for cloning. It produces
24 kHz WAV output and supports `--stream` for incremental audio chunks. For
smaller first chunks, use `--stream-chunk-tokens 5`. English text can include
vocal event markers such as `(laugh)` and `(sigh)`. The Breeze-only `--seed`
control repeats the sampling sequence; `--cfg-scale` sets voice instruction
guidance from 0 through 20 (default 4). Studio exposes both controls for
Breeze style and clone modes.

`speech-tts-qwen3-customvoice` also speaks as one of its named speakers in
style mode: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`,
`uncle_fu`, and `vivian`. Pass `--speaker`, and use `--voice` only for an
optional delivery instruction; the default voice description is
not sent with a speaker. A local CustomVoice checkpoint's speakers come from
its `config.json` (`talker_config.spk_id`), and an unknown name stops before
the weights load. Other checkpoints, and clone mode, ignore `--speaker` with a
`Warning:` line.

```bash
swift run mere.run speech synthesize \
  "Hello from mere.run" \
  --model speech-tts-qwen3-customvoice --speaker ryan \
  --output ./ryan.wav
```

### Transcribe audio

```bash
swift run mere.run speech transcribe ./hello.wav --backend auto
```

The four ASR backends differ in what they accept:

| Backend | Select it with | Tasks | Live input | Word timestamps |
| --- | --- | --- | --- | --- |
| Parakeet | Default for `--backend auto` transcription | Transcribe | `--stream` | Yes |
| Qwen3-ASR | `--backend qwen`, translation, or an unrecognized language hint | Transcribe, translate to English | `--stream`, `speech listen` | No |
| Whistle | `--backend whistle` or a Whistle `--model` | Transcribe in seven languages | `--stream`, `speech listen` | Estimated |
| Nemotron 3.5 Transcribe | `--backend nemotron` or a Nemotron `--model` | Transcribe in 40 languages, translate to English | No | No |

Automatic selection never picks Whistle or Nemotron 3.5 Transcribe. Install
them and select them explicitly; see [Native Whistle](./whistle.md) and
[Nemotron 3.5 Transcribe](./nemotron-transcribe.md) for their limits.

The CLI chooses the backend in this order:

1. `--backend whistle` or `--backend nemotron` runs that backend. With
   `--backend auto`, a Whistle or Nemotron Transcribe `--model` does the same.
   Whistle rejects translation, and both reject a `--language` hint they don't
   support, before any model loads.
2. `--task translate` runs Qwen3-ASR, because Parakeet doesn't translate.
3. A `--language` hint that Parakeet's router doesn't recognize runs Qwen3-ASR,
   which supports more languages.
4. `--backend parakeet` or `--backend qwen` runs that backend.
5. A `--model` runs its own backend.
6. Otherwise, Parakeet runs.

When the choice overrules a flag, the flag has no effect, and a managed model
of another backend is replaced by the chosen backend's default. The CLI prints
a `Warning:` line in either case, for example for `--model speech-asr-parakeet
--task translate`. A local model folder of another backend stops the run
instead.

Options outside the chosen backend's scope also produce a warning:

- `--timestamps` and `--no-timestamps` apply to Parakeet and Whistle. Qwen3-ASR
  and Nemotron 3.5 Transcribe return no word timestamps.
- `--max-tokens` (default 448) applies to Qwen3-ASR, Whistle, and Nemotron 3.5
  Transcribe. Whistle caps it at 318 per window. Parakeet ignores it.

Translation always produces English, whatever language the audio is in, so
`--language` has no effect with `--task translate` and the CLI says so.
Qwen3-ASR's model card documents recognition only; mere.run asks the model for
an English translation in the prompt, after the audio, and the answer's
language tag is English. Review translations before relying on them.

```bash
swift run mere.run speech transcribe ./interview-de.wav --task translate
```

Streaming transcription chooses the backend and model the same way, with two
exceptions: a local model folder runs its own backend whatever the `--language`
hint says, and Nemotron 3.5 Transcribe rejects `--stream` before loading. A
stream reads raw `pcm-s16le/16000/mono` on stdin, and its transcript carries no
timestamps. Use `--backend parakeet` or `--backend qwen` to pin the backend:

```bash
audio-source | swift run mere.run speech transcribe - \
  --stream --input-format pcm-s16le --sample-rate 16000 --jsonl
audio-source | swift run mere.run speech transcribe - \
  --stream --backend parakeet \
  --input-format pcm-s16le --sample-rate 16000 --jsonl
swift run mere.run speech listen --list-devices
swift run mere.run speech listen --device <core-audio-uid>
```

`speech listen` is the macOS microphone command. It defaults to Qwen3-ASR;
`--model speech-asr-parakeet` selects Parakeet's live session, and
`--model speech-asr-whistle` selects Whistle with the same decoding controls
as `speech transcribe`. Unsupported Parakeet language hints select Qwen3-ASR,
as they do for streamed transcription. `--model speech-asr-nemotron-35-transcribe`
selects Nemotron Transcribe, which commits text per utterance after a pause.
`--list-devices` lists inputs for `speech listen` and `speech diarize-live`
before any model is read.

#### Record and retry a transcription

To keep the input audio and result for later inspection, use a new run directory:

```bash
mere.run speech transcribe ./meeting.wav --run-dir ./runs/meeting
mere.run run inspect ./runs/meeting --json
mere.run run list --root ./runs --json
mere.run run retry ./runs/meeting --json
```

The directory contains `transcription-run.json`, a retained audio copy under
`inputs`, `result.json` with available alignments, and a plain `transcript.txt`.
Recording is optional. It preserves normal transcript output and the existing
`--receipt` format. An optional `--output` file must be outside the run directory.

Records distinguish success, failure, cancellation, and process interruption.
Inspection marks an abandoned nonterminal record as interrupted when its
directory is writable, and reads finished records without creating lock files.
Retry creates
a new sibling directory with the original run ID recorded as its parent; it
does not modify the original or resume a partially completed decoder.

Retry uses the saved model path, backend, provider, task, language, and token
limit. It verifies retained audio and captured model metadata before running.
If either changed, start a new transcription command. Runs without captured
local model metadata cannot be retried; install the model before recording if
you need that behavior. Model metadata checks do not fingerprint tensor weights
or guarantee identical recognition results.

The API can retain uploaded audio and transcripts with
[`--transcription-run-records`](api-server.md#keep-transcription-run-records).
Run directories remain until you remove them. `--run-dir` supports non-streaming
file input; live stdin and streamed-file sessions keep their existing protocol.

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

Core ML needs `--backend parakeet` and a transcription task, and an explicit
Core ML request fails if its language hint would route to Qwen. Qwen rejects
`--provider coreml` and `--coreml-encoder` before loading, including for
`--task translate`. To use Qwen, select `--backend qwen` and omit both.

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

Schema-v4 packages use floating-point mask construction in the encoder and
grouped token selection in the decoder. These equivalent graphs remove the
integer and boolean operations that caused CPU fallback in schema-v3 packages.
Token IDs use two exact base-128 FP16 digits because a single FP16 value cannot
represent every vocabulary index. The runtime also accepts older packages.

This provider is limited to non-streaming files. Each Core ML encoder call has
batch size 1 and a maximum 15-second input. Longer files use windows of up to
15 seconds. The next window starts in a nearby quiet interval when one is
available, keeping between two and eight seconds of overlap. This reduces
speech loss from starting a window inside a word. Without a quiet interval,
the runtime retains two seconds of overlap. It decodes as many as 16 windows in one
Core ML call, then reconciles matching token IDs and global timestamps at each
boundary. Mere's native Swift/Accelerate feature extractor remains in the path,
but the standalone artifact no longer requires `speech-asr-parakeet` or its
complete 2.3 GiB weight file.

Core ML is configured for CPU and Neural Engine execution. Check the compiled
graphs on the target Mac with the placement audit:

```bash
uv run --script scripts/model-conversion/inspect_parakeet_coreml.py \
  /path/to/parakeet-coreml --require-ane --output placement.json
```

The audit fails if either graph has a nonconstant operation assigned to another
device or an unknown placement. Its compute plan reports anticipated placement;
an Instruments Core ML trace establishes hardware execution. Audio I/O,
feature extraction, decoder orchestration, and transcript merging remain host
work. Treat performance, placement, boundary merging,
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
outside this model's supported range. Sortformer runs offline on files. For
speaker activity during capture, use Nemotron 3 with
[`speech diarize-live`](#diarize-live-audio).

### Diarize with Nemotron 3

Pull the released Nemotron 3 checkpoint after reviewing its OpenMDW-1.1 terms:

```bash
mere.run model pull speech-diarization-nemotron3 --accept-model-license
mere.run speech diarize ./meeting.wav --model speech-diarization-nemotron3 \
  --format json --output ./meeting-speakers.json
```

The native MLX runtime accepts a 16 kHz mono waveform after CLI decoding and
returns anonymous speaker time ranges. It preserves overlapping speakers and
uses cache-aware chunks, so long recordings are not limited to one attention
window. `--latency offline` uses a 30.4-second input buffer. For the released
model's smaller buffer settings, choose `--latency 1.04`, `0.64`, or `0.32`.
These values describe buffered audio, not total processing time. The
four-speaker Sortformer remains the command default, and `--latency` applies
only to Nemotron 3. A local model folder runs Nemotron 3 when it holds
`Nemotron-3-Diarization.nemo`, and Sortformer otherwise. A buffer other than
`offline` for Sortformer stops before any model is loaded.

### Diarize live audio

`diarize-live` keeps the speaker cache and recent context across incoming audio
chunks. It emits `schema_version: 1` JSON Lines events (`ready`, `activity`, and
`final`) before the input ends. Each `activity` event covers a nonoverlapping
time range and includes zero or more anonymous speaker segments. Adjacent
chunks can contain segments from the same continuous turn.

```bash
mere.run speech diarize-live --model speech-diarization-nemotron3 \
  --latency 1.04
```

Use `--device <core-audio-uid>` to choose a microphone or `--list-devices` to
list UIDs. For an external 16 kHz mono PCM source, pass signed 16-bit
little-endian samples on stdin:

```bash
ffmpeg -re -i ./meeting.wav -ar 16000 -ac 1 -f s16le - |
  mere.run speech diarize-live --stdin --latency 1.04
```

The 1.04-second input buffer is the live default. `--latency 0.64` and `0.32`
reduce buffered audio and can reduce speaker consistency. These durations
exclude capture, computation, and output transport. `speech diarize` returns a complete timeline after it processes the whole file;
choose `diarize-live` when results must arrive during capture.

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
reference audio. `--language` hints the language (default `auto`). Add
`--stream` to emit audio incrementally while generating; `--stream-chunk-tokens`
sets the chunk interval (default 25).

Each mode reads its own options. Clone mode ignores `--voice`, and style mode
ignores `--profile`, `--ref-audio`, `--ref-text`, and `--save-profile`; the CLI
prints a `Warning:` line for any it ignores. With `--profile`, clone mode also
ignores `--ref-audio` and `--save-profile`.

## DRAMA-3 on a 36 GB Mac

Use DRAMA-3 for multilingual scenes, inline speaker tags, vocal events, and voice
cloning. The runtime uses Swift and MLX for the text backbone, ten-codebook audio
tower, and 44.1 kHz DAC codec. Built with Fish Audio.

Review the [Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/dd64a85cd82e9521f7202610ede97b367fc83946/LICENSE)
before installing. The managed model requires explicit license acceptance and
supports the local CLI. The API server excludes this model, including local paths.

The managed model downloads the [Sawfwair expert-only Q4 export](https://huggingface.co/Sawfwair/DRAMA-3-MLX-4bit)
at immutable revision `543e9d0829856cc0657fa0fd6bb2903015c094fb`. Reserve at least 28 GB
for the bundle, plus inference headroom. You don't need to download or convert
the original source checkpoint:

```bash
mere.run model pull speech-tts-drama-3 --accept-model-license
mere.run speech synthesize '<|speaker:0|> [sighs] Fine. [laughs] That seems fair.' \
  --model speech-tts-drama-3 --seed 42 --output ./scene.wav
```

The export packs only the routed experts as affine Q4 with groups of 64. It decodes
the original FP8 experts using their 128 by 128 block scales. Attention, routing,
embeddings, and audio-tower weights retain BF16. The codec file is preserved;
native codec execution uses FP32. You don't need Python or an upstream serving
process for inference.

The public bundle includes `LICENSE`, `Notice`, `SHA256SUMS`, the original export
receipt, and bounded qualification receipts. The
[publication receipt](../qualification/drama3-native-20261015/huggingface-publication.json)
records every published file's SHA-256 and size. The Fish Audio source pin and
license acknowledgment remain unchanged.

If you already have the original `fishaudio/drama-3` checkpoint, the loader also
accepts it directly. To export a standalone Q4 directory from that source:

```bash
mere.run model optimize ./drama-3-source --output ./drama-3-q4
```

The source tensors total 39,594,895,616 bytes, plus the 1,871,099,728-byte codec.
The Q4 model and codec total approximately 27,365,269,584 bytes (25.49 GiB), before
tokenizer files, receipts, and runtime allocations. The exporter checks free space,
preserves the source, refuses an existing destination, and records SHA-256 hashes
for source and exported files in `drama3-quantization.json`. Preserve its `LICENSE`
and `Notice` files when moving the checkpoint.

The [released-checkpoint export](../qualification/drama3-native-20261015/README.md)
produced 27,408,212,020 bytes across 99 files and peaked at 2.62 GB of physical
memory during conversion. The reader uses bounded tensor reads on external
volumes; conversion and hashing release temporary buffers within each iteration.
These measurements describe export, not inference memory.

To use an exported checkpoint on an external volume through the model ID:

```bash
mere.run model location bind speech-tts-drama-3 /Volumes/T7/drama-3-q4 \
  --accept-model-license
```

The catalog declares a 32 GB minimum and recommends 36 GB. The minimum remains
an estimate. Seven bounded probes pass on the tested 36 GiB system: batch and
streamed paragraphs, reference conditioning, and cancellation. Peak process
footprint reaches 32.86 GiB for paragraph streaming, with system swap in use.
Cold first audio takes 96–113 seconds. See the
[qualification record](../qualification/drama3-native-20261015/README.md) for
receipts and remaining quality limits.

For cloning, supply a reference clip and its exact transcript:

```bash
mere.run speech synthesize '<|speaker:0|> Welcome to the harbor.' \
  --model ./drama-3-q4 --mode clone --ref-audio ./reference.wav \
  --ref-text 'This is the exact reference transcript.' --output ./clone.wav
```

References must be nonempty and at most 30 seconds. The CLI accepts one reference;
the native prompt builder also accepts multiple references. Use `<|speaker:0|>`
through `<|speaker:15|>` and event tags such as `[sighs]` or `[laughs]` in the text.
DRAMA-3 infers language from text and doesn't use `--voice`, named `--speaker`
values, or classifier-free guidance. `--stream` publishes causal waveform prefixes;
its codec recomputes the accumulated prefix at each chunk. Reaching the acoustic
frame limit fails the operation instead of publishing a truncated final WAV.

## Phonon 31m

Use the Apache-2.0 English checkpoint with its bundled voice:

```bash
mere.run model pull speech-tts-phonon-31m
mere.run speech synthesize "Hello from mere run." --model speech-tts-phonon-31m --seed 42 -o phonon.wav
```

Phonon loads original safetensors through native Swift/MLX and produces 24 kHz
mono audio. Add `--stream` for frame-by-frame audio. The managed download is
about 103 MB; GGUF weight formats are not supported. Qwen3-TTS remains the
`speech synthesize` default.

Omit voice descriptions, named speakers, clone controls, and CFG guidance.
Phonon accepts English at its native speaking rate. Spell out structured text
such as numbers, dates, and email addresses; Gradium's rewrite pipeline is not
included. See `mere.run guide --model speech-tts-phonon-31m` for the checkpoint pin, chunk
policy, API usage, and the bounded released-weight quality and speed comparison.

For full F32 checkpoint comparisons on Linux CUDA, set `MLX_ENABLE_TF32=0`
and `MLX_USE_CUDA_GRAPHS=0` before launching the CLI. The bounded
[CUDA qualification](../qualification/phonon-cuda-2026-10-23.json) records
the tested host and numerical tolerances.

## Runtime entry points

### CLI

- `Sources/MereRunCLI/Commands/SpeechSynthesizeCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechTranscribeCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechDiarizeCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechDiarizeLiveCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechListenCommand.swift`
- `Sources/MereRunCLI/Commands/SpeechProfileCommand.swift` (the `list`, `create`, and `delete` profile subcommands)

### TTS runtime

- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Loading.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Generation.swift`
- `Sources/AudioTTS/Qwen3TTS/Qwen3TTSGenerator+Support.swift`

Tokenizer internals:

- `Sources/AudioQwen3TTSModel/Qwen3TTSSpeechTokenizer.swift`
- `Sources/AudioQwen3TTSModel/Qwen3TTSSpeechTokenizer+Encoder.swift`
- `Sources/AudioQwen3TTSModel/Qwen3TTSSpeechTokenizer+Decoder.swift`

### STT runtime

- `Sources/AudioSTT/SpeechTranscriptionResolver.swift` (backend routing)
- `Sources/AudioSTT/Qwen3ASR/Qwen3ASRGenerator.swift`
- `Sources/AudioSTT/Qwen3ASR/Qwen3ASRLiveSession.swift`
- `Sources/AudioSTT/Parakeet/ParakeetGenerator.swift`
- `Sources/AudioSTT/Parakeet/ParakeetCoreMLEncoder.swift`
- `Sources/AudioSTT/Parakeet/ParakeetASRLiveSession.swift`
- `Sources/AudioSTT/Whistle/WhistleGenerator.swift` (model code in `Sources/AudioWhistleModel/`)
- `Sources/AudioSTT/NemotronTranscribe/NemotronTranscribeGenerator.swift` (model code in `Sources/AudioNemotronTranscribeModel/`)

### Diarization runtime

- `Sources/AudioSortformer/SortformerDiarizer.swift`
- `Sources/AudioSortformer/SortformerModel.swift`
- `Sources/AudioSortformer/SortformerFeatures.swift`
- `Sources/AudioSortformer/Nemotron3DiarizationModel.swift`
- `Sources/AudioSortformer/Nemotron3DiarizationStreaming.swift`

## How speech synthesis flows

At a high level:

1. The CLI validates text, temperature, and streaming options before resolving
   the model or preparing a clone reference.
2. The CLI resolves profiles and optional reference transcription. The API
   maps model and voice aliases. Both construct a shared synthesis plan.
3. AudioTTS loads the model and generates host waveform samples through one
   native path for offline and streaming requests.
4. AudioCore encodes the samples and publishes the completed WAV. CLI progress
   and receipts and API response formatting remain in their adapters.

`--temperature` must be finite and between 0 and 2. Streaming chunk intervals
must be greater than zero. Offline output remains unnormalized, undithered
PCM16. `--stream` retains float32 output, including finite samples outside
the unit range. Both replace nonfinite samples with zero.

Streaming writes to a temporary sibling file and publishes the destination
after generation ends and the WAV header is complete. Failure or cooperative
cancellation removes the temporary file and preserves an existing destination.
The receipt therefore describes the finalized file. With `--progress-json`,
streaming token progress is emitted even when you also specify `--quiet`.

See [shared speech synthesis](../internals/speech-synthesis-operation.md) for
execution, export, and model lifetime ownership.

## How transcription flows

1. The runtime decodes audio input into the expected local format.
2. The selected backend loads its model components.
3. The backend produces a transcript.
4. The CLI prints or writes the output.

## How diarization flows

1. The runtime decodes and resamples audio input to 16 kHz mono.
2. The MLX feature extractor produces NeMo-compatible mel features.
3. Sortformer predicts activity independently across four speaker channels.
4. Thresholding, minimum-duration filtering, and same-speaker gap merging
   produce time segments.
5. The CLI emits versioned JSON or standard RTTM text.

Nemotron 3 uses eight speaker channels and a cache across chunks. Its pinned
BF16 initializer is restored into FP32 model weights, matching NeMo's execution
precision.

The Spark CUDA regression checks a 46.57-second real A-B-A recording with a
returning speaker across the default chunk boundary. NVIDIA's pinned NeMo
implementation and native FP32 inference both preserve the returning speaker;
forced native BF16 inference failed that recording. This is a bounded regression
check, not a general diarization accuracy score.

Run the checkpoint regression with `MERERUN_NEMOTRON3_MODEL_DIR` set to the
installed model directory:

```bash
swift test --filter SortformerDiarizationTests/testNemotron3ReturningSpeakerSurvivesChunkBoundaryWhenConfigured
```

Fixture attribution and exact turn boundaries are in
`Tests/MereRunCoreTests/Fixtures/Nemotron3/README.md`.

## Notes for contributors

- Speech code spans both `AudioTTS` and `AudioSTT`. Do not assume it all lives
  under `MereRunCore`.
- Profile management is CLI-facing but depends on the same canonical model and
  model-store conventions as the rest of the repository.

See the [architecture map](../architecture.md) for a recommended reading order.
