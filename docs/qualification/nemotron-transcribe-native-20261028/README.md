# Nemotron Transcribe native checkpoint validation, 2026-10-28

The original NVIDIA BF16 checkpoint passed end-to-end transcription through
`mere.run speech transcribe --backend nemotron` on an Apple M5 Max with
128 GiB unified memory and macOS 27.2. The release CLI loaded the original
weights directly, processed local audio, decoded text, and published durable
runs in the `succeeded` state. No external inference server or Python model
runtime was used. Python only orchestrated the CLI and scored its output.

## Checkpoint integrity

- Repository: `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`.
- Revision: `ef905575a106fdad904f3b0ce661dccad654607d`.
- `model.safetensors`: 67,496,013,734 bytes.
- SHA-256: `03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2`.
- The installed file matched the publisher's SHA-256. All 1,269 inference
  tensor shapes and dtypes matched the native runtime inventory.

`report.json` records the binary checksum, source checksums, exact references,
transcripts, word differences, execution times, and process memory measurements.

## Results

| Recording | Duration | Word errors | WER | CLI wall time |
| --- | ---: | ---: | ---: | ---: |
| Known sentence, macOS Samantha voice | 3.52 s | 0 / 11 | 0% | 17.43 s |
| Human speech, two speakers and three turns | 46.57 s | 2 / 129 | 1.55% | 49.39 s |

The short output was exactly:

> The parcel arrived on time, but the glass inside was broken.

The human recording crosses the 30-second encoder boundary and emitted one
complete transcript from two encoded audio chunks. Its two substitutions were
`Kaliko` → `Calico` and `sceptre` → `scepter`. The latter is a spelling variant;
it remains an error in the reported strict word comparison.

Scoring lowercases text, normalizes curly apostrophes, and ignores punctuation.
It preserves apostrophes inside words and does not apply spelling equivalences.
Each case ran in a fresh CLI process. Times include model loading, audio
processing, prefill, decoding, and durable output publication. File/shader
caches were not cleared; these are not isolated throughput measurements.
The observed peak process memory footprints were 61.33 and 61.95 GiB;
`/usr/bin/time` maximum resident sizes were 52.52 and 49.03 GiB, respectively.

## Regression found and fixed

The first checkpoint run reached decoding and exposed an incompatible float32
causal mask passed to BF16 scaled-dot-product attention. The runtime now casts
that mask to the activation dtype. A BF16 full-prefill/cached-decode test covers
the failure, and an opt-in installed-checkpoint test requires the exact known
sentence. All 19 focused tests passed on Metal with the installed checkpoint.

## Reproduce

The historical default ID in `report.json` selected BF16 when measured.
Reproduction now uses the explicit BF16 model ID.

```bash
swift build --build-system native -c release --product mere.run
.build/release/mere.run model pull speech-asr-nemotron-35-transcribe-bf16 --accept-model-license
.build/release/mere.run speech transcribe Tests/MereRunCoreTests/Fixtures/D1/released/speech.wav \
  --backend nemotron --model speech-asr-nemotron-35-transcribe-bf16 --max-tokens 96 --run-dir ./asr-short-validation
.build/release/mere.run speech transcribe Tests/MereRunCoreTests/Fixtures/Nemotron3/returning-speaker.wav \
  --backend nemotron --model speech-asr-nemotron-35-transcribe-bf16 --max-tokens 512 --run-dir ./asr-human-validation
```

Use new run directories for each invocation. The native SwiftPM build system
was used because the default Xcode build system rejected an existing missing
vendored llama dSYM path. No vendor files were changed.

Run the installed-checkpoint and numerical regressions with:

```bash
MLX_ENABLE_TF32=0 MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_NEMOTRON_TRANSCRIBE_PATH="$HOME/Library/Application Support/MereRun/models/speech-asr-nemotron-35-transcribe-bf16" \
  swift test --build-system native --filter NemotronTranscribe
```

## Inputs and limits

The human recording is the repository's returning-speaker fixture, derived
from [Mini LibriSpeech SLR31](https://www.openslr.org/31/) under CC BY 4.0.
The reference transcript concatenates `5694-64038-0022`, `1272-135031-0024`,
and `5694-64038-0017`. The publisher archive's MD5 was verified as
`6d7ab67ac6a1d2c993d050e16d61080d`, and the WAV SHA-256 was verified as
`042749dfa02b8f159228426d8aba1a7c3b6f8ee293a2c5349a770cd75b672189`.
The synthetic sentence reuses the D1 released-checkpoint fixture; neither
reference text was supplied to the model in its prompt.

These two English cases establish the original-checkpoint file-transcription
path. They do not qualify multilingual or translation accuracy, one-hour
recordings, general memory requirements, or broad ASR benchmark performance.
CTC timestamps, live streaming, and speculative decoding remain outside this
supported path.
