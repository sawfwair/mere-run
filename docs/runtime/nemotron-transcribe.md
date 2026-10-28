# Nemotron 3.5 Transcribe

`mere.run` includes a native Swift/MLX implementation of NVIDIA's
[Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
BF16 checkpoint. It runs offline file transcription and speech translation to
English. The model lists 40 transcription languages. Language and translation
quality still require evaluation on your recordings.

## Install and run

The pinned original weights occupy approximately 67.5 GB. The managed model
lists 96 GB minimum and 128 GB recommended unified memory; active parameter
count does not reduce the memory required to hold every expert. Review the
[OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/) before installation.
The runtime requires explicit installation and never downloads these weights
automatically.

```bash
mere.run model pull speech-asr-nemotron-35-transcribe --accept-model-license
mere.run speech transcribe ./recording.wav --backend nemotron \
  --max-tokens 4096
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe \
  --task translate --max-tokens 4096
```

A local original checkpoint folder works with
`--model /absolute/path/to/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`.
It must contain `config.json`, `model.safetensors`, `tokenizer.json`, and
`tokenizer_config.json`. The runtime validates the configuration and original
BF16 tensor shapes and dtypes. The managed revision is
`ef905575a106fdad904f3b0ce661dccad654607d`.

`--language de` supplies a transcription language hint. Translation always asks
for English output. Explicit selection of Nemotron keeps this backend for both
tasks. Existing automatic selection continues to prefer Parakeet for
transcription and Qwen3-ASR for translation.

## Local 4-bit routed experts

The native runtime also accepts a local MLX affine W4A16 conversion. Only the
main decoder's 46 routed expert matrices are quantized, using groups of 64
with BF16 scales and biases. Routers, shared experts, audio encoders, Mamba
projections, and every other tensor retain their published dtype. The derived
weight files occupy approximately 25.07 GB. Runtime memory also includes
activations and recurrent/attention caches.

From a source checkout, convert an installed original checkpoint and select
the result by path:

```bash
uv run scripts/model-conversion/quantize_nemotron_transcribe_experts.py \
  --source "$HOME/Library/Application Support/MereRun/models/speech-asr-nemotron-35-transcribe" \
  --output ./nemotron-transcribe-experts-int4
mere.run speech transcribe ./recording.wav --backend nemotron \
  --model ./nemotron-transcribe-experts-int4 --max-tokens 4096
```

Conversion requires the pinned Python/MLX environment declared in the script
and verifies the original checkpoint SHA-256. It writes a new directory,
retains model licenses, and records output shard hashes in
`MERERUN_CONVERSION.json`. Inference uses Swift/MLX packed expert multiplication
without Python or full-bank dequantization. The managed model ID continues to
install BF16. NVIDIA FP8/NVFP4 checkpoints use different formats.

On twelve English recordings (522 reference words), the paired comparison
measured 0.38% WER for BF16 and 0.77% for INT4: two additional word errors.
The 46.57-second recording's median CLI time fell from 46.74 to 6.45 seconds;
the largest observed process footprint fell from 61.74 to 21.83 GiB.
These are small-set results on an M5 Max with 128 GiB, with file/shader caches
retained. See the [BF16/INT4 comparison](../qualification/nemotron-transcribe-int4-20261028/README.md)
for all cases, repeated timings, memory metrics, and reproduction commands.

## Native execution

The model uses its ASR Transformer and eight-speaker Sortformer branch,
thresholded speaker-feature fusion, and Nemotron 3.5 Lightning language decoder.
It uses checkpoint-embedded mel filters and window coefficients. All feature
extraction, expert routing, Mamba recurrence, attention, and decoding execute
locally; inference requires no Python, conversion, or external server.

Audio is decoded to mono 16 kHz without automatic gain adjustment. The
published NeMo SALM recipe splits audio into 30-second waveform rows before
feature extraction, with a tiny final row folded into the preceding row.
The native runtime follows that recipe, concatenates the encoded rows, and
generates one transcript with greedy decoding and reasoning disabled. Speaker
indices belong to each encoder row; cross-recording speaker identity is not
provided.

The runtime accepts 20 ms through one hour of audio, subject to the 65,536-token
combined input/output context. Set `--max-tokens` high enough for the expected
transcript; the command's existing default is 448. Long recordings can require
substantially more output tokens. Internal chunking does not provide live ASR.

## Qualification limits

This path supports original BF16 weights, local 4-bit routed experts, file transcription, and English
translation. FP8/NVFP4 checkpoints, live streaming, CTC word timestamps,
speaker timelines, audio question answering, and MTP/DFlash speculation are
outside its supported CLI surface. `--stream` is rejected before model loading.
The timestamp setting produces plain text when the backend has no alignments.

Independent scalar fixtures cover feature stacking, Q/K normalization,
half-head rotary attention, hybrid language decoding, cached Mamba/attention
state, sigmoid expert selection, and unbiased mixing weights. The original BF16 checkpoint passed two English CLI cases: an exact
11-word sentence and a 46.57-second human recording with two word errors out
of 129 words (1.55% WER), including the 30-second encoder boundary. See the
[checkpoint validation report](../qualification/nemotron-transcribe-native-20261028/README.md).
Multilingual and translation accuracy, one-hour workloads, and broad performance
claims still require separate qualification.
