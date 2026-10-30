# Nemotron 3.5 Transcribe

`mere.run` runs NVIDIA's
[Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
natively in Swift/MLX for offline file transcription and speech translation to
English. The managed default uses 4-bit routed experts; the original BF16
checkpoint is a separate option. NVIDIA lists 40 transcription languages.
Evaluate multilingual and translation accuracy on your own recordings.

File transcription and English translation are also available through
`POST /v1/audio/transcriptions` with either managed Nemotron model ID.
The API uses the same offline generator and returns no timestamps; it does
not add live transcription support.

## Install and run

Two managed models are available:

| Managed model | Precision | Download | Memory budget: minimum / recommended |
| --- | --- | ---: | ---: |
| `speech-asr-nemotron-35-transcribe` | 4-bit routed experts, other tensors at source precision | 25.09 GB | 32 / 48 GB |
| `speech-asr-nemotron-35-transcribe-bf16` | Original BF16 | 67.5 GB | 96 / 128 GB |

Memory budgets are planning estimates. The measured comparison ran on a
128 GiB M5 Max; it doesn't qualify a minimum-memory hardware configuration.
Neither model downloads automatically. Review the
[OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/), then install the default
and transcribe or translate a file:

```bash
mere.run model pull speech-asr-nemotron-35-transcribe --accept-model-license
mere.run speech transcribe ./recording.wav --backend nemotron --max-tokens 4096
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe \
  --task translate --max-tokens 4096
```

To use the original BF16 checkpoint, install and select it explicitly:

```bash
mere.run model pull speech-asr-nemotron-35-transcribe-bf16 --accept-model-license
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe-bf16 \
  --max-tokens 4096
```

If a complete original BF16 installation is stored under the previous default
ID, pulling either ID first preserves it as
`speech-asr-nemotron-35-transcribe-bf16` and then installs INT4 as the default.
The preservation step shares regular payload files and keeps stable cache
links, so it doesn't download the 67.5 GB checkpoint again.

### Backend selection

Automatic selection (`--backend auto`) still prefers Parakeet for transcription
and Qwen3-ASR for translation. Select Nemotron 3.5 Transcribe in either of
these ways:

- Pass `--backend nemotron`, which runs the INT4 default for either task.
- Pass a Nemotron model to `--model`: a managed ID, a repository name, or a
  local checkpoint folder.

The NVIDIA repository name resolves to BF16, and the Sawfwair repository name
resolves to INT4. A `--language` hint must be one of the 40 supported language
codes; any other hint stops the command before the model loads. See
[Transcribe audio](./speech.md#transcribe-audio) for the full routing order.

### Checkpoints and local folders

The INT4 source is
[Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit](https://huggingface.co/Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit),
pinned to revision `c20414875a76f4b9de213a4a084c9731069a29b3`. Original BF16
is pinned to NVIDIA revision `ef905575a106fdad904f3b0ce661dccad654607d`.

To run a local checkpoint folder, pass `--model /absolute/checkpoint/path`.
BF16 folders contain `config.json`, `model.safetensors`, `tokenizer.json`, and
`tokenizer_config.json`. INT4 folders also contain a versioned
`expert-quantization.json`, a safetensors index, and 29 packed shards. The
loader validates the configuration, tensor shapes and data types, and shard
membership.

## Quantization and measured tradeoff

The main decoder's 46 routed expert matrices use MLX affine W4A16, group size
64, with BF16 scales and biases. All other 1,223 tensors keep their published
values and data types, including the audio encoders, routers, shared experts,
Mamba layers, and the retained but unexecuted multi-token prediction (MTP)
tensors. Weight files occupy 25.07 GB. Inference reads the packed weights
directly, without Python or dequantizing the full expert bank. NVIDIA's FP8
and NVFP4 releases use different formats.

On twelve English recordings (522 reference words), the paired comparison
measured 0.38% word error rate (WER) for BF16 and 0.77% for INT4: two
additional word errors. For the 46.57-second recording, median CLI time fell
from 46.74 to 6.45 seconds, and the largest observed process footprint fell
from 61.74 to 21.83 GiB. These small-set results used an M5 Max with 128 GiB,
with file and shader caches retained. The
[BF16/INT4 comparison](../qualification/nemotron-transcribe-int4-20261028/README.md)
records every transcript, repeated timing, memory metric, and reproduction
command.

The pinned source converter remains available from a source checkout:

```bash
uv run scripts/model-conversion/quantize_nemotron_transcribe_experts.py \
  --source "$HOME/Library/Application Support/MereRun/models/speech-asr-nemotron-35-transcribe-bf16" \
  --output ./nemotron-transcribe-experts-int4
```

The converter verifies the original checkpoint's SHA-256, writes a new
directory, keeps the licenses, and records output shard hashes in
`MERERUN_CONVERSION.json`. Managed installs download the published conversion
directly.

## Native execution

The graph uses the ASR Transformer, the eight-speaker Sortformer branch,
thresholded speaker-feature fusion, and the hybrid Mamba/attention/MoE decoder.
Mel filters and window coefficients embedded in the checkpoint drive feature
extraction.

Audio is decoded to mono 16 kHz without automatic gain adjustment. Following
the published NeMo SALM recipe, the runtime splits audio into 30-second
waveform rows before feature extraction and folds a short final row into
the preceding row. Encoded rows share one context and one greedy text decode,
with reasoning disabled. Speaker indices belong to each encoder row; the model
doesn't provide speaker identity across rows or recordings.

## Limits

- **Input length:** 20 ms through one hour of audio, subject to the
  65,536-token combined input and output context. Set `--max-tokens` for the
  expected transcript length; the command default is 448.
- **No live ASR:** `speech transcribe --stream` stops before the model loads,
  and `speech listen` doesn't run this model. Internal chunking doesn't make
  the backend a streaming one.
- **No timestamps:** the output is plain text. CTC timestamps aren't
  supported, and `--timestamps` has no effect.
- **No speaker timelines:** use [`speech diarize`](./speech.md#identify-speakers)
  for who spoke when.
- **Unsupported features:** audio question answering, FP8 and NVFP4
  checkpoints, and MTP or DFlash speculation.

## Validation

Independent scalar fixtures cover the encoder and hybrid decoder; CPU and Metal
regressions cover packed projections, routing, prefill, and cached decoding.
Both installed variants passed an exact-sentence regression. The
[original checkpoint validation](../qualification/nemotron-transcribe-native-20261028/README.md)
and the paired INT4 report record real transcription. These small English
cases don't establish noisy or far-field accuracy, multilingual or translation
accuracy, hour-long workloads, or broad benchmark results.
