# Nemotron 3.5 Transcribe

`mere.run` runs NVIDIA's
[Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
natively in Swift/MLX for offline file transcription and speech translation to
English. The managed default uses 4-bit routed experts; the original BF16
checkpoint is a separate option. NVIDIA lists 40 transcription languages;
multilingual and translation accuracy still require evaluation on your recordings.

## Install and run

| Managed model | Precision | Download | Memory budget: minimum / recommended |
| --- | --- | ---: | ---: |
| `speech-asr-nemotron-35-transcribe` | 4-bit routed experts, other tensors at source precision | 25.09 GB | 32 / 48 GB |
| `speech-asr-nemotron-35-transcribe-bf16` | Original BF16 | 67.5 GB | 96 / 128 GB |

Memory budgets are planning estimates. The measured comparison ran on a
128 GiB M5 Max; it does not qualify a minimum-memory hardware configuration.
Both variants require explicit installation. Review the
[OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/) before downloading.

```bash
mere.run model pull speech-asr-nemotron-35-transcribe --accept-model-license
mere.run speech transcribe ./recording.wav --backend nemotron --max-tokens 4096
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe \
  --task translate --max-tokens 4096
```

Select original BF16 explicitly:

```bash
mere.run model pull speech-asr-nemotron-35-transcribe-bf16 --accept-model-license
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe-bf16 \
  --max-tokens 4096
```

Pulling either ID preserves a complete original BF16 installation stored under
the previous default ID as `speech-asr-nemotron-35-transcribe-bf16` before
replacing the default with INT4. The preservation step shares regular payloads
and retains stable cache links, without redownloading the 67.5 GB checkpoint.

The INT4 source is
[Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit](https://huggingface.co/Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit).
The managed catalog pins an immutable Hub revision. Original BF16 is pinned to
NVIDIA revision `ef905575a106fdad904f3b0ce661dccad654607d`. INT4 is pinned to
`c20414875a76f4b9de213a4a084c9731069a29b3`. Selecting the NVIDIA
repository name resolves to BF16; selecting the Sawfwair repository resolves
to INT4. Existing automatic backend selection continues to prefer Parakeet for
transcription and Qwen3-ASR for translation. `--backend nemotron` selects
Nemotron and its INT4 default for either task.

A local checkpoint folder works with `--model /absolute/checkpoint/path`.
BF16 folders contain `config.json`, `model.safetensors`, `tokenizer.json`, and
`tokenizer_config.json`. INT4 folders additionally use versioned
`expert-quantization.json`, a safetensors index, and 29 packed shards. The
loader validates configuration, tensor shapes/dtypes, and shard membership.

## Quantization and measured tradeoff

The main decoder's 46 routed expert matrices use MLX affine W4A16, group size
64, with BF16 scales and biases. All other 1,223 tensors retain their published
values and dtypes, including audio encoders, routers, shared experts, Mamba
layers, and retained but unexecuted MTP tensors. Weight files occupy 25.07 GB.
Inference consumes the packed weights directly, without Python or full-bank
dequantization. NVIDIA's FP8/NVFP4 releases use different formats.

On twelve English recordings (522 reference words), the paired comparison
measured 0.38% WER for BF16 and 0.77% for INT4: two additional word errors.
The 46.57-second recording's median CLI time fell from 46.74 to 6.45 seconds;
the largest observed process footprint fell from 61.74 to 21.83 GiB.
These small-set results used an M5 Max with 128 GiB, with file/shader caches
retained. See the [BF16/INT4 comparison](../qualification/nemotron-transcribe-int4-20261028/README.md)
for every transcript, repeated timing, memory metric, and reproduction command.

The pinned source converter remains available from a source checkout:

```bash
uv run scripts/model-conversion/quantize_nemotron_transcribe_experts.py \
  --source "$HOME/Library/Application Support/MereRun/models/speech-asr-nemotron-35-transcribe-bf16" \
  --output ./nemotron-transcribe-experts-int4
```

It verifies the original checkpoint SHA-256, writes a new directory, retains
licenses, and records output shard hashes in `MERERUN_CONVERSION.json`.
Managed installs download the published conversion directly.

## Native execution and limits

The graph uses the ASR Transformer, eight-speaker Sortformer branch,
thresholded speaker-feature fusion, and hybrid Mamba/attention/MoE decoder.
Checkpoint-embedded mel filters and window coefficients drive feature extraction.

Audio is decoded to mono 16 kHz without automatic gain adjustment. The
published NeMo SALM recipe splits audio into 30-second waveform rows before
feature extraction, folding a tiny final row into the preceding row. Encoded
rows share one context and greedy text decode, with reasoning disabled.
Speaker indices belong to each encoder row; cross-recording speaker identity
is not provided.

Audio can be 20 ms through one hour, subject to the 65,536-token combined
input/output context. Set `--max-tokens` for the expected transcript; the
command default is 448. Internal chunking does not provide live ASR.
`--stream` is rejected before loading. CTC timestamps, speaker timelines,
audio question answering, FP8/NVFP4 checkpoints, and MTP/DFlash speculation
are outside this supported surface. The timestamp setting produces plain text
when the backend has no alignments.

Independent scalar fixtures cover the encoder and hybrid decoder; CPU/Metal
regressions cover packed projections, routing, prefill, and cached decoding.
Both installed variants passed an exact-sentence regression. The
[original checkpoint validation](../qualification/nemotron-transcribe-native-20261028/README.md)
and paired INT4 report record real transcription. No noisy/far-field,
multilingual/translation accuracy, hour-long workload, or broad benchmark
claim is established by these small English cases.
