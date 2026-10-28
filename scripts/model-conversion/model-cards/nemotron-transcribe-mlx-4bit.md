---
license: other
license_name: openmdw-1.1
license_link: LICENSE
base_model: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
base_model_relation: quantized
pipeline_tag: automatic-speech-recognition
library_name: mlx
tags:
  - mere-run
  - mlx
  - nemotron
  - asr
  - 4-bit
  - w4a16
---

# Nemotron 3.5 Transcribe — MLX 4-bit routed experts

This is a Sawfwair conversion of NVIDIA's
[NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
at revision `ef905575a106fdad904f3b0ce661dccad654607d` for native Swift/MLX
inference in [mere.run](https://github.com/sawfwair/mere-run). The original
model materials are by NVIDIA Corporation and use OpenMDW-1.1. The original
model card is retained as `UPSTREAM_MODEL_CARD.md`; licenses and notices are
included. See `MODIFICATIONS.md` for the conversion policy.

## Format and runtime

The 46 main-decoder routed expert matrices use MLX affine 4-bit packing,
group size 64, and BF16 scales/biases. All other 1,223 tensors retain their
original values and dtypes, including audio encoders, routers, shared experts,
Mamba layers, and retained but unexecuted MTP tensors. Weight files occupy
25.07 GB. No training or fine-tuning was performed.

This artifact requires mere.run's native routed-expert INT4 implementation
([PR #606](https://github.com/sawfwair/mere-run/pull/606)); use that source
implementation until it is merged/released. It is distinct from NVIDIA's
NVFP4 format and generic Transformers or mlx-lm checkpoints. Inference runs
locally in Swift/MLX and uses packed expert multiplication without Python.

```bash
mere.run model pull speech-asr-nemotron-35-transcribe --accept-model-license
mere.run speech transcribe ./recording.wav --backend nemotron --max-tokens 4096
```

The original BF16 variant is available separately:

```bash
mere.run model pull speech-asr-nemotron-35-transcribe-bf16 --accept-model-license
mere.run speech transcribe ./recording.wav --model speech-asr-nemotron-35-transcribe-bf16
```

The native path exposes offline transcription and translation to English.
NVIDIA lists 40 transcription languages; this conversion's accuracy evaluation
covers English only. CTC timestamps, live streaming, and speculative decoding
are outside the qualified runtime path.

## Measured comparison

Apple M5 Max, 128 GiB unified memory, macOS 27.2. Twelve recordings totaling
177.73 seconds and 522 reference words; 32 scored CLI runs plus two warm-ups.

| Metric | BF16 | Routed-expert INT4 |
| --- | ---: | ---: |
| Word errors / reference words | 2 / 522 | 4 / 522 |
| Aggregate WER | 0.383% | 0.766% |
| 46.57-second recording, median CLI time | 46.74 s | 6.45 s |
| Short recording, median CLI time | 14.94 s | 3.28 s |
| Largest observed process memory footprint | 61.74 GiB | 21.83 GiB |

INT4 added the substitutions `Kaliko` → `Calico` and `nor` → `or`.
Both formats produced the exact short sentence. WER counts each recording
once; repeated timing cases do not receive extra weight. Each timing fixture
ran three times per format. Times include model loading, audio processing,
prefill, decoding, and durable output publication in fresh CLI processes.
File/shader caches were retained, format order alternated, and TF32 was
disabled for both. Packed and BF16 implementations use different prefill
schedules; these are complete CLI comparisons, not isolated GPU throughput.

These small English cases do not establish broad accuracy equivalence,
noisy/far-field or multilingual/translation accuracy, hour-long workloads,
or minimum-memory hardware requirements. See `QUALIFICATION.md` and
`QUALIFICATION.json` for all transcripts, measurements, source hashes,
reproduction commands, and limitations. Reference speech/text uses
[Mini LibriSpeech SLR31](https://www.openslr.org/31/) under CC BY 4.0, plus
the repository's synthetic Samantha sentence.

## Integrity and provenance

The original BF16 source SHA-256 is
`03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2`.
`MERERUN_CONVERSION.json` records every converted weight shard's size and
SHA-256; `SHA256SUMS` covers published files. The reproduction scripts live in
mere.run under `scripts/model-conversion/quantize_nemotron_transcribe_experts.py`
and `scripts/model-conversion/publish_nemotron_transcribe_mlx.py`.
