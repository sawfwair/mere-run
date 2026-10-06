---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
language:
  - en
  - de
base_model: Aleph-Alpha/Kolibri-1-BF16
base_model_relation: quantized
tags:
  - mlx
  - mere-run
  - kolibri1
  - mixture-of-experts
  - reasoning
  - conversational
  - 8-bit
---

# Kolibri-1 MLX 8-bit

This is a native **mere.run Swift/MLX** artifact for Aleph Alpha's Kolibri-1.
It contains 83.65 GB of logical tensor storage and is the candidate for a 128 GB
Apple Silicon machine. Full-checkpoint Apple memory fit and throughput remain
unverified. It requires a development build with the native Kolibri runtime
(source snapshot `8bfa3f5d6d9f23097cd1446935acc4589ce6fa72`).

## Quantization and source

The source is `Aleph-Alpha/Kolibri-1-BF16` at
`7a8f290e7858825c3cf5e4c447ba68345de9f1d3`. Routed experts, attention, and
shared-expert projections use MLX affine 8-bit/group-64 weights. Embeddings,
learned norms, routers, and the vocabulary head retain source precision.
Router correction biases are converted losslessly to FP32; routing and logits
compute in FP32. No retraining or instruction fine-tuning occurred.

Expert banks are stacked in numeric order, and `config.json` records every
projection's policy. Use the native mere.run Kolibri loader. This custom layout
is not a generic `mlx-lm` or Transformers checkpoint and is not the upstream
vLLM tensor layout.

## Measured diagnostic

Paired native BF16 and quantized scoring ran on a Runpod NVIDIA B200 with the
same token sequences, score boundaries, and teacher-forced continuations.
The suite contains 16 cases, including 3 calibration cases and 13 heldout cases.
The heldout split has only 198 target tokens; it is a small diagnostic rather
than a general capability benchmark. Q8 was measured as a larger-budget control
separately from the selection between baseline and fitted mixed 2-bit.

| Heldout measurement | Result |
| --- | ---: |
| Mean full-vocabulary KL | 0.001694 |
| Top-token agreement with BF16 | 98.99% |
| Target perplexity increase | 0.74% |
| Peak CUDA MLX allocation | 86.12 GB |

Overall, English, and German diagnostic gates passed independently. Peak MLX
allocation excludes other process/system memory and does not prove Apple
memory fit. All diagnostic cases disable thinking; reasoning quality and
large-context quality remain unmeasured here.

`KOLIBRI_NATIVE_DIAGNOSTIC.json` contains source/binary hashes, exact sequences,
conversion manifests, fitting statistics, and per-case results for all variants.
`KOLIBRI_QUALIFICATION.json` binds this artifact to its scoring receipt.
The conversion manifest retains `quality_qualified: false`; public availability
does not establish general accuracy or register a managed mere.run download.

## Run

Download this repository, then pass the local directory to a mere.run build
containing the native Kolibri runtime:

```bash
hf download Sawfwair/Kolibri-1-MLX-8bit --local-dir ./Kolibri-1-MLX-8bit

mere.run text chat --model ./Kolibri-1-MLX-8bit \
  --prompt "Explain why the sky is blue." --context-size 8192 --max-tokens 256

mere.run api serve --engine text-chat-kolibri \
  --model ./Kolibri-1-MLX-8bit --context-size 8192
```

The runtime uses the source tokenizer/chat template, supports tool messages and
opt-in token logprobs, and defaults native chat to an 8,192-token budget. The
262,144-token limit in the pinned configuration is not a memory-fit claim.
LoRA, constrained JSON, KV quantization, prefix reuse, and continuous batching
are not implemented for this family.

## License and provenance

Apache 2.0; see `LICENSE`, `MODIFICATIONS.md`, and `UPSTREAM_MODEL_CARD.md`.
Aleph Alpha's upstream model card describes the original model's intended use,
training, and limitations. `KOLIBRI_CONVERSION.json` records pinned source and
output-shard SHA-256 values; `SHA256SUMS` covers the published bundle.
