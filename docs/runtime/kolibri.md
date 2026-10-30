# Native Kolibri-1

Aleph Alpha's Kolibri-1 is a sparse mixture-of-experts chat model. `mere.run`
runs converted Kolibri
checkpoints natively in Swift/MLX for `text chat` and `api serve`. Kolibri isn't
a managed download: you download a converted checkpoint yourself and pass its
directory to `--model`.

## Architecture and checkpoint formats

The model has 50 transformer layers, 384 routed experts per layer, six selected
experts per token, and one shared expert. Four sliding attention layers precede
each full attention layer. Sliding layers apply RoPE; full layers use no
positional encoding.

The converted format supports a BF16 reference, an 8-bit variant, and explicit
mixed quantization policies. Mixed 2-bit keeps routed experts at 2-bit with
128-value affine groups and attention and shared experts at 8-bit. Embeddings,
norms, routers, and output-head weights keep their original precision. Routing
and the output head compute in FP32.

## Install a converted checkpoint

The selected [mixed 2-bit artifact](https://huggingface.co/Sawfwair/Kolibri-1-MLX-Mixed-2bit)
and the [Q8 control](https://huggingface.co/Sawfwair/Kolibri-1-MLX-8bit) are
public. Both require mere.run v0.62.0 or later. To reproduce the measured
weights, download the immutable release revisions:

```bash
hf download Sawfwair/Kolibri-1-MLX-Mixed-2bit \
  --revision 73be205486dd71273a30ab9ce3441e548f0a33f4 \
  --local-dir ./Kolibri-1-MLX-Mixed-2bit

hf download Sawfwair/Kolibri-1-MLX-8bit \
  --revision 792bfb73b9ab915724ccfd58916587330a642c16 \
  --local-dir ./Kolibri-1-MLX-8bit
```

The [publication receipt](../benchmarks/receipts/kolibri-publication-2026-10-04.json)
records all 64 bundle file hashes and sizes for each release.

The mixed 2-bit tensors occupy 24.66 GB, which makes it a candidate for a 36 GB
Mac. The 83.65 GB Q8 control is a candidate for 128 GB. Apple Silicon memory fit
and throughput for the full checkpoints are still unvalidated.

## Chat and serve

Pass the converted checkpoint directory itself. The upstream FP8 and BF16
repositories use a different tensor layout, so convert them first.

```bash
mere.run text chat \
  --model ./Kolibri-1-MLX-Mixed-2bit \
  --prompt "Erkläre, wie ein Regenbogen entsteht." \
  --context-size 8192 --max-tokens 256

mere.run api serve \
  --engine text-chat-kolibri --model ./Kolibri-1-MLX-Mixed-2bit \
  --context-size 8192
```

The runtime uses the checkpoint's chat template and tokenizer. It supports text,
reasoning, tool messages, seeded sampling, stop sequences, and opt-in token log
probabilities.

`text chat` defaults to an 8,192-token context. `api serve` uses the server's
shared default instead, so set `--context-size` explicitly when you serve
Kolibri. The runtime rejects oversized prompts instead of truncating them. The
architecture's 262,144-token limit isn't a memory-fit claim.

## Limits

LoRA, constrained JSON, KV quantization, prefix reuse, and continuous batching
aren't implemented for this family.

## Convert and measure quality

The remote conversion tools pin `Aleph-Alpha/Kolibri-1-BF16` at
`7a8f290e7858825c3cf5e4c447ba68345de9f1d3`, verify source hashes, stack expert
banks in numeric order, and record every output shard's checksum. Run them on a
remote worker with room for the 156 GB source and the BF16 reference.

On October 6, 2026, Hugging Face returned HTTP 404 (`Invalid rev id`) for that
recorded upstream revision. The public converted release revisions still
resolve. To reproduce the conversion, you need a preserved original source
snapshot with the recorded checksums; a replacement upstream revision needs
separate validation.

The following commands convert the checkpoint, prepare the diagnostic suite,
score the reference and a candidate, and compare them:

```bash
python scripts/model-conversion/convert_kolibri_mlx.py \
  --source /workspace/kolibri/source --output /workspace/kolibri/artifacts \
  --profiles reference mixed2 mixed2-down3 q8

python scripts/reference-parity/prepare_kolibri_suite.py \
  --source /workspace/kolibri/source --output /workspace/kolibri/suite

mere.run model benchmark kolibri-logprobs \
  --model-root /workspace/kolibri/artifacts/Kolibri-1-MLX-reference \
  --suite /workspace/kolibri/suite/suite.json \
  --output /workspace/kolibri/results/reference \
  --calibration-output /workspace/kolibri/results/input-moments.safetensors

mere.run model benchmark kolibri-logprobs \
  --model-root /workspace/kolibri/artifacts/Kolibri-1-MLX-mixed2 \
  --suite /workspace/kolibri/suite/suite.json \
  --output /workspace/kolibri/results/mixed2

python scripts/reference-parity/compare_kolibri_logprobs.py \
  --reference /workspace/kolibri/results/reference \
  --candidates /workspace/kolibri/results/mixed2 \
  --output /workspace/kolibri/results/comparison.json
```

For an activation-weighted Q2 candidate, use the reference's calibration
statistics and keep the separate scoring receipt:

```bash
python scripts/model-conversion/refit_kolibri_mixed2.py \
  --reference /workspace/kolibri/artifacts/Kolibri-1-MLX-reference \
  --baseline /workspace/kolibri/artifacts/Kolibri-1-MLX-mixed2 \
  --moments /workspace/kolibri/results/input-moments.safetensors \
  --reference-results /workspace/kolibri/results/reference \
  --output /workspace/kolibri/artifacts/Kolibri-1-MLX-mixed2-calibrated
```

Every comparison uses identical token IDs, prompt boundaries, and teacher-forced
continuations. It reports full-vocabulary KL divergence, next-token agreement,
changes in target-token log probabilities, and perplexity ratios by case and
language.

Only cases labeled `calibration` contribute input second moments. Candidate
selection uses the calibration split; held-out cases report quality without
changing the fit. English and German must each pass their diagnostic gates.
Select between the baseline and fitted Q2 candidates, and report the Q8 and
2/3-bit variants as controls. Their larger weight budgets don't replace the Q2
selection. The included English and German suite is a small diagnostic, not a
standardized benchmark or a general capability guarantee.

## Validation

The independent PyTorch synthetic fixture checks attention, routing, shared
experts, norms, and chunked cache semantics on CPU and Apple GPU. The original
BF16 PyTorch operator reference adds a checkpoint cross-check; it doesn't
certify bitwise parity with official vLLM fused kernels.

The [October 4 full-checkpoint diagnostic](../benchmarks/kolibri-native-qualification-2026-10-04.md)
records the Runpod results and selects baseline mixed 2-bit by calibration KL.

For other text models, see the [text runtime](./text.md).
