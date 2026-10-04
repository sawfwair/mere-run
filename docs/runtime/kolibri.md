# Native Kolibri-1

Kolibri-1 runs through native Swift/MLX computation. It uses 50 transformer
layers, 384 routed experts per layer, six selected experts per token, and one
shared expert. Four sliding attention layers precede each full attention
layer. Sliding layers apply RoPE; full layers use no positional encoding.

The checkpoint format supports a BF16 reference, an 8-bit variant, and explicit
mixed quantization policies. Mixed 2-bit keeps routed experts at 2-bit with
128-value affine groups, attention and shared experts at 8-bit, and embeddings,
norms, routers, and output-head weights at their original precision. Routing
and the output head compute in FP32.

## Run a converted checkpoint

```bash
swift run mere.run text chat \
  --model /path/to/Kolibri-1-MLX-mixed2 \
  --prompt "Erkläre, wie ein Regenbogen entsteht." \
  --context-size 8192 --max-tokens 256

swift run mere.run api serve \
  --engine text-chat-kolibri --model /path/to/Kolibri-1-MLX-mixed2 \
  --context-size 8192
```

Use the converted checkpoint directory itself. The upstream FP8 and BF16
repositories use a different tensor layout and must be converted first.
Managed downloads will require a published, pinned artifact; this runtime
addition alone does not register an unqualified model download.

The runtime uses the checkpoint's chat template and tokenizer, supports text,
reasoning, tool messages, seeded sampling, stop sequences, and opt-in token
logprobs. LoRA, constrained JSON, KV quantization, prefix reuse, and continuous
batching are not implemented for this family. Native chat defaults to an
8,192-token context budget; set it explicitly when serving through the API,
which otherwise uses the server's shared default. The runtime rejects oversized
prompts; it does not silently truncate them.
The architecture's 262,144-token limit is not a memory-fit claim.

## Conversion and quality measurements

The remote conversion tools pin `Aleph-Alpha/Kolibri-1-BF16` at
`7a8f290e7858825c3cf5e4c447ba68345de9f1d3`, verify source hashes, stack expert
banks in numeric order, and record every output shard's checksum. Keep the
156 GB source and BF16 reference on a sufficiently large remote worker.

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
statistics and retain the separate scoring receipt:

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
language. Only cases explicitly labeled `calibration` contribute input second
moments. Candidate selection uses the calibration split; heldout cases report
quality without changing the fit. The included small English/German suite is a
diagnostic, not a standardized benchmark or a general capability guarantee.
English and German must also pass their diagnostic gates separately. Select
between the baseline and fitted Q2 candidates; report the Q8 and 2/3-bit
variants as controls rather than allowing their larger weight budgets to
replace that Q2 selection.

The independent PyTorch synthetic fixture checks attention, routing, shared
experts, norms, and chunked cache semantics on CPU and Apple GPU. The original
BF16 PyTorch operator reference provides an additional checkpoint cross-check;
it does not certify bitwise parity with official vLLM fused kernels.

The [October 4 full-checkpoint diagnostic](../benchmarks/kolibri-native-qualification-2026-10-04.md)
records the Runpod results and selects baseline mixed 2-bit by calibration KL.
Its 24.66 GB tensor storage is a candidate for a 36 GB machine; the 83.65 GB Q8
control is a candidate for 128 GB. Actual Apple full-checkpoint memory fit and
throughput still require validation.
