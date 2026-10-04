# Native Kolibri mixed 2-bit checks — October 4, 2026

The native Swift/MLX Kolibri runtime loaded and scored the full BF16 checkpoint
and four quantized variants on a Runpod NVIDIA B200. Calibration selected
`mixed2` by its lower mean full-vocabulary KL. The fitted candidate's calibration
KL was 0.012610, compared with 0.011358 for the baseline, so the fit was not
promoted.
Both 2-bit variants use **24.66 GB** of logical tensor storage.

The [machine-readable receipt](./receipts/kolibri-native-2026-10-04.json)
contains source pins, conversion manifests and shard checksums, exact token
sequences, scoring receipts, binary and source hashes, fitting statistics, and
per-case comparisons. These measurements cover a small diagnostic suite;
they do not establish general model accuracy or Apple Silicon memory fit.

## Weight policies and results

Mixed 2-bit uses 2-bit affine routed expert weights with group size 128.
Attention and shared experts use 8-bit/group-64 weights. Embeddings, learned
norms, routers, and the output head retain source precision; router correction
biases are converted losslessly from BF16 to FP32. Routing and output logits
compute in FP32. The 2/3-bit control changes only routed down projections to
3-bit/group-64; the Q8 control uses 8-bit/group-64 for all quantized projections.

| Variant | Logical weights | Peak MLX allocation | Calibration KL | Heldout KL | Top-token agreement | Perplexity change | Overall and language gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline mixed 2-bit | 24.66 GB | 26.57 GB | 0.011358 | 0.026801 | 97.47% | +3.46% | Pass |
| Calibrated mixed 2-bit | 24.66 GB | 26.55 GB | 0.012610 | 0.034015 | 98.48% | +2.48% | Pass |
| 2-bit gate/up, 3-bit down | 28.60 GB | 30.52 GB | 0.012606 | 0.019127 | 96.46% | +1.46% | Pass |
| 8-bit control | 83.65 GB | 86.12 GB | 0.001036 | 0.001694 | 98.99% | +0.74% | Pass |

Calibration improved target perplexity and top-token agreement, while increasing
full-distribution KL on both splits. Groupwise weight-error improvements did not
translate into a consistent distribution-level improvement.

GB uses decimal bytes. MLX allocation peaks are measured on CUDA and exclude
other process and operating-system memory. The native BF16 reference used
156.21 GB of weights and peaked at 157.80 GB of MLX allocations. Candidate
results use a release harness; the BF16 reference uses a debug harness. A tiny packed
fixture produced identical logits in those two builds. Timings in the receipt
are per-case inference/export timings and omit checkpoint verification/loading;
they are not end-to-end CLI latency comparisons.

For the selected 2-bit candidate:

| Language | Heldout target tokens | Mean KL | Top-token agreement | Perplexity change | Worst-case perplexity ratio | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| de | 86 | 0.034296 | 96.51% | +3.74% | 1.2249 | Pass |
| en | 112 | 0.021046 | 98.21% | +3.25% | 1.1673 | Pass |

Top-token agreement compares candidate and reference distributions; it is not
an answer-accuracy score. Perplexity ratios use the fixed teacher-forced target
continuations. Project diagnostic limits are mean KL ≤ 0.15, worst-case mean
KL ≤ 0.30, perplexity ratio ≤ 1.10, worst-case perplexity ratio ≤ 1.25, and
agreement ≥ 80%. Overall results and each language must pass separately.

## Calibration and heldout separation

The suite has 16 cases, 3,832 total tokens, and 251 scored target tokens.
Three calibration cases contribute 53 scored tokens; 13 heldout cases contribute
198. Cases cover English/German knowledge, arithmetic, logic, Python,
translation, two prompts crossing the 513-token sliding window, and a tool call.
The window prompts contain roughly 1,600 tokens. All cases disable thinking and
use manually specified continuations followed by the checkpoint EOS token.

Only the three calibration cases contribute expert-input second moments.
The fit uses diagonal input moments, a small variance floor, eight Lloyd
iterations, BF16 affine levels, and groupwise weighted-error acceptance. Signed
affine scales and original packed codes are preserved unless a group's weighted
error improves. The heldout split contributes neither statistics nor candidate
selection. Q8 and 2/3-bit are separate controls with larger storage budgets.

The conversion manifests retain `quality_qualified: false`: passing this small
suite does not justify a broad quality claim or a managed public download.

## Native reference checks and local validation

The checkpoint is
[`Aleph-Alpha/Kolibri-1-BF16`](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
at `7a8f290e7858825c3cf5e4c447ba68345de9f1d3`.
Architecture semantics follow Aleph Alpha's Apache-2.0 inference implementation
at `049a6a7bd2405b27d6d280d256bd3d585191c7ae`.

An independent PyTorch BF16 operator implementation read the original tensor
layout and scored 19 calibration-English positions. Against native BF16, mean
full-vocabulary KL was 0.001975, top-token agreement was 100%, and target-token
logprob RMSE was 0.09080. Maximum target-token logprob difference was 0.35784;
maximum raw-logit difference was 7.87034. This is approximate numerical
agreement on one case, not bitwise parity with official vLLM fused kernels.
Independent FP32 synthetic fixtures also check attention, routing, shared
experts, norms, and chunked-cache behavior on CPU and Apple GPU.

The CUDA executable temporarily isolates the production model, loader, and
scorer from unrelated CLI targets. Swift 6.2, pinned MLX Swift, MLX core 0.32.1,
and conversion MLX 0.32.2 are recorded in the receipt. All tested production
math/loader/scorer source hashes match the final source. A later change to
`KolibriResources.swift` only adds local-directory identification and preserves
its checkpoint pin; that helper is not used by the scoring path.

The full repository gate (`scripts/check.sh`) passes: 5,304 XCTest cases
(439 skips, zero failures), plus 191 Swift Testing cases.
Three signed-affine refit and four comparator regression tests also pass.
Native chat with the real pinned
tokenizer, a 41 MB synthetic quantized checkpoint, tool history, and captured
token logprobs passed locally. The ordinary CLI admission check correctly
blocked generation while this Mac had less than its 16 GB memory-headroom floor.

Full converted checkpoints have **not** been run on 36 GB or 128 GB Apple
machines. Mixed 2-bit is the candidate for 36 GB; Q8 is the candidate for 128 GB.
CUDA peaks and file sizes support that sizing direction, but actual Apple
footprint, prefill/decode throughput, and larger context budgets still need
real-checkpoint runs. No large checkpoint was downloaded to this Mac.

## Reproduction

Follow the [native runtime and conversion workflow](../runtime/kolibri.md).
Run the native BF16 scorer with `--calibration-output`, fit the Q2 candidate,
then score each quantized checkpoint with the same `suite.json` and chunk size
32. Compare the two Q2 results together to select by calibration; compare Q8
and 2/3-bit separately as controls. The receipt contains the exact sequences
and rendered text, so boundaries and targets can be reproduced without
regenerating the suite. Keep checkpoint manifests and raw logits with their
recorded checksums.
