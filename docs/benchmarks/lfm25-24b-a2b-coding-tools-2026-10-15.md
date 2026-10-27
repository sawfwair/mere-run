# LFM2.5 24B-A2B: coding and tool quantization

The 23.94 GB artifact remains the preferred preserved control. Within this
smaller-bundle experiment, the previous 18.95 GB artifact was retained as the
baseline. The broader candidate improves source-distribution fidelity, but
native coding correctness remains unscored, so its frozen nonregression check
is unverified. Preserve both smaller artifacts as separate experiments and keep
the larger control unchanged. All candidates retain `quality_qualified: false`:
the absolute source-quality gates remain unmet.

The new broader bundle contains **18,945,361,325 bytes** (18.95 decimal GB),
with the same 18,926,985,216 logical weight bytes as the previous small quant.
It runs through the native Swift/MLX path on the 36 GiB M3 Max at the tested
output and context limits. Bundle size does not describe runtime memory demand.

## Artifacts and method

Source: `LiquidAI/LFM2.5-24B-A2B` at
`1769516ac588c4c558089c94f27d96b0dc67b183`. All ten BF16 source shards and
metadata were hash-verified. The [previous 18.95 GB quant](./lfm25-24b-a2b-18gb-2026-10-15.md)
and [23.94 GB control](./lfm25-24b-a2b-integration-2026-10-13.md) remain unchanged.
The remote larger-control reproduction matches all original shards and receipts.

Both refinements retain 70 expert projection banks at 5 bits and 44 at 6 bits,
group size 64, FP16 scales/biases, and 314 unchanged non-expert tensors. Full-column
activation covariance guides joint fitting of affine coefficients and integer
codes, with descending activation order, one percent damping, and cross-block
error feedback. Each output row retains its parent values unless measured
reconstruction loss improves. The broader candidate adds five percent isotropic
covariance regularization and coding/tool covariance weight three.

The focused candidate uses 96 calibration cases (32 coding, 32 tools, 32 prose).
The broader candidate uses 324 cases, combining the earlier 260 training cases
with the 64 canonical coding/tool training cases. No evaluation prompts feed
coefficient fitting. Frozen fixture audits confirm no prompt overlap between
calibration, the distribution holdout, and either HumanEval subset.
The focused pass visits every expert in all 38 sparse layers, with minimum
routed count six; rare-expert coverage remains limited.

The original selection objective is 40% coding KL, 40% tool KL, and 20% prose KL.
It selected the focused candidate before holdout scoring, at 20:47:04 UTC.
Weighted calibration KL was 0.007249 versus 0.013546 for the parent,
0.008616 for the broader candidate, and 0.009493 for the larger control.
The focused candidate subsequently regressed on HumanEval. The prebuilt broader
candidate was explored separately. Its original advancement screen failed
(tied coding count and lower exact matches on underspecified tool strings).
That failed decision remains preserved. A separate plan froze the broader model
and fresh fixtures for final confirmation; no model parameters were refitted.

The broader conversion receipt SHA-256 is
`90ff17a6f0749565ef7f9967916fa00a68f705d989350ba5f32d46388dfa2ce6`.
Its 115 shards, 656 tensors, source config, generation config, metadata closure,
and refinement receipts pass local verification after inference. The original
refinement receipt remains as ancestry, with an explicit pointer to the new
joint-refinement receipt. The bundle is self-contained.

## Distribution fidelity

The holdout has 56 cases and 3,611 scored tokens: 12 new coding families,
12 tool cases, and 32 fresh WikiText test windows. The canonical 152-case
calibration/holdout suite SHA-256 is
`0c50530b3d74b1d44ba2759d1b944dbbd763eac89cf133659f34e92147e6255e`.
Native prompt/token parity passes, including complete tool schemas.

References use the original BF16 weight values promoted to FP32, identical
input tokens, disabled TF32, and all 128,000 vocabulary logits. Quant references
use their stored affine values. KL measures source-distribution distortion;
it is distinct from coding correctness or successful tool use.

| Artifact | Coding KL | Tool KL | Prose KL | Overall KL | Top-token agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous 18.95 GB | 0.011196 | 0.006948 | 0.039418 | 0.029700 | 93.71% |
| Focused 18.95 GB | 0.005751 | 0.004662 | 0.045858 | 0.032634 | 93.38% |
| Broader 18.95 GB | 0.005665 | 0.003410 | 0.033601 | 0.024213 | 94.21% |
| Larger 23.94 GB control | 0.004875 | 0.003208 | 0.027379 | 0.019834 | 94.60% |

Against the parent, the broader candidate lowers coding KL 49.4%, tool KL 50.9%,
prose KL 14.8%, and overall KL 18.5%. Its weighted objective improves 31.6%.
The focused candidate improves targeted fidelity but worsens overall and prose
fidelity. The larger control remains closer to the source overall.

Unchanged absolute gates require mean KL at most 0.01, case KL at most 0.03,
top-token agreement at least 98%, aggregate target perplexity ratio at most
1.01, and case ratio at most 1.05. The broader artifact has mean KL
0.024213, worst case KL 0.054565,
top-token agreement 94.21%, aggregate perplexity ratio
0.995228, and worst case ratio
1.162412. Its aggregate source-quality gate fails.
The parent, focused candidate, and larger control also fail their overall gates.

## Functional coding and tools

Coding tests use two disjoint deterministic 32-problem subsets of
[OpenAI HumanEval](https://github.com/openai/human-eval), pinned to
`6d43fb980f9fee3c892a914eda09951f772ad10d`, with seeds 2026101547 and 2026101559.
Prompts request complete functions through chat. Greedy decoding permits 4,096
output tokens, with repetition penalty 1.05 over 64 tokens and 8-bit affine KV.
These are chat-adapted diagnostic subsets, not the full 164-problem benchmark
or a published comparable pass@1 score. Reference answers pass the evaluator.
Fresh means unseen by this quantization experiment; upstream pretraining
contamination was not assessed.
Fresh sampling excludes four random-test problems and two whose canonical
answers fail this restricted harness; those exclusions remain recorded.

| CUDA artifact | Initial coding | Fresh coding | Combined coding | Fresh exact tools |
| --- | ---: | ---: | ---: | ---: |
| Source BF16 values / FP32 | 30/32 | 27/32 | 57/64 | 12/12 |
| Previous 18.95 GB | 29/32 | 27/32 | 56/64 | 12/12 |
| Focused 18.95 GB | 27/32 | Not run | Not run | Not run |
| Broader 18.95 GB | 29/32 | 27/32 | 56/64 | 12/12 |
| Larger 23.94 GB control | 29/32 | 27/32 | 56/64 | 12/12 |

The fresh 44-case fixture adds 12 explicit tool requests to the 32 coding
problems. Schemas constrain language and extension spellings, while prompts
specify exact paths, line conventions, ordering, and coverage flags. All four
tools are offered per request. The original 24 authored results remain in the
receipt: they include underspecified free-string spellings and an ambiguous
coding `key` parameter, limiting capability comparisons. Original scores were
not revised. Adaptive 4,096-token retries recover five of six focused native
coding failures and three of five parent failures; these followups retain the
original 2,048-token results separately.

The evaluator preserves helpers supplied in the HumanEval prompt before the
entrypoint. Its focused regression check fixes the former missing-helper error;
reevaluation leaves all original aggregate scores unchanged. Remote generated code was
checked in resource-limited Linux subprocesses with a seccomp allowlist blocking
file opens, networking, and process execution. Isolation probes and reference
checks pass. Tool calls are parsed and checked, never executed.

## Paired native execution

Both artifacts run the same fresh 24 cases: 12 of the new HumanEval problems
(seed 2026101561) and all 12 explicit tool requests. Full schemas and canonical
tokens pass native parity. Both use greedy decoding, 4,096 output tokens,
8K context, FP32 activations, 8-bit affine KV, disabled prefix caching and
continuous batching, and one resident model at a time. The user discontinued RunPod and waived
Linux validation before the native answers were scored. No Linux VM was started
and no native generated Python was executed. Native tool calls were safely parsed
and checked against the frozen schema and expected arguments; native coding
correctness remains unscored. The completed remote coding scores are unchanged.

| Native artifact | Fresh coding | Fresh exact tools | Schema valid | Parser preserves arguments | Elapsed | Incremental peak host swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous 18.95 GB | Not scored | 12/12 | 12/12 | 12/12 | 2152 s | 3.77 GB |
| Broader 18.95 GB | Not scored | 12/12 | 12/12 | 12/12 | 2268 s | 0.09 GB |

The broader full-logit native run completes 152 cases
and 8,673 scored tokens. Peak MLX allocation
is 20.93 GB; the largest recorded process footprint
is 22.55 GB. Host swap increases
0.37 GB above its
recorded baseline. Background activity affects these capacity observations.

Native versus CUDA mean KL is 0.000000543, worst case KL
0.000058463, and top-token agreement
100.0000%. The unchanged backend gates require mean KL at
most 0.000001, case KL at most 0.00001, and agreement at least 99.99%; the
broader backend gate **fails**. The focused
candidate also fails the global backend gate, despite passing the same gate
within coding and tool categories. Category results do not override global
qualification, and remaining numerical differences are not causally isolated.

## Context and capacity

The release CLI runs fresh-process ASCII-copy probes at temperature 0.1,
top-k 50, 8-bit KV, and a 768-token output limit:

| Native artifact | Context window | Prompt tokens | Exact final ASCII literal | Incremental peak host swap |
| --- | --- | ---: | --- | ---: |
| Previous 18.95 GB | 2K | 1275 | Pass | 2.11 GB |
| Previous 18.95 GB | 8K | 7415 | Pass | 0.17 GB |
| Broader 18.95 GB | 2K | 1275 | Pass | 1.20 GB |
| Broader 18.95 GB | 8K | 7415 | Pass | 0.15 GB |

The target is exactly `TANGERINE-4827`. These explicit ASCII prompts differ from
the earlier broad-language copy probe. They establish bounded retrieval at the
tested 2K/8K windows; they do not verify the full advertised model context.
The curated receipts retain source/CUDA comparisons and host memory samples.

Shared admission remains 16 GiB reclaimable memory and 8 GiB free disk.
No wired-memory or GPU limits were changed. Refused starts and scheduler waits
are preserved separately from completed model runs. Runtime allocation is about
21 GB, and host swap can increase; the 18.95 GB file size is not a zero-swap
or multitasking guarantee.

## Verification and disposition

The final repository gate passes lint, build, CLI help, package policy, hygiene,
5,453 XCTest cases (452 expected skips, zero failures), and 194 Swift Testing
cases across seven suites. An optional installed-test output-budget setting
allows the matched 4K comparison; its default stays 2K. Production sources and
the release CLI remain unchanged from the previous small-quant run.
Hosted CI has not run for this worktree.

Preferred local profile: **`parent`**. The choice follows the frozen
size, all-family fidelity, remote functional nonregression, native schema,
parser, and context checks, with native code nonregression still unverified.
The original plan requires that check before preferring the broader candidate. It does not change the absolute quality gates.
The source, larger control, previous small quant, and both new candidates remain
preserved. No managed-model promotion or Hugging Face upload occurred for this
pass. RunPod operations ended at the user's request. Its last observed state was
EXITED after a provider balance refusal; no restart, new query, or Linux fallback
was attempted after that instruction. All remote results used here were already
recovered and were locally hash-verified again after native execution.

The [remote receipt](./receipts/lfm25-code-tools-remote-2026-10-15.json) and
[native receipt](./receipts/lfm25-code-tools-native-2026-10-15.json) contain
model identities, hashes, per-case outcomes, failed gates, and software checks.
Full logits, frozen inputs, source snapshots, executed tools, generated outputs,
and memory samples remain in the T7 qualification directory. Prior reports
retain their original evidence and conclusions.
