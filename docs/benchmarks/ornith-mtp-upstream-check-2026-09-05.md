# Ornith MTP upstream comparison

Replacing only Ornith 1.5's MTP draft head improved draft acceptance in the
existing mere.run runtime. With three drafts per verification round, code
acceptance increased from 34.8% to 70.7%, and prose acceptance increased from
13.4% to 31.0%. All 40 measured requests preserved the serial target's exact
output hash and 256-token count. These results identify head quality as a
limiting factor on these prompts; they do not establish that the runtime has
no remaining performance issues.

This is a follow-up to the [complete generation workload report](ornith-qwen-generation-tuning-2026-09-05.md).
The replacement head remains an experiment. The scheduling defaults in
[PR #443](https://github.com/sawfwair/mere-run/pull/443) retain the installed
model weights and head selection.

## References and implementation checks

The main architecture reference is vLLM's
[Qwen3.5 MTP implementation](https://github.com/vllm-project/vllm/blob/f4eccdadefc6501fafeb1a0bf7f171ff24f984b0/vllm/model_executor/models/qwen3_5_mtp.py).
It normalizes the next-token embedding and target hidden state separately,
concatenates embedding then hidden, applies the fusion projection and a
full-attention decoder layer, and returns the head's normalized output.
mere.run follows these operations and reuses the target output projection.

The Ornith target supplies its post-normalization hidden state to MTP. This
matches the [merged llama.cpp correction](https://github.com/ggml-org/llama.cpp/pull/24025).
The installed raw HF head uses zero-centered RMSNorm weights, which mere.run
loads without subtracting one and evaluates with a `1 + weight` scale.
The inspected vLLM path uses the corresponding Gemma-style RMSNorm.

An Apple Silicon reference is
[oMLX's Qwen3.5 MoE MTP adapter](https://github.com/jundot/omlx/blob/e467261edc786efd33b1e9023d5c4a827f8aa1c1/omlx/patches/mlx_vlm_mtp/qwen35_moe_vlm_runtime.py).
That revision captures a pre-normalization target state, so it is not an
interchangeable reference for the hidden-state input checked here. It also
handles mixed raw-HF and converted-MLX head normalization conventions. The
installed head inspected in this experiment uses the raw-HF convention.

[Shisa's replacement head](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY)
provides a concrete head-quality comparison. Its publisher reports weak stock
head acceptance and documents a Qwen3.6-initialized, KL-distilled replacement.
The publisher marks it experimental and reports unresolved runtime correctness
issues in some vLLM tests. Their throughput figures are not Mac measurements.

Our native head contains 785 tensors. Local samples include a projection
standard deviation near 0.02 and `mtp.norm.weight` mean 0.022813. These are
consistent with the reported stock-head statistics; they do not prove whether
or how the head was trained.

## Head comparison

Both model roots use the same installed Ornith Q4 target files. The experiment
uses a separate root with symbolic links to the target and a replacement MTP
component. No installed model files were changed.

The replacement is pinned to revision
`2b19b31bfe1659c6b0d9459ec3cbd87e34a322ef`. Its original 1,689,283,688-byte
safetensors file matches the publisher's SHA-256:
`73c6e839971fff3c6d78dbcb6a15895bbab340a2898e98aa6943070751de712e`.
The 19 fused tensors were expanded into 785 per-expert tensors for the existing
loader by copying BF16 bytes unchanged. Every output tensor was read back and
hash-checked. No normalization values or quantization were changed.

### Three-draft diagnostic

Each row contains one measured 256-token request after a 256-token warm-up.
The diagnostic uses block size 4 and a draft-cost estimate of `1e-100`.
Counters verify three proposed drafts in every round except the final token
budget tail. This cost override is a measurement control, not a selected default.

| Workload | Stock accepted / drafted | Stock acceptance | Replacement accepted / drafted | Replacement acceptance | Stock / replacement rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 130 / 374 | 34.8% | 174 / 246 | 70.7% | 125 / 82 |
| Prose | 73 / 544 | 13.4% | 123 / 397 | 31.0% | 182 / 133 |

### Adaptive policy

These rows use the default draft-cost estimate of 0.18 and block size 4, with
three measured requests after a 256-token warm-up. Acceptance counts were
identical across repetitions. Decode TPS is the median with minimum and maximum
in parentheses.

| Workload | Stock acceptance | Replacement acceptance | Stock adaptive TPS (range) | Replacement adaptive TPS (range) |
| --- | ---: | ---: | ---: | ---: |
| Code | 51.9% | 71.7% | 94.4 (77.2–96.9) | 99.6 (93.3–110.9) |
| Prose | 25.9% | 38.3% | 73.8 (66.6–74.7) | 76.8 (75.1–84.9) |

The adaptive policy can stop drafting, so its acceptance rates are conditional
on the drafts it chooses to attempt. They are not directly comparable to the
three-draft diagnostic. The CLI's `forced` variant also adapts draft depth; it
forces MTP admission and is retained as an additional repetition control.

Other inference was detected during some runs. The same-target serial medians
varied between 80.7 and 85.4 TPS for code and between 78.2 and 83.0 TPS for prose.
These shared-workstation timings do not isolate the speedup caused by changing
the head. Higher draft acceptance reduces verification work, but total latency
also depends on draft computation, verification cost, and system load.

## Evidence and limits

The eight [machine-readable receipts](receipts/ornith-mtp-head-comparison-2026-09-05.json)
retain all 40 measured requests, output hashes, acceptance counts, timing, GPU
and CPU samples, model provenance, conversion hashes, and script hashes. The
artifact SHA-256 is
`7505b6706c2da8effd233aafc82a698bb659ae146b441086bf459fd1317f42ba`.
The runtime is the unchanged `c3ef55db5f09b2bcc6526bec4dd2a78eaa27aee6` release
build used in the generation report.

This check covers two short text prompts with greedy, fixed-length generation.
It does not qualify replacement-head behavior for long contexts, tools,
multimodal input, sampling, or concurrent serving. Exact output checks establish
the tested verifier behavior; they do not measure downstream answer quality.
The replacement needs broader model qualification before becoming a default.
