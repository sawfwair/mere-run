# LFM2.5 24B-A2B native routing investigation

This report records native numerical qualification for runtime maintainers.
The runtime matches upstream's positive-score expert partition ordering.
The isolated CUDA comparison failure also occurs in the independent upstream
implementation on Metal. A diagnostic replay identifies a nearly tied expert
selection as the cause of the large downstream difference.

## Full frozen confirmation

Both native Swift and the independent upstream implementation complete all
**152 cases and 8,673 scored token positions** on the same Mac.
The converted model remains the broader 18.95 GB artifact from the
[previous coding and tool experiment](./lfm25-24b-a2b-coding-tools-2026-10-15.md).
Its conversion receipt SHA-256 remains
`90ff17a6f0749565ef7f9967916fa00a68f705d989350ba5f32d46388dfa2ce6`.
Stored weights and source-quality thresholds remain unchanged.
The ordering change lowers average error against CUDA and raises average error
against the local upstream Metal reference. Both averages stay below the limit;
neither case-level gate clears. This is a reference-order alignment, with no
established coding-capability gain.

| Comparison | Mean KL | Worst case KL | Top-token agreement |
| --- | ---: | ---: | ---: |
| Previous Swift / CUDA | 5.430266782e-07 | 5.846276233e-05 | 100.0000% |
| Aligned Swift / CUDA | 5.215834069e-07 | 5.845218087e-05 | 100.0000% |
| Previous Swift / upstream Metal | 9.491802571e-08 | 1.074574291e-05 | 100.0000% |
| Aligned Swift / upstream Metal | 1.190090729e-07 | 1.074442289e-05 | 100.0000% |
| Upstream Metal / CUDA | 6.360880244e-07 | 5.846219808e-05 | 100.0000% |

All three implementation comparisons use the original limits: mean KL at most
`1e-6`, case mean KL at most `1e-5`, and top-token agreement at least `0.9999`.
The aligned Swift/CUDA gate **fails**.
The aligned Swift/upstream-Metal gate **fails**.
The independent upstream-Metal/CUDA gate
**fails**.
These are separate results; a passing local comparison does not override a
failed CUDA comparison.

Aligned Swift cases above the CUDA case-KL limit: `v4-heldout-prose-24`.
Aligned Swift cases above the upstream-Metal case-KL limit: `v3-wikitext-calibration-008`.
The local Metal case-limit miss is also present in the preserved pre-change
native output (`1.07457e-5`); it is not introduced by the order alignment.

The source-distortion holdout remains unqualified: mean KL
0.02421046, top-token agreement 94.2121%, and
worst case mean KL 0.05456486.
The model retains `quality_qualified: false`.

## Cause and controlled experiments

The instrumented legacy path reproduces the five saved native outputs with
zero logit differences. The original executable remains preserved.
The focused suite includes three numerical outliers, one coding case, and one
tool case. Its rows are copied unchanged from the frozen corpus.

For `v4-heldout-prose-24`, layer 12, input position 344, the fourth and fifth
expert scores differ by approximately `8.20e-8`. The native and independent
Metal traces select expert 3 at this boundary. A diagnosis-only replay replacing
expert 3 with expert 49 reduces mean KL against the saved CUDA output from
approximately `5.85e-5` to `4.12e-11`, with the same top token at every position.
The CUDA choice is inferred from this replay's output agreement; CUDA
intermediate routing was not directly captured.

The intervention establishes causality for the branch. It is excluded from the
production runtime and every qualification result. Full confirmation uses the
ordinary router with unchanged sigmoid probabilities, selection-only expert
bias, normalization epsilon, and routed scaling.

The local probes also test:

- Upstream positive-score partitioning, which removes the smaller calibration
  discrepancy in `v3-wikitext-calibration-025` against CUDA.
- Sorted expert evaluation and one-bank-at-a-time FP32 dequantized evaluation.
- The exact eight-case final CUDA reference batch and its padding width.
- Explicit FP32 parameter compute types and complete-expression SwiGLU compilation.
- CPU float64 router and normalization calculations from saved native inputs.

The additional probes leave the failing CUDA branch unresolved. The independent
upstream implementation on the same Metal device reproduces the same failure.
This case therefore does not establish a Swift-specific mathematical bug.
No targeted branch override is accepted as a runtime correction.

## Implementation and verification

`LFM2FeedForward` selects the final top-k entries from a positive-score partition,
matching the pinned upstream model. The analytic sigmoid-router test also covers
selecting every available expert, with and without probability normalization.
The opt-in diagnostic test records layer states, expert choices, and output
logits under the shared inference lease.

The reference uses [the pinned MLX-LM model source](https://github.com/ml-explore/mlx-lm/blob/ba8a08fd2ed72bbc53de84f89933f91fe4743e4a/mlx_lm/models/lfm2_moe.py),
its matching switch layers, and MLX/Metal 0.32.1. It promotes one active layer's
BF16/FP16 parameters to FP32, matching the saved reference scorer. A namespace
shim avoids unrelated application imports; original model source and initializer
files remain preserved with their hashes and license.

The canonical fixture SHA-256 remains
`0c50530b3d74b1d44ba2759d1b944dbbd763eac89cf133659f34e92147e6255e`.
Tokenization parity passes before native weight evaluation. Full-vocabulary
comparison verifies every saved logit file against its receipt hash.
The focused software tests pass 14 XCTest cases, including one expected skip.
An initial full repository check failed while the Mac was locked: protected
temporary-record reads returned EPERM, with 52 failures across 5,454 XCTest
cases (453 skips). The separate 194 Swift Testing cases passed. A disposable-file
control reproduced the protected-read failure.

The subsequent unlocked-device rerun passed `./scripts/check.sh`: 5,454 XCTest
cases (453 skips, zero failures) and 194 Swift Testing cases. The
[refinement receipt](./receipts/lfm25-local-refinement-2026-10-16.json)
records that software gate separately from the failed numerical gates. The
passing software check does not change the quantization or backend results.
Hosted CI has not run for this followup.

The full-vocabulary measurements use the Debug XCTest build. Release CLI
verification is recorded separately. The full comparison's source and test hashes remain unchanged. A later repository check relinked the Debug test executable; its retained copy is not the exact executable used for the numerical sweep. The sweep's executable hash and all output receipts remain recorded. The exact release executable used for the probes is preserved.

The optimized release CLI build passes. Its SHA-256 is `d45a79e4fabaf8e6bc12736106456cc1f466d7fe247fed9354a58e819b0acccc`.

| Context limit | Exact ASCII result | Elapsed | Incremental peak host swap |
| --- | --- | ---: | ---: |
| 2,048 | Pass | 50.2 s | 0.00 GB |
| 8,192 | Pass | 80.3 s | 0.00 GB |

Both probes use the frozen ASCII-copy prompts, 8-bit KV, and a 768-token output cap. They verify the tested 2K/8K windows; full advertised context capacity remains unverified.

Shared admission remains 16 GiB of reclaimable headroom and one resident model.
The original model files, larger control, previous small quant, release executable,
and pre-change sources remain preserved. This investigation uses local Metal
computation; it performs no RunPod operations or Linux validation. Native coding
assertions remain unscored under the earlier scope amendment.

The [machine-readable followup receipt](./receipts/lfm25-native-routing-2026-10-16.json)
records both backend boundaries, the failed gates, causal probe, hashes, and
software evidence. Full logits, frozen inputs, original binaries, layer traces,
and executed tools remain in the T7 qualification directory.
