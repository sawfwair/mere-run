# Bonsai 2 Small upstream throughput and scheduling fix

This report is for runtime contributors comparing the native 2B, 4B, and 9B packs with their pinned upstream MLX implementations. A shared compiled-graph cache caused a model-order performance penalty. Each Small-model request now owns its compiled activation graphs. A fused Metal transform also reduces short-row signed-Hadamard dispatch work.

## Cause and control

Native Q35 activation helpers use compiled functions for SwiGLU and recurrent gates. The shared functions can retain a graph traced on the first generator’s task-local GPU stream. MLX’s compile-cache lookup uses its backend default stream, which does not identify the Swift task-local stream. Later models then reuse cross-stream activation graphs and incur synchronization.

The unchanged executable reproduces the effect when load order is reversed: 4B runs quickly when first, while 2B becomes slow when second. Both use 512-element Hadamard blocks. Enabling only `MERERUN_Q35_SCOPED_COMPILE=1` removes the large penalty in the original order. The source fix enables that qualified feature for the three Small IDs. Other scheduling features and diagnostic overrides retain their behavior.

| Cohort | Order | 2B text tok/s | 4B text tok/s | 9B text tok/s |
| --- | --- | ---: | ---: | ---: |
| baseline-order-4-2-9 | 4B → 2B → 9B | 29.22 | 81.54 | 18.50 |
| scoped-order-2-4-9 | 2B → 4B → 9B | 123.03 | 76.60 | 46.30 |
| optimized-default-2-4-9 | 2B → 4B → 9B | 123.68 | 78.19 | 55.23 |
| fused-reference-2-4-9 | 2B → 4B → 9B; Metal off | 134.85 | 90.13 | 58.52 |
| fused-default-2-4-9 | 2B → 4B → 9B | 142.41 | 76.24 | 46.47 |
| fused-default-9-4-2 | 9B → 4B → 2B | 138.37 | 90.95 | 50.86 |
| fused-reference-9-4-2 | 9B → 4B → 2B; Metal off | 130.57 | 87.26 | 58.25 |
| selected-default-2-4-9 | 2B → 4B → 9B | 105.78 | 81.79 | 57.96 |
| selected-default-9-4-2 | 9B → 4B → 2B | 116.30 | 76.45 | 52.56 |

The [earlier quality/throughput trial](./bonsai2-small-fixed-2026-10-16.md) used the previous executable in 2B → 4B → 9B order and reported 133.32, 16.20, and 16.39 text tok/s. Its quiet timing confirmation did not control model order; those observations remain preserved.

## Upstream comparison

All values are medians of three resident requests after one discarded 128-token warm-up per model and modality. Text requests allow 512 output tokens; vision requests allow 256. The primary native cohort uses the rebuilt production defaults in 2B → 4B → 9B order, without environment overrides. The reverse-order cohort is a separate confirmation, not a source of selected faster samples.

| Model | Upstream text tok/s | Native text tok/s | Upstream vision tok/s | Native vision tok/s |
| --- | ---: | ---: | ---: | ---: |
| 2B | 156.15 | 105.78 | 160.89 | 111.29 |
| 4B | 93.63 | 81.79 | 72.53 | 78.83 |
| 9B | 60.58 | 57.96 | 60.12 | 55.79 |

| Model | Upstream text range | Native text range | Upstream vision range | Native vision range |
| --- | --- | --- | --- | --- |
| 2B | 153.44–163.36 | 104.74–117.47 | 157.49–168.66 | 99.20–111.33 |
| 4B | 93.52–94.41 | 78.33–87.49 | 51.69–91.87 | 69.07–81.13 |
| 9B | 57.81–60.95 | 57.67–58.23 | 52.56–60.67 | 55.72–56.00 |

The reverse-order confirmation uses the same selected executable and reports every measured request separately.

| Model | Reverse native text tok/s | Reverse native vision tok/s |
| --- | ---: | ---: |
| 2B | 116.30 | 116.29 |
| 4B | 76.45 | 67.97 |
| 9B | 52.56 | 46.80 |

The primary 4B and 9B text rates are about 5.0 and 3.5 times the earlier 16.20 and 16.39 tok/s observations. The selected 2B cohort remains below upstream and below earlier native cohorts. Removing the experimental 4096 path does not change 2B's 512-element path; these desktop runs do not isolate the cause of that remaining gap. The unchanged-executable order and feature controls establish the shared-graph defect independently of these later timing differences.

## Transform optimization

The selected FP32 butterfly adds 512-element blocks to the existing 1024 block. The 512 path fuses at most eight activation rows; longer prefills retain the reference path. An experimental 4096 kernel passed numerical checks but reduced whole-model 9B text throughput in both model orders, so the selected runtime retains the reference path for 4096 blocks. Host-computed normalization bits preserve the reference scale exactly. Forward and inverse GPU tests have zero maximum numerical error across FP16, BF16, and FP32. Packed checkpoint weights are unchanged.

The `fused-reference-2-4-9` cohort disables all signed-Hadamard Metal kernels in the same executable. This includes the previously available 1024-element vision path, so it measures all transform fusion rather than only the new Small decode support. The scope-only cohort separately preserves the runtime before the new block. The additional reference/reverse control was planned after an early fused regression coincided with a nominal-to-fair thermal transition. Every cohort is retained; final results use the selected executable in the predetermined forward order, with reverse order reported separately. The experimental 4096 fused/reference text ratios were 0.79 forward and 0.87 reverse. Desktop activity and thermal variation prevent attributing every 512 timing difference to the kernel.

Experimental component profiles use real checkpoint gate, down, and output projections with three alternating-order trials of 50 synchronized evaluations after five warm-ups. Their binary identity is recorded separately from the final executable. The machine-readable summary retains reference/fused transform and packed-2-bit/expanded-4-bit timings. These isolate operator dispatch; they do not establish whole-model throughput. Expanded 4-bit output heads are slower in all three component profiles and require more storage; the shipped weights remain packed 2-bit. Some smaller MLP projections have overlapping timings, so the head result is not generalized to every projection.

## Conditions and interpretation

Host: Apple M3 Max, 36 GiB unified memory, macOS 26.6.2. Models, metadata, images, Python environment, build products, profiles, and raw results are on T7. The [summary](./bonsai2-small-throughput-2026-10-16-summary.json) retains source revisions, weight hashes, runtime versions, timers, thermal/memory snapshots, model-order controls, and admission checks.

Baseline release SHA-256: `6468cd000e7268c9ba1678892a96beccbf13f296fceacf970a4af7f72322d23a`. Optimized release SHA-256: `a1e3224fae3c4c85d54bcae618de8602c5cbfdd041105292f535ffbef25d54b3`. The native backend is MLX 0.32.1 through Swift fork `001d1aa3e5655d7efc073fb9632f952f57cdf528`. Upstream uses unmodified pack sources, verified against their pinned Git blobs, with MLX 0.32.0, mlx-vlm 0.6.3, mlx-lm 0.31.3, and Transformers 5.5.0. These backend versions are recorded separately; the unchanged-binary scheduling ablation isolates the native source change.

Every native and upstream request body matches the earlier frozen trial. All upstream prompt IDs match native tokenization exactly, and native prompt counts agree for both model orders. Requests use catalog/default thinking, temperature 1.0, top-p 0.95, top-k 20, min-p 0, no penalties, and no seed control. Native applies top-k before top-p; the pinned upstream sampler applies top-p before top-k. Knobs agree, while sampled-policy ordering differs. Input context is bounded to 8,192 tokens. Normal machine and request admission remain enabled; known native workloads are checked before and during measurement.

Decode rates exclude loading and prefill and include reasoning tokens. Vision rates describe language decoding after image input, not images per second. Upstream reports its own `stream_generate` generation counter, including detokenization; native uses product `lastRequest` counters. First-token timer boundaries differ, so these are product-level comparisons rather than instruction-identical microbenchmarks. Native prefix reuse follows server defaults; upstream requests start with fresh caches.

Three samples on an interactive desktop do not establish sustained performance or a population estimate. Thermal and process observations are evidence, not proof that every desktop GPU workload was absent. The previous sampled quality corpus is not regenerated for these performance changes. All 18 frozen text and vision cases and 930 teacher-forced positions pass with the same gates. Numerical parity and local validation receipts are retained separately.

## Validation and evidence

Evidence root: `/Volumes/T7/mere-run-qualification/bonsai2-small-throughput-2026-10-16`. The frozen upstream parity corpus, comparison outputs, focused tests, full repository gate, and final delivery receipts are preserved there. The selected runtime passes all 18 cases and 930 positions with unchanged tolerances; release build, full local gate, and focused GPU checks pass. Admission-only startup rejections are preserved separately and were retried without lowering the normal 16 GiB floor. The initial scope-only reverse cohort collected no requests and contributes no timing results. No hosted CI or PR publication was run.

To compare the pinned upstream pack, run `Bonsai2SmallThroughputTests` with the evidence root, model size, and reference Python specified in `scripts/reference-parity/README.md`. That test acquires normal standard admission before launching the reference process and checks exact native/upstream prompt-token equality. The product runtime remains native Swift/MLX.

Use the unchanged resident API collector for native timing, preserve every cohort, and compare both forward and reverse model order. Set `MERERUN_Q35_SCOPED_COMPILE=0` only for the diagnostic baseline.
