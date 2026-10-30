# Bonsai 2 Small fixed-runtime quality and throughput — 2026-10-16

This fresh run measures the native 2B, 4B, and 9B checkpoints on an Apple M3 Max after the [precise Metal correction](./bonsai2-small-upstream-parity-2026-10-16.md). Strict quality scores were 12/16 for 2B, 10/16 for 4B, and 12/16 for 9B, with median text decode rates of 133.32, 16.20, and 16.39 tok/s. It uses the same primary requests and scoring as the [original run](./bonsai2-small-2026-10-15.md), with release SHA-256 `6468cd000e7268c9ba1678892a96beccbf13f296fceacf970a4af7f72322d23a`. Python orchestrates and scores; inference is native Swift/MLX.

All model storage and raw results are on T7. The [machine-readable summary](./bonsai2-small-fixed-2026-10-16-summary.json) contains scores, selected outputs, native timing samples, source identities, and validation receipts. Public email domains are redacted as in the original report; raw T7 data is unchanged.

## Conditions

Host: Apple M3 Max, 36 GiB unified memory, macOS 26.6.2. Runtime source is `e419af12`; its full local gate passed in `7d63adc6`. The preserved executable, runtime source hashes, 50 upstream source files, and four image fixtures were verified. All 72 request bodies match the original trial. No model conversion, new quantization, MTP, or admission override was used.

Fresh loopback sessions use an 8,192-token context and serial requests. Quality was collected once. An unrelated active XCTest overlapped the initial throughput batch, so all 24 throughput requests were repeated in a fresh session after three quiet-start observations; the initial batch remains intact on T7 and in the summary. Reported throughput uses the complete confirmation cohort, selected for contention rather than speed. It follows the catalog thinking default and sidecar sampler: temperature 1.0, top-p 0.95, top-k 20, min-p 0, no penalties, and no seed control. Quality requests allow 4,096 output tokens. No output grammar is enforced. Scored failures are retained without regenerating answers. The prior four-case presence/budget probe is not repeated.

## Quality

The suite contains eight grounded cases, two generated Python functions with five tests each, four images, arithmetic, and reasoning. Strict scoring checks listed facts and parses JSON without fences; it is not a complete JSON schema or broad quality assessment. Code runs in the same closed evaluator with no imports, limited builtins/methods, a CPU limit, and a short wall timeout. Saved scores were independently recomputed. The following table shows strict scores per model.

| Model | Passed cases | Text cases | Grounded | Code tests | Vision strict | Arithmetic | Reasoning strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2B | 12/16 | 9/12 | 6/8 | 5/10 | 3/4 | Pass | Pass |
| 4B | 10/16 | 9/12 | 6/8 | 6/10 | 1/4 | Pass | Pass |
| 9B | 12/16 | 10/12 | 7/8 | 5/10 | 2/4 | Pass | Pass |

One complete outer Markdown fence is removed only for a separate recovery diagnostic. Primary strict scores remain unchanged. The following tables show recovery and latency results, then the failed primary cases.

| Model | Vision after fence recovery | Correct fields after recovery | Median quality request (s) | Slowest quality request (s) | 4,096-token caps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B | 3/4 | 17/18 | 7.50 | 34.78 | 0 |
| 4B | 3/4 | 17/18 | 29.56 | 104.50 | 0 |
| 9B | 3/4 | 17/18 | 39.78 | 229.30 | 2 |

| Model | Failed primary case IDs |
| --- | --- |
| 2B | 2b-grounded-3, 2b-grounded-7, 2b-code-merge_intervals, 2b-vision-shapes |
| 4B | 4b-grounded-3, 4b-grounded-7, 4b-code-merge_intervals, 4b-vision-shapes, 4b-vision-receipt, 4b-vision-chart |
| 9B | 9b-grounded-7, 9b-code-merge_intervals, 9b-vision-shapes, 9b-vision-chart |

All three models counted every shape correctly but failed the exact heading check: 2B returned `BONALCHECK`, 4B returned `shapes`, and 9B returned `BONSAI CHECK47` rather than `BONSAI CHECK 47`. The 4B receipt and chart and the 9B chart pass only after removing one complete outer Markdown fence. All three interval-merging answers fail the executable-code gate; 9B exhausts its output budget in a visible thinking trace. These failures are retained without changing the scorer.

## Resident throughput

The later [upstream comparison and performance fix](./bonsai2-small-throughput-2026-10-16.md) identifies a compiled-graph model-order penalty in this executable. The measurements in this section remain the original observations; use the later report for the corrected runtime.

Each model/modality has one discarded 128-token warm-up and three measured resident requests. Text budgets are 512 tokens; image budgets are 256. Native `lastRequest` timings provide decode seconds and token counts, checked against API usage. Prefix reuse follows the server default; the collector unloads each model between sizes. Decode TPS includes reasoning tokens and excludes model load and prefill. These capped requests are not scored for completion or quality. The following tables show decode rates, then prefill, first-token, and request medians.

| Model | Text median tok/s | Text range | Vision median tok/s | Vision range |
| --- | ---: | --- | ---: | --- |
| 2B | 133.32 | 132.73–133.81 | 126.21 | 123.49–126.27 |
| 4B | 16.20 | 16.15–16.24 | 17.24 | 16.72–77.25 |
| 9B | 16.39 | 16.15–17.02 | 16.57 | 15.83–54.84 |

| Model | Modality | Median prefill (s) | Median internal first token (s) | Median request (s) |
| --- | --- | ---: | ---: | ---: |
| 2B | text | 0.001 | 0.011 | 3.84 |
| 2B | vision | 0.616 | 0.627 | 2.68 |
| 4B | text | 0.001 | 0.065 | 31.61 |
| 4B | vision | 1.116 | 1.157 | 15.97 |
| 9B | text | 0.001 | 0.068 | 31.24 |
| 9B | vision | 1.903 | 1.948 | 17.36 |

Process, memory-pressure, and thermal snapshots are retained for every request. All measured request-end snapshots show nominal memory pressure. Thermal snapshots are `fair` for 2B, mixed `fair`/`nominal` for 4B, and `nominal` for 9B. No active matching native inference/compiler/test workload was observed in request-boundary or 15-second monitoring snapshots during the confirmation throughput span. Other queued native test processes, where present, are retained separately; admission and CPU snapshots distinguish them from running inference. This does not prove the absence of every desktop GPU workload.

## Original-run reference

Both runs use the same request bodies, but each has one unseeded quality trial and three timing repetitions per reported cohort on an interactive desktop. Differences are fresh observations, not an isolated causal estimate of the numerical fix or statistically established speed gains. The following table compares the two runs.

| Model | Original quality | Fixed quality | Original text tok/s | Fixed text tok/s | Original vision tok/s | Fixed vision tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2B | 9/16 | 12/16 | 22.52 | 133.32 | 112.55 | 126.21 |
| 4B | 12/16 | 10/16 | 15.47 | 16.20 | 15.41 | 17.24 |
| 9B | 12/16 | 12/16 | 16.52 | 16.39 | 18.52 | 16.57 |

## Evidence and reproduction

Evidence root: `/Volumes/T7/mere-run-qualification/bonsai2-small-fixed-2026-10-16`. It preserves the exact collector, frozen plan, source/fixture receipts, server command and status, every request/response, timing snapshots, and a hash-identical executable. Model storage remains in `/Volumes/T7/mere-run-qualification/bonsai2-small-2026-10-15/models` through verified T7 links. Normal machine and request admission stayed enabled. Each server waited for normal admission and was stopped after its collection. The confirmation session has its own command, status, quiet-start checks, and completion receipt under `throughput-confirmation`.

Use a fresh evidence directory with the same verified model and fixture layout. Start the fixed native loopback server with `--models-root` pointing to T7, `--context-size 8192`, `--max-active-requests 1`, `--no-warmup`, and its default model path on T7; keep `MERERUN_HUB_CACHE` and `HF_HOME` on T7. Run the preserved collector first with `--phase quality`, then `--phase throughput`, passing the exact binary, evidence root, and endpoint. Completed quality results resume only when their binary/corpus/request hashes match; scored failures are never retried.

The fixed runtime passed the separate 18-case upstream parity suite. This sampled quality run does not establish broad capability, long-context quality, smaller-memory admission, cross-device performance, or sustained long-run throughput. No hosted CI or PR publication was run.
