# Qwen and Ornith generation tuning

Managed Qwen3.8 27B Q4 and Ornith 1.5 Q4 now keep compiled activation graphs
inside each request's MLX stream, submit short decoder blocks asynchronously,
and use pipelined target decoding after permanent greedy MTP fallback. These
changes address repeated-request stalls. Other model IDs retain their scheduling
defaults. Draft-cost and generation block-size defaults are unchanged.

This follow-up to the [Flash-Next transfer assessment](ornith-qwen-flash-transfer-2026-09-05.md)
and [PR #442](https://github.com/sawfwair/mere-run/pull/442) measures complete
generation. Target-verification rates from that assessment are not generation
throughput. [PR #443](https://github.com/sawfwair/mere-run/pull/443) contains the
scheduling changes and the repeated-request benchmark harness.

## Final workload results

The tables show median decode tokens per second, with all three repetitions'
minimum and maximum in parentheses. Serial disables MTP; adaptive MTP is the
managed model's default policy. Both use the new scheduling defaults. Request
TPS includes prefill and request overhead, with model loading warmed separately.
The MTP/serial column divides the two decode medians.

These are observations from an active desktop on an Apple M4 Max with 128 GB
unified memory. No exclusive-machine throughput ceiling is claimed. Fixed-token
runs ignore EOS; the prompts measure runtime behavior, not completion quality.

### Ornith 1.5 35B-A3B Q4

| Workload | Serial decode TPS (range) | Adaptive MTP decode TPS (range) | MTP request TPS | MTP TTFT (ms) | MTP/serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 130.8 (129.5–132.4) | 65.9 (65.8–68.1) | 64.1 | 166 | 0.50× |
| Chat | 64.3 (61.8–69.7) | 56.0 (55.9–62.2) | 54.0 | 199 | 0.87× |
| Prose | 67.0 (58.2–69.1) | 64.4 (62.2–68.0) | 62.7 | 115 | 0.96× |
| Math | 70.3 (66.1–73.8) | 80.7 (52.5–81.9) | 78.3 | 106 | 1.15× |
| Summarization | 73.1 (62.5–83.9) | 81.6 (78.0–89.4) | 64.5 | 1101 | 1.12× |

### Qwen3.8 27B Q4

| Workload | Serial decode TPS (range) | Adaptive MTP decode TPS (range) | MTP request TPS | MTP TTFT (ms) | MTP/serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 12.1 (11.9–12.5) | 28.8 (28.4–30.1) | 26.8 | 703 | 2.38× |
| Chat | 11.8 (10.9–12.7) | 20.9 (20.7–21.3) | 18.7 | 1441 | 1.78× |
| Prose | 13.3 (13.2–13.5) | 14.6 (14.4–16.0) | 13.8 | 935 | 1.09× |
| Math | 11.2 (10.1–11.7) | 23.8 (23.7–23.9) | 22.0 | 855 | 2.13× |
| Summarization | 11.8 (11.1–11.9) | 23.5 (22.5–23.6) | 17.8 | 3483 | 1.99× |

MTP does not beat serial decoding on every prompt. Absolute rates also differ
substantially between the selection sweep, the final table, and the reversed
order controls. These runs do not isolate the cause of that variation or establish
a release-to-release speedup. Keep both repetition ranges and order controls
visible when comparing settings on a shared workstation.

### Reversed order

These additional code and prose runs execute adaptive MTP before serial. The
main code and prose table runs serial first. Each order has its own warm-up and
three measured requests; the results are not pooled.

| Model | Workload | Reversed serial TPS (range) | Reversed MTP TPS (range) |
| --- | --- | ---: | ---: |
| Ornith 1.5 35B-A3B Q4 | Code | 50.2 (45.5–54.7) | 47.8 (43.9–53.1) |
| Ornith 1.5 35B-A3B Q4 | Prose | 78.4 (71.5–83.9) | 44.4 (25.6–61.2) |
| Qwen3.8 27B Q4 | Code | 12.5 (9.2–14.2) | 24.1 (22.9–24.8) |
| Qwen3.8 27B Q4 | Prose | 8.8 (8.3–9.5) | 16.0 (9.9–17.5) |

### Sampled generation

Separate runs use temperature 0.7 and top-p 0.9. Output hashes and every
repetition are retained, but sampled outputs do not claim greedy parity or
identical continuations across variants.

| Model | Workload | Serial decode TPS (range) | Adaptive MTP decode TPS (range) | MTP request TPS |
| --- | --- | ---: | ---: | ---: |
| Ornith 1.5 35B-A3B Q4 | Chat | 44.2 (26.7–52.4) | 51.4 (44.5–51.8) | 49.0 |
| Ornith 1.5 35B-A3B Q4 | Prose | 89.7 (76.2–106.0) | 63.3 (62.3–68.0) | 61.7 |
| Qwen3.8 27B Q4 | Chat | 14.3 (9.1–15.4) | 20.3 (20.0–21.0) | 18.4 |
| Qwen3.8 27B Q4 | Prose | 15.7 (14.7–16.0) | 17.0 (16.7–17.6) | 16.1 |

## Setting selection

The selection sweep used the same earlier optimized binary with explicit
scheduling switches off or on. It compared the existing draft-cost estimate
0.18 against 0.35 for Ornith and 0.30 for Qwen. Each setting retained every
repetition, including the large stalls in the switches-off control. These
controls describe the tested switches, not a separate release-to-release
benchmark.

| Model | Settings | Code MTP TPS (range) | Prose MTP TPS (range) |
| --- | --- | ---: | ---: |
| Ornith 1.5 35B-A3B Q4 | Scheduling off, cost 0.18 | 94.3 (33.8–97.2) | 23.0 (19.1–74.9) |
| Ornith 1.5 35B-A3B Q4 | Scheduling on, cost 0.18 | 133.6 (133.1–137.5) | 119.5 (119.0–121.1) |
| Ornith 1.5 35B-A3B Q4 | Scheduling on, cost 0.35 | 135.2 (122.8–141.0) | 116.1 (109.8–127.7) |
| Qwen3.8 27B Q4 | Scheduling off, cost 0.18 | 30.0 (29.5–31.5) | 22.5 (19.0–22.9) |
| Qwen3.8 27B Q4 | Scheduling on, cost 0.18 | 34.8 (33.7–35.6) | 21.8 (21.4–21.9) |
| Qwen3.8 27B Q4 | Scheduling on, cost 0.30 | 36.5 (30.1–37.0) | 22.2 (21.8–22.5) |

Higher draft costs did not establish a consistent benefit across these prompts,
so the cost remains 0.18. Generation blocks remain eight tokens for Qwen and
four for Ornith. The scheduling changes passed exactness checks and address the
repeated-request stalls; these desktop measurements do not establish a universal
percentage speedup.

## Runtime behavior and controls

Swift inference uses task-local MLX streams, while the C++ compilation cache
uses the thread's default stream in its cache key. Sharing compiled functions
across requests can reuse graphs traced on an earlier request's stream. Each
managed Q4 request now owns its SiLU, SwiGLU, precise gated-normalization, and
decay-preparation graphs.

Supported Q4 Qwen 27B and Ornith geometries submit short decoder blocks
asynchronously. Final logits are evaluated before target acceptance or cache
commit. Prefill and unsupported geometries retain blocking boundaries. Once the
greedy adaptive policy permanently selects zero draft tokens, the remaining
request uses the existing pipelined target decoder.

| Environment variable | Explicit control | Managed Q4 default |
| --- | --- | --- |
| `MERERUN_Q35_SCOPED_COMPILE` | `0` restores shared compiled graphs | On |
| `MERERUN_Q35_ASYNC_DECODE_BLOCKS` | `0` restores blocking submission | On |
| `MERERUN_Q35_MTP_PIPELINED_FALLBACK` | `0` restores synchronous target fallback | On |
| `MERERUN_Q35_MTP_HEAD_COST_RATIO` | Positive finite value up to `5` | `0.18` |
| `MERERUN_Q35_MTP_PROFILE` | `1` records synchronized phase timings | Off |

Other model IDs require an explicit `1` to opt into the experimental scheduling
controls; geometry and runtime-path eligibility checks still apply. Synchronized
profiling changes scheduling and is excluded from all throughput tables.

## Measurement and provenance

Each final variant keeps its model loaded for one 256-token warm-up followed
by three 256-token measured requests. Prefix caching and continuous batching are
disabled. Greedy runs use temperature 0, top-p 1, and an 8,192-token context
limit. Main code, prose, and summarization cases run serial first; chat and
math run MTP first. The five fixed prompts are retained in the receipts. The 18 final receipts
contain 108 measured requests: 84 greedy and 24 sampled. All 84 greedy requests
preserved exact output hashes and token counts, including reversed-order runs.
Code and prose also match the selection sweep's frozen output hashes.

The collector's default admission requires at most 10% initial GPU utilization,
normal memory pressure, and no detected inference or active HyperFrames GPU
worker. This run explicitly uses `--allow-desktop-load`, which records
`measurementMode: desktop-load` and `uncontended: false` on every receipt. It
retains GPU samples, CPU load averages, memory pressure, swap state, and detected
worker counts. GPU samples during inference include this benchmark's own work.
Polling cannot establish the absence of every source of contention.

This task's runtime builds and correctness tests finished before the final throughput
run. Earlier selection runs overlapped local CPU validation in some cases and
remain separate from the final workload tables. Longer requests, exclusive-machine
measurements, vision workloads, and model-quality evaluation are outside these
results.

Across final receipts, initial GPU utilization ranged from 0% to 72%,
and the initial one-minute CPU load average ranged from 3.96 to 45.20.
The maximum recorded memory-pressure level was 1; the largest per-receipt
swap increase was 0.00 MiB.

| Model | Max post-request serial RSS (GiB) | Max post-request MTP RSS (GiB) |
| --- | ---: | ---: |
| Ornith 1.5 35B-A3B Q4 | 8.71 | 8.82 |
| Qwen3.8 27B Q4 | 14.38 | 14.74 |

Resident memory is sampled after each request; the receipts separately record system pressure and swap.

The final release binary was built from `c3ef55db5f09b2bcc6526bec4dd2a78eaa27aee6` using
`swift build -c release --product mere.run --disable-index-store`. Its SHA-256 is
`37ae108285410acec9ee3863de69475fb74ce9b472bf23c05e0e0df8ce26414c`.
Report-only changes after that revision do not change
its runtime sources. The environment used macOS 26.5.2 (25F84), Apple Swift 6.3,
and mlx-swift revision `7558b9cff75746e3ce25802aecbdc498b240af7f`.

The [machine-readable receipts](receipts/ornith-qwen-generation-tuning-2026-09-05.json)
include every measured repetition from both selection and final runs, binary and
collector hashes, the build revision, synthetic prompts, model source revisions,
and numeric load samples. Machine-local paths and process identifiers are
omitted. The receipt file's SHA-256 is
`cb73aa4dfe4f007301277af9ab28354a0ed51faf196e8138bcdda73edb65fd76`.

To reproduce the greedy workload sweep with installed models:

```bash
swift build -c release --product mere.run --disable-index-store
python3 scripts/benchmark-q35-generation.py \
  --binary .build/release/mere.run \
  --build-source-revision "$(git rev-parse HEAD)" \
  --model ornith --output .build/q35-generation-ornith \
  --workloads code chat prose math summary \
  --tokens 256 --warmups 1 --warmup-tokens 256 --repetitions 3 \
  --variants baseline,adaptive --allow-desktop-load
```

Repeat with `--model qwen` and a new output directory. Reverse the variants for
an order control; use MTP first for chat and math to match the main table.
Add `--temperature 0.7 --top-p 0.9` for separate sampled runs.
Omit `--allow-desktop-load` when collecting a quiet-window qualification.

## Validation

The runtime revision passed `./scripts/check.sh`: 3,909 XCTest cases with 309
opt-in skips and no failures, plus 51 Swift Testing checks. This gate includes
strict lint, build, CLI help, documentation examples, and hygiene checks. Five
receipt-collector tests passed, including contention labeling, critical-pressure
termination, and rejecting a binary replaced during measurement.

Eight targeted Metal checks passed for request-owned graphs, FP32/FP16/BF16
activation arithmetic, Q4 verification, accepted-prefix restoration, and draft
history. Both installed managed Q4 checkpoints passed a prompt over 4,096 tokens,
exact cache replay, follow-up reuse, preservation of the original cached prompt,
and exact 256-token code/prose agreement with target-only decoding using the
new defaults. The final greedy receipts add exactness checks across all five
workloads, with order controls for code and prose. The documentation build passed.
