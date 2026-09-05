# Qwen and Ornith generation tuning

This qualification follows the [Flash-Next transfer assessment](ornith-qwen-flash-transfer-2026-09-05.md)
and merged [PR #442](https://github.com/sawfwair/mere-run/pull/442). It tests
complete generation with the installed Qwen3.8-27B Q4 and Ornith 1.5 Q4
checkpoints on an Apple M4 Max with 128 GB unified memory.

The implementation is experimental. The final workload table and production
settings remain pending quiet-window measurements. Target-verification rates
from the transfer assessment are not generation throughput.

## Changes under qualification

Each Q35 request can own its compiled activation graphs. Swift inference uses
task-local MLX streams, while the C++ compilation cache uses the thread's
default stream in its cache key. Reusing compiled functions across requests
can retain graphs traced on an earlier request's stream. The request scope
covers SiLU, SwiGLU, precise gated normalization, and decay preparation.

The Q4 Qwen 27B and Ornith geometries can submit short decoder blocks
asynchronously. Final logits are evaluated before target acceptance or cache
commit. Prefill and unsupported geometries retain blocking boundaries.

When the greedy drafting policy permanently selects zero draft tokens, the
remaining generation can use the existing pipelined target decoder. The
draft-cost estimate is configurable for matched comparisons.

The experiment controls are as follows:

| Environment variable | Experimental value | Default |
| --- | --- | --- |
| `MERERUN_Q35_SCOPED_COMPILE` | `1` creates request-owned compiled graphs | Off |
| `MERERUN_Q35_ASYNC_DECODE_BLOCKS` | `1` submits supported short blocks asynchronously | Off |
| `MERERUN_Q35_MTP_PIPELINED_FALLBACK` | `1` pipelines permanent target fallback | Off |
| `MERERUN_Q35_MTP_HEAD_COST_RATIO` | Positive finite value up to `5` | `0.18` |
| `MERERUN_Q35_MTP_PROFILE` | `1` records synchronized phase timings | Off |

Generation block defaults remain eight tokens for Qwen and four for Ornith.
The cost and block-size sweep must complete before changing these defaults.

## Measurement contract

Use an optimized binary. Keep a model loaded across warm-up and measured
requests, disable prefix caching and continuous batching, and retain every
measured repetition. Report decode throughput and complete-request throughput
separately. Complete-request timing includes prefill and request overhead but
excludes the initial model load warmed in a separate request.

The receipt collector covers code, chat, prose, math, and summarization.
Greedy comparisons require identical output hashes and generated-token
counts. Single-variant receipts require an explicit matching target-only
reference. Sampled chat and prose use a separate measurement and do not claim
greedy-output parity. The fixed-token benchmark ignores EOS, so these prompts
measure runtime behavior rather than completion quality.

Before a run, the collector requires normal memory pressure, no detected
competing inference or active HyperFrames GPU worker, and device utilization
of at most 10%. It records competing processes throughout the run. Retain
affected receipts as diagnostics and repeat them in a separate directory.
Process polling cannot prove the absence of every possible source of GPU
contention.

The initial selection uses 256-token warm-up and measured requests, with three
measured repeats. Final measurements use two 512-token warm-ups and three
512-token measured requests per variant. Compare reversed variant order to
check order effects. Synchronized profiling is excluded from TPS conclusions.

## Validation receipts

The runtime at `f847be2ef7ca2f8896829c3cc0b6a1a55ea72ea9` passed the local
gate: 3,907 XCTest checks with 309 opt-in skips and no failures, plus 51 Swift
Testing checks. The gate also completed lint, build, CLI help, and hygiene
checks. The documentation build passed.

Installed-checkpoint checks passed on both models with the proposed scheduling
and fallback controls enabled. These checks cover a prompt over 4,096 tokens,
exact cache replay, follow-up reuse, preserved original checkpoints, and exact
code and prose agreement with target-only decoding. Separate Metal checks
cover request-owned graphs, activation arithmetic in FP32, FP16, and BF16,
draft-history snapshots, attention boundaries, and verifier parity.

Earlier timing receipts predate the GPU-utilization start check. They remain
diagnostic evidence and do not establish the final quiet-window TPS table.
No final performance promotion is claimed by these correctness checks.
