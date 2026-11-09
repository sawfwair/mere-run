# Qwen3.8 27B: matched Splash and mere.run measurements

Splash exceeded 100 tokens per second on this M5 Max. The highest qualified
request reached 165.7 TPS. On the main code prompt with reasoning off,
Splash's median was 121.5 TPS and mere.run DFlash 2's was
70.6 TPS. With medium reasoning, the medians were
107.5 and 55.9 TPS.

Both engines loaded the same pinned Q4 target and BF16 draft source files. Each
Splash request received the exact prompt token IDs captured from mere.run.
Both used greedy decoding, BF16 KV, FP32 recurrent state, an 8,192-token context
limit, and a 1,024-token output limit with EOS stopping enabled. Splash's Neural
Engine prefill path was disabled.

## Main results

Cells show the median of three requests and their minimum and maximum. Rates
exclude the first emission. The summary retains every measured request,
actual output count, acceptance rate, and timing field.

| Workload / reasoning | mere.run DFlash 2 TPS (range) | Splash 1.6.0 TPS (range) | Splash / mere.run |
| --- | ---: | ---: | ---: |
| Code / off | 70.6 (56.8–78.9) | 121.5 (117.3–144.4) | 1.72× |
| Code / medium | 55.9 (54.2–59.7) | 107.5 (104.5–123.8) | 1.92× |
| Prose / off | 24.5 (23.7–24.7) | 46.5 (45.3–48.3) | 1.90× |
| Prose / medium | 25.9 (24.8–26.7) | 50.9 (50.4–55.8) | 1.97× |

These measurements use two local prompts. The [Splash launch suite](https://github.com/incoai/splash/blob/1.6.0/docs/performance.md#splash-10-launch-benchmarks)
uses selected SPEED-Bench coding prompts; this run doesn't reproduce that
complete suite. It does independently test Splash's 100+ TPS claim on the same
Mac used for mere.run.

### Reversed code order

The main cases ran mere.run before Splash. These separate controls ran Splash
first, then mere.run, with their own warm-ups and three requests per engine.
Their results are kept separate from the main medians.

| Reasoning | mere.run DFlash 2 TPS (range) | Splash 1.6.0 TPS (range) | Splash / mere.run |
| --- | ---: | ---: | ---: |
| Off | 71.0 (70.8–75.9) | 143.6 (139.0–165.7) | 2.02× |
| Medium | 57.9 (57.8–63.0) | 108.1 (106.0–122.3) | 1.87× |

## Acceptance and generated tokens

| Workload / reasoning | mere.run acceptance | Splash acceptance | mere.run tokens | Splash tokens |
| --- | ---: | ---: | ---: | ---: |
| Code / off | 80.0% | 80.9% | 970 | 947 |
| Code / medium | 63.4% | 58.5% | 1024 | 1024 |
| Prose / off | 18.9% | 17.9% | 601 | 594 |
| Prose / medium | 20.8% | 21.0% | 781 | 742 |

The engines produce different greedy continuations despite identical source
weights and prompt tokens. Each engine repeated its own output hash within a
condition. The hashes use different output representations and don't establish
cross-engine output parity. Output lengths and acceptance are retained because
they affect decode throughput.

The code results also separate proposal quality from execution cost. Mere.run's
medium-reasoning code acceptance exceeded Splash's, but its throughput was
lower. Better acceptance alone doesn't close the measured gap.

## Separate synchronized profiles

Each profile uses the same prompt, reasoning setting, and token limit as its
main case, with one warm-up and one measured mere.run request. Its output hash
matches the corresponding unprofiled request. Profiling inserts synchronization
boundaries after drafting, target verification, and confirmed-context updates.
These timings are excluded from the throughput tables.

| Workload / reasoning | Draft | Target verification | Acceptance and emission | Cache update and repair | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code / off | 11.5% | 86.2% | 0.0% | 2.3% | 0.0% |
| Code / medium | 11.4% | 85.9% | 0.0% | 2.6% | 0.0% |
| Prose / off | 11.2% | 85.4% | 0.0% | 3.4% | 0.0% |
| Prose / medium | 11.3% | 85.1% | 0.0% | 3.6% | 0.0% |

Target verification is the largest recorded phase in all four profiles,
accounting for 85.1–86.2% of profiled decode time. This
points to the target verification path as the next optimization focus.

For code with reasoning off, the recorded target evaluation/readback wait was
10.75 seconds and graph-submission time was 0.34 seconds across 147 rounds.
Most of the measured verification cost occurs while obtaining target results.

Graph-submission time includes any waits inside MLX calls. Cache-update and
repair time includes target cache prefix restoration or replay and the draft's
confirmed-context projections. These are synchronized wall-time diagnostics,
not isolated per-kernel GPU timings or a direct decomposition of Splash's
execution. They don't identify a particular Metal kernel as the cause.

The port shares the DFlash 2 proposal mechanism with Splash but retains
mere.run's target execution path. Splash's
[shape-specialized runtime](https://github.com/incoai/splash/blob/1.6.0/README.md#design)
and its [Q4/group-64 draft preparation](https://github.com/incoai/splash/blob/1.6.0/runtime/model/DraftCheckpoint.cpp)
use a different execution path from mere.run's checkpoint integration. The
measured gap remains in target evaluation and verification.

## Isolation and provenance

The host has a 40-core Apple M5 Max and 128 GiB of unified memory. Both engines
ran from release binaries on the same workstation. Each condition had one
256-token warm-up and three measured requests. The collector waited for low
GPU utilization, normal memory pressure, no compilation, and no recognized
competing inference or rendering work. The summary records thermal states,
swap changes, every request, exact input token arrays, and file hashes.

The temporary Splash server stayed resident during mere.run to avoid repeating
its approximately four-minute startup. Its GPU counters had to stay unchanged
during every qualified mere.run request. Resident memory therefore differs
between the two engines' conditions; this is a shared-host comparison. GPU
frequency wasn't fixed, and the repetition ranges show substantial variation.

A Python inference runtime appeared around the first prose measurement window.
Both earlier prose attempts and an incomplete medium-prose attempt were
excluded conservatively. The prose reruns and reversed code controls use an
extended monitor that checks per-process GPU counters for Python and other
compute runtimes. Main code runs used the original known-process monitor;
the summary labels each monitor version. No unrelated service was stopped.

Splash used an isolated 64 MiB SSD-cache quota. Its startup disabled the cache
because a recurrent state requires about 152 MiB; `/status` confirmed zero disk
cache capacity and reads. Prefix reuse in memory remained available to Splash.
Mere.run's experimental DFlash route used a fresh prompt prefill. Decode rates
exclude prefill; client request timings in the summary include that difference.

The [machine-readable results](qwen38-splash-matched-2026-11-08-summary.json) contain
immutable model revisions, source-file digests, binary hashes, and every
request's metrics. Source models were reused locally; no model source files
were modified. The explicit text-only mere.run assembly also included official
Qwen vision configuration metadata under `vision/`, without vision weights.
Image inference was outside this comparison. The official Splash archive is
[`splash-1.6.0-arm64-macos26.tar.gz`](https://github.com/incoai/splash/releases/tag/1.6.0),
SHA-256 `da3aef0a581491b0eb4c1168bda07f520e5ca8368764abff426d95ac1c77e5c4`.
Its native binary and kernels were unchanged. The Python launch wrapper only
redirected its thinking-key file to the experiment directory.

After measurement, the three benchmark controls were registered in the shared
capability catalog and Studio's flag constants were regenerated. Inference
source is unchanged. The summary distinguishes
the measured binary and source digest from this CLI metadata correction and
the final release build.

## Reproduce the controls

For mere.run, select the pinned local target and draft, then use
`model benchmark q36-mtp` with `--variants adaptive --temperature 0 --top-p 1
--decode-tokens 1024 --stop-on-eos --context-size 8192 --warmups 1
--warmup-tokens 256 --repetitions 3 --json`. Medium reasoning adds `--thinking
--reasoning-effort 0.5`. Set `MERERUN_Q35_DFLASH2_PATH` to the draft directory
and `MLX_ENABLE_TF32=0`. The internal
`MERERUN_Q35_DEBUG_PROMPT_TOKENS` hook writes the exact prompt array as CSV.

For Splash, serve the same local target and draft with `--language-only
--kv-format bf16 --state-format fp32 --disable-ane --max-context 8192
--max-memory 48G`. Submit the captured integer array to `/v1/completions` as
`prompt`, with `temperature: 0`, `top_p: 1`, `max_tokens: 1024`, and
`ignore_eos: false`. A 256-token warm-up precedes the three measured requests.
Use the summary's input arrays to reproduce this run's exact prompts.

For diagnostic profiles, run mere.run separately with
`MERERUN_Q35_MTP_PROFILE=1` and `--repetitions 1`; keep those receipts out of
throughput comparisons. The new CLI controls and phase observer have focused
CPU and GPU coverage, including nonzero recurrent-cache parity with profiling.

## Validation

The required `./scripts/check.sh` gate passed with the native SwiftPM wrapper:
5,829 XCTest cases (483 skipped), 197 Swift Testing checks, lint, CLI help, and
repository hygiene checks. All 16 focused GPU tests passed. The final release
CLI builds, prints help, and exposes all three benchmark controls in its
capability catalog.

Release builds used `swift build --build-system native -c release --product
mere.run --disable-index-store`. Native SwiftPM avoids the checkout's existing
missing vendor dSYMs; no vendored artifact changed. The measured and final
binary hashes are recorded separately because the catalog and generated Studio
metadata were aligned after measurement.
