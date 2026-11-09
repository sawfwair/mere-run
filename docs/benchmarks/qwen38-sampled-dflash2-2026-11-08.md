# Qwen3.8 sampled DFlash 2 throughput

Status: **12 of 12 conditions qualified**. Collection is complete.

These measurements use Qwen's recommended sampled presets on a 40-core M5
Max with 128 GiB of memory. All reported conditions passed resource
qualification. They measure decode speed for the listed prompts and seed;
they do not establish output quality or a speed guarantee across workloads.

M5-backed DFlash 2 improves code throughput with both sampled presets and
thinking-mode prose. Non-thinking prose is 13% slower than target-only decoding.
The default policy accepts that measured tradeoff in exchange for the gains
across the other cases; it does not try to classify prose or code.

| Preset / workload | Target-only TPS | DFlash 2 TPS | DFlash 2 + M5 TPS |
| --- | ---: | ---: | ---: |
| instruct / code | 30.3 | 52.6 | 65.5 |
| instruct / prose | 33.0 | 24.5 | 28.7 |
| thinking / code | 33.1 | 45.3 | 46.8 |
| thinking / prose | 33.0 | 28.3 | 40.4 |

Each cell is the median of three measured requests. Code uses one loaded
process and one 256-token warm-up before the three requests. Prose uses
three separately loaded processes, each with its own 256-token warm-up,
so each measured request can qualify in a shorter idle window. All backends
within a workload use the same protocol. Load and warm-up time are excluded.
Requests stop at EOS with a 512-token cap and an 8,192-token context limit.
These are short prompts, rather than an 8K-prefill benchmark; exact prompt
token counts appear in the machine-readable summary.
Seed 42 resets for each request. Repeated seeded continuations measure
timing variation; these are not three independent output samples. Thinking
uses the model template's default reasoning effort. Its TPS includes reasoning
tokens, and the output cap can end a request before its final answer.

| Preset | Temperature | Top-p | Top-k | Presence penalty |
| --- | ---: | ---: | ---: | ---: |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 1.5 |
| Thinking | 1.0 | 0.95 | 20 | 0 |

Both presets use min-p 0, frequency penalty 0, and repetition penalty 1.
They follow [Qwen's published guidance](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices).

Target-only uses the existing projection backend and excludes both MTP and
DFlash. DFlash 2 uses that same backend with the trained draft. The M5
condition enables the tiled matrix backend for all supported target
projections and the fused dense FFN; independent draft layers keep their
reference backend. Each condition uses the same release binary and model
assets as the other conditions in this report. M5 defines a different
numerical reference. Speculation also consumes different random draws from
serial sampling, so equal seeds do not imply equal cross-route outputs.

## Timing detail

| Preset / workload / backend | Median TPS (range) | Generated tokens | Draft acceptance |
| --- | ---: | --- | ---: |
| instruct / code / target | 30.3 (28.0–31.7) | 512, 512, 512 | n/a |
| instruct / code / dflash | 52.6 (51.8–53.2) | 512, 512, 512 | 71.0% |
| instruct / code / m5 | 65.5 (58.2–77.1) | 512, 512, 512 | 73.3% |
| instruct / prose / target | 33.0 (32.9–33.4) | 512, 512, 512 | n/a |
| instruct / prose / dflash | 24.5 (23.9–24.5) | 512, 512, 512 | 16.9% |
| instruct / prose / m5 | 28.7 (28.0–29.8) | 512, 512, 512 | 15.1% |
| thinking / code / target | 33.1 (31.9–33.7) | 512, 512, 512 | n/a |
| thinking / code / dflash | 45.3 (42.6–48.4) | 512, 512, 512 | 54.3% |
| thinking / code / m5 | 46.8 (44.4–47.3) | 512, 512, 512 | 36.2% |
| thinking / prose / target | 33.0 (32.0–33.5) | 512, 512, 512 | n/a |
| thinking / prose / dflash | 28.3 (28.0–28.3) | 512, 512, 512 | 23.4% |
| thinking / prose / m5 | 40.4 (39.6–42.6) | 512, 512, 512 | 25.8% |

## Qualification and provenance

Admission requires 30 seconds without competing GPU compute or Swift
compilation, normal memory pressure, and at most fair thermal pressure.
Background desktop GPU utilization may be up to 50%. Resident servers are
allowed only while their compute GPU counters stay unchanged. Monitoring
continues throughout loading, warm-up, and measured requests. Contended
attempts are retained separately and excluded.

The [machine-readable summary](qwen38-sampled-dflash2-2026-11-08-summary.json)
retains each request, sampling settings, acceleration diagnostics, output
hashes, source hashes, and resource receipts. TPS is generated tokens divided
by decode seconds; load and prefill are excluded. No synchronized phase
profiling is enabled.

Release binary SHA-256:
`f03cb5c4482f9d96ea0334b6a4056eb304ac6ae36a54254b32d256f5f648e56d`.

The repository gate passed before this measurement: 5,852 XCTest cases,
491 skipped, zero failures, plus 197 additional tests. Sampled probability
law, serial conditional logits, cache recovery, history isolation, and seeded
repeatability have local tests. This timing experiment is not a broader
quality evaluation.

## Default policy follow-up

The November 9 default enables DFlash 2 and tiled M5 kernels for eligible
Qwen3.8 27B Q4 text requests on macOS 26 or later with an Apple10 GPU, at least
64 GiB of memory, and the installed draft. Both thinking and non-thinking
requests use it. The sampling presets are unchanged, and there is no prose
classifier or adaptive routing gate. `MERERUN_Q35_DFLASH2=0` opts out.
Unsupported request types retain reference projections. The measurements
above remain pinned to their original binary and collection protocols.
