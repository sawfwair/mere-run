# Qwen3.8 27B DFlash 2 and greedy readback measurements

On an Apple M5 Max, experimental DFlash 2 decoded the code prompt at a median
77.0 tokens per second, compared with 67.3 for ordinary MTP and 33.2 for serial
text decoding. On prose, DFlash 2 fell to 23.0 TPS, below MTP's 38.2 and serial's
31.7. DFlash 2 remains opt-in because its acceptance and throughput vary by
workload.

Packing greedy MTP verification readbacks produced mixed results: code was
-2.8% and prose was
+4.0% relative to the previous
readback path. These short runs don't establish a consistent gain, so packed
MTP readbacks also remain opt-in. The default MTP policy is unchanged.

## Decode throughput

Each cell shows the median of three requests, followed by their minimum and
maximum. Every request generates exactly 256 tokens. Decode TPS excludes
prefill and model loading; request metrics are retained in the
[machine-readable summary](qwen38-dflash2-and-greedy-readback-2026-11-08-summary.json).

| Workload | Serial TPS (range) | MTP, previous readbacks | MTP, packed readbacks | DFlash 2 TPS (range) | DFlash 2 / ordinary MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 33.2 (32.9–33.8) | 67.3 (65.2–73.3) | 65.4 (62.2–67.8) | 77.0 (73.8–84.2) | 1.14× |
| Prose | 31.7 (31.1–31.8) | 38.2 (37.4–38.7) | 39.7 (37.7–40.6) | 23.0 (21.1–23.4) | 0.60× |

These are same-binary feature comparisons, not release-to-release speedups.
The measurements don't isolate individual fusion or scheduling changes inside
DFlash 2, and they don't compare directly with another runtime's published TPS.

## Acceptance, memory, and parity

| Workload | MTP draft acceptance | DFlash 2 draft acceptance | MTP peak RSS | DFlash 2 peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Code | 73.0% | 82.0% | 14.72 GiB | 17.94 GiB |
| Prose | 38.5% | 17.2% | 14.74 GiB | 17.94 GiB |

RSS is the collector's peak process resident memory sampled every two seconds;
it isn't a complete accounting of GPU allocations or unified memory. Low prose
draft acceptance explains the additional target verification work in DFlash 2.
The runtime doesn't select a draft based on the prompt text.

All 24 measured requests in the main comparison matched their workload's serial
output SHA-256 and generated-token count. This check covers the two tested
256-token greedy continuations. It doesn't establish parity for every prompt,
long context, sampled generation, or image input.

## Protocol and qualification

The host has an Apple M5 Max with a 40-core GPU and 128 GiB of unified memory.
The collector waited for low GPU utilization and normal memory pressure before
admitting each condition. The main comparison recorded no competing inference
processes or known rendering workers, maintained normal memory pressure, and
had no swap growth. Approximately 8 GiB of previously allocated swap remained
on this shared workstation; these aren't exclusive-machine measurements.

Each condition ran in a separate release CLI process, with one 256-token warm-up
and three measured requests. The context limit was 8,192 tokens, temperature
was zero, top-p was one, thinking display was off, EOS was ignored, and
`MLX_ENABLE_TF32=0`. Synchronized profiling was disabled. Code ran serial,
previous MTP, packed MTP, then DFlash 2. Prose used the reverse order. Exact
prompts and each request's timings, diagnostics, and output hash are in the
summary.

An additional packed-MTP code control measured 62.7 TPS (61.1–63.2). Another
inference process appeared during its paired previous-readback run. That
contended run is excluded, and the unpaired packed control is kept separate
from the main comparison. The reversed-order control therefore doesn't
establish a readback speedup.

The target and companions use immutable revisions:

| Component | Repository | Revision |
| --- | --- | --- |
| dflash2 | `incoai/Qwen3.8-27B-DFlash2` | `015e795645c74b1a0eeef3b570031fb62e769bc5` |
| mtp | `morgan/qwen38-27b-mtp-r20k-lr3-q4-g64-q2-rerank` | `fd4a99c590dd6e468c0e2a28168c235e32151a4b` |
| target | `EigenLabs/Qwen3.8-27B-4bit` | `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` |
| Vision metadata only | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The explicit text-only model assembly links the Q4 target, MTP head, and official
vision configuration. It omits vision weights and isn't qualified for image
requests. DFlash 2 loads the published BF16 draft and quantizes its linear
projections to affine Q4/group-64 in memory. Draft norms, convolution bases,
and selector codebooks retain BF16; checkpoint files stay unchanged.

The release binary was built from a modified worktree with
`swift build --build-system native -c release --product mere.run --disable-index-store`.
The native build system avoids the checkout's existing missing vendor dSYMs.
The summary records the source base revision, runtime source digest, and binary
SHA-256. Both MTP routes were selected explicitly with
`MERERUN_Q35_PACKED_GREEDY_VERIFICATION=0` or `1`. After measurement, feature
selection changed: an unset switch selects the previous route, and the internal
decoder accepts an explicit selection so tests exercise both paths. The
verification arithmetic is unchanged.

The local gate passes with the native SwiftPM build-system wrapper. Focused
GPU checks cover packed greedy selection, the actual MTP loop, draft geometry,
multi-layer projection quantization, and cache behavior. Checkpoint parity in
this report comes from the measured requests, separate from synthetic tests.

To reproduce the feature selections, see
[Qwen runtime configuration](../configuration.md#mererun-q35-packed-greedy-verification)
and [the generation collector](https://github.com/sawfwair/mere-run/blob/main/scripts/benchmark-q35-generation.py).
Use an installed target and companion assembly with `--model-root`; retain each
receipt and exclude samples with competing inference or elevated pressure.
