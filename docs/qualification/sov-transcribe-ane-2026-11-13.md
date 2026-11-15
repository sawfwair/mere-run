# SOV-Transcribe-1 ANE qualification

The [optimization follow-up](./sov-transcribe-ane-optimization-2026-11-13.md)
records subsequent layout changes and new measurements. This page retains the
initial qualified conversion and its raw receipts.

Measured on 2026-11-13 with an Apple M5 Max, 128 GiB unified memory,
macOS 27.2 (26B123), and Xcode 27.2 (27B52).

The native encoder/projector runs on ANE and passes the numerical and token
checks below. Resident inference is slower than MLX on this machine. The
startup-inclusive CLI measurements have a lower median with ANE, with
overlapping ranges. These are separate measurements with different costs.

The [JSON receipt](./sov-transcribe-ane-2026-11-13.json) contains individual
runs, tokens, DFlash counts, source revisions, artifact hashes, and code hashes.

## Resident inference

Both providers use the original BF16 Gemma target, FP32 router, the pinned
DFlash companion, the default verbatim prompt, and a 448-token budget. Each
process loads its models once, performs one unmeasured warmup, and measures
three repetitions. No Instruments recording or repository checks ran during
these measurements.

All table values after the recording duration are seconds. Encoder/projector
time includes the Core ML input/output copies and MLX handoff. The complete
prepared pipeline includes features, prompt/prefill, generation, and cache
cleanup. Audio-file decoding, model load, process launch, and model unload are
outside this resident measurement.

| Recording | Provider | Encoder/projector | Prompt/prefill | Generation | Complete prepared pipeline |
| --- | --- | ---: | ---: | ---: | ---: |
| 3.51831 s | MLX | 0.059 | 0.258 | 0.063 | 0.412 |
| 3.51831 s | ANE + MLX | 0.423 | 0.262 | 0.062 | 0.781 |
| 46.57000 s | MLX | 0.109 | 0.490 | 1.471 | 2.158 |
| 46.57000 s | ANE + MLX | 0.851 | 0.494 | 1.461 | 2.880 |

The ANE pipeline is 1.89× slower for the short fixture and 1.33× slower for
the two-window fixture. Its encoder/projector is 7.22× and 7.81× slower,
respectively. MLX remains the default provider.

## Parity

Every measured run produced identical greedy tokens across the providers:
13 tokens for the short fixture and 138 for the long fixture. The installed
checkpoint regression also passed for the known short sentence.

The projected-output comparison checks finite values, relative RMSE at most
5%, and cosine similarity at least 0.999. These metrics compare unscaled
projected embeddings before the Core ML FP16 output is cast to the decoder's
BF16 precision.

| Recording | Relative RMSE | Cosine similarity | Exact token parity |
| --- | ---: | ---: | --- |
| Short | 3.687% | 0.999339 | Pass |
| Two windows | 2.280% | 0.999740 | Pass |

An independent PyTorch FP32 graph using the pinned original audio/projector
weights provides another numerical check. MLX BF16 differs from that reference
by 2.879% relative RMSE; Core ML FP16 differs by 3.581%. Four-head grouping
produces bit-identical Core ML outputs to the initial single-head graph and
reduces isolated warmed prediction time from 519 ms to 418 ms.

## Actual ANE execution

The selected graph's compute plan assigns all 2,223 nonconstant operations to
ANE. That describes anticipated placement. A separate trace of the native
Swift CLI records `encoder_main__Op0_AneInference Prediction` on the Apple
Neural Engine for **422.881 ms**, confirming actual hardware execution.

The [exported hardware intervals](./sov-transcribe-ane-2026-11-13-ane-events.xml)
include a **17.276 s** first-use ANE compilation interval. This cost is excluded
from the resident results and from the cache-warm CLI runs below. Full
Instruments traces and compiled artifacts remain under
`.build/qualification/sov/` in the qualification worktree.

## Complete CLI wall time

These measurements invoke `speech transcribe` in a fresh process each time,
including audio decoding, model loading, inference, model unload, and process
overhead. The filesystem, Metal, and Core ML caches are warm. There are three
runs per provider in MLX/ANE/ANE/MLX/MLX/ANE order. No system cache flush or
thermal control was applied.

| Provider | Median wall time | Range | Maximum reported peak memory footprint |
| --- | ---: | ---: | ---: |
| MLX | 33.74 s | 27.16–38.14 s | 99.30 GiB |
| ANE + MLX | 24.95 s | 24.69–33.56 s | 96.77 GiB |

All six invocations returned the expected transcript. The startup-inclusive
median favors ANE by 1.35× in this sample. The ranges overlap; these runs do not
isolate the cause of the first-inference difference or establish cold-cache
latency. The receipt retains maximum RSS separately from the operating system's
reported peak memory footprint. Lower memory capacities remain unqualified.

## Reproduce and validate

Follow the [runtime guide](../runtime/sov-transcribe.md#core-ml-encoder-and-projector)
to convert the pinned checkpoint, run `model benchmark sov`, compare the
projected tensors, and record a native Instruments trace. Conversion defaults
to groups of four attention heads; `--head-group-size 1` reproduces the initial
layout comparison. The compiled graph requires macOS 15 or newer.

The full repository gate (`./scripts/check.sh`) and the installed-checkpoint
cross-provider test passed. Gate coverage includes artifact integrity, routing,
CLI parsing, processor/config validation, and independent numerical fixtures.
The selected artifact also passed both full-checkpoint benchmark comparisons.

This is qualification on two English fixtures and one machine. Multilingual
accuracy, word error rates, energy use, and memory capacity limits remain
unqualified.
