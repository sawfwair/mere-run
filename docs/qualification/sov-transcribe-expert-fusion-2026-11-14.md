# SOV exact expert activation fusion — 2026-11-14

The fused Q8 gate/up projection and activation preserves exact arrays and all
**112 measured transcripts and DFlash acceptance sequences**, but provides
**no reliable resident latency improvement** over the current optimized expert
layout. Fusion remains opt-in. Ordinary inference retains the existing kernel.

The [JSON receipt](./sov-transcribe-expert-fusion-2026-11-14.json) contains raw
native reports, microbenchmarks, paired statistics, source and binary hashes,
dependency state, and frozen-reference hashes.

## Native paired results

Both modes share one resident target and draft, saved affine Q8 group-64
weights, optimized attention, rollback KV storage, and full native audio
padding. The control uses the current eight-column, route-first expert layout.
The candidate fuses gate/up projection, approximate GELU, and multiplication;
down projection is unchanged. Neither mode repacks or replaces weights.

There are two fresh processes per recording, with recording order and starting
mode reversed. Each process warms both modes twice, then alternates adjacent
inference order. The table reports median adjacent-pair fractional latency
reductions. Positive percentages mean faster; negative percentages mean slower.

| Audio | Control resident median | Fusion resident median | Paired resident reduction (95% CI) | Paired generation reduction (95% CI) | Resident wins/pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.52 s | 0.4812 s | 0.4789 s | -0.04% (-1.21 to 1.74%) | -1.21% (-1.88 to 0.72%) | 7/16 |
| 46.57 s | 1.6883 s | 1.6788 s | -0.10% (-1.34 to 1.45%) | 0.95% (-0.96 to 1.44%) | 12/24 |
| 93.14 s | 3.2023 s | 3.1951 s | -0.02% (-1.86 to 1.75%) | 0.17% (-0.98 to 1.37%) | 8/16 |

Paired ratios can differ in sign from the ratio of separate medians. The
intervals use 10,000 bootstrap resamples within each process. None establishes
a resident or generation improvement. These are shared-machine measurements;
absolute times should not be compared with older benchmark cohorts to infer a
regression. Resident time includes feature extraction, encoder/projector,
prefill, generation, synchronization, and cleanup. Loading, audio I/O, and
admission queueing are excluded.

All 32 short, 48 long, and 32 wrapped outputs match frozen token IDs and
acceptance lengths. The wrapped fixture crosses the draft context window.
Parity here is relative to the existing qualified outputs and Q8 arithmetic;
it does not establish equivalence for every possible recording.

## Exact BF16 activation

The previous fusion prototype failed array parity because it retained FP32
activation intermediates. Native MLX rounds each operation in its approximate
GELU to BF16. The new kernel preserves those stages:

1. Round the gate/up projection results to BF16.
2. Compute BF16 `0.5 * gate` and `pow(gate, 3)`.
3. Round the cubic coefficient product, polynomial addition, and coefficient
   multiplication separately to BF16.
4. Use precise tanh, round its result and the addition of one to BF16.
5. Round the activation product, then multiply by the BF16 up projection.

This sequence follows the pinned local MLXNN activation and Metal primitive
sources. A native Swift export checks all **65,280 finite BF16 input patterns**:
zero differing output bit patterns. The export buffer contains 65,536 entries;
the 256 nonfinite input encodings are replaced with zero. Nonfinite inputs are
not qualified. Another 90,112 strided native GLU values match bit for bit.

The full projection kernel retains the existing affine Q8 products, FP32 group
sums and SIMD reduction order. Two adjacent routes share weight loads; each
SIMD group computes four gate/up output pairs. Eight lanes independently
perform the two-route activation work after reduction. This avoids serializing
all activations on one lane. Unsupported shapes, unsorted routes, dense prefill,
and biased projections retain the existing path. Current parameters are inputs
on every call. Eight retained policy identities distinguish compiled MoE graphs
across unpacking, layout, and fusion switches.

## Isolated screens and their limits

On captured layer-zero verification inputs, the selected two-route,
four-pair, two-SIMD-group variant preserves every activation and final mixed
expert-block bit. The uncompiled block screen improves by **2.46%** in median
paired time, from separate medians of 1.2566 ms to 1.2154 ms. A compiled whole
block screen improves by only **0.75%**. Larger tiles, single-route variants,
and cooperative staging are slower or provide smaller gains.

These isolated results do not survive as a reliable whole-model improvement.
Fusion saves an intermediate gate/up buffer and an activation dispatch while
retaining the same expert weight reads. The measured native result determines
the default policy. No vocabulary kernel, packed format, quantization, or
precision policy changes in this pass.

Two attempted native activation captures contain zero command buffers. They
are excluded from execution evidence; no generated shader inspection or
hardware-counter attribution is claimed from them. Numerical checks and the
microbenchmarks execute Metal kernels on the GPU. This pass does not establish
a hardware performance limit.

## Reproduce

```bash
.build/release/mere.run model benchmark sov /path/to/audio.wav \
  --model-root /path/to/original-sov --draft-model /path/to/dflash \
  --packed-weights /path/to/q8-g64 --experimental-decoder q8 \
  --experimental-attention --generation-experiment rollback \
  --interleave-metal --metal-baseline layout --experimental-expert-fusion \
  --warmups 2 --repetitions 12 --json
```

Add `--metal-first` to reverse starting order. Standalone fusion requires
`--experimental-decoder q8-metal --experimental-expert-fusion`. The public
generator initializer also accepts `experimentalExpertFusion: true`; its
default is false. The ordinary speech CLI gains no new setup requirement.

## Validation and artifacts

Release build, strict lint without its cache, and `git diff --check` pass.
The focused GPU, decoder, CLI, and documentation selection executes 40 tests:
39 pass, one optional residency test skips, zero failures. New tests check exact
activation bits across K tails, full projection widths, odd and repeated routes,
parameter replacement, fallback, policy identities, and CLI validation.
The full repository gate remains deferred at the user's explicit request.

The machine is an Apple M5 Max with 40 GPU cores and 128 GiB memory, running
macOS 27.2 (26B123) and Xcode 27.2. Both modes use the same local MLX build,
including the pre-existing generation-17 affine-Q8 NAX dispatch patch recorded
in the receipt. This patch is preserved. The comparison is between custom
kernel variants, not a new comparison against pristine upstream MLX.

Research scripts, shaders, native activation references, raw benchmark reports,
source snapshots, logs, and a verified hash manifest are stored on the external
SSD at `/Volumes/SALVATION/MereRunProfiles/sov-exact-fusion-2026-11-14/`.
Existing packed weights load directly; no new checkpoint copies are created.
