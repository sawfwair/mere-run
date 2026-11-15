# SOV distributed expert reduction — 2026-11-14

Compared with the current custom kernel, resident latency improves 3.05% on
the short fixture and 1.38% on the wrapped fixture. The long fixture has no
reliable resident change, though generation improves 1.49%. Against local MLX,
the long fixture improves 7.05% in resident latency and 10.82% in generation.

The candidate distributes expert reductions and output stores across SIMD
lanes. It preserves sampled FP32 reduction results, every checked intermediate
array, and all **160 measured outputs and DFlash acceptance sequences**.
This qualification used an opt-in kernel. On 2026-11-15,
[automatic selection](./sov-transcribe-automatic-reduction-2026-11-15.md) was added
for the qualified M5 Max configuration. Other devices retain the existing
reduction unless explicitly overridden.

The [JSON receipt](./sov-transcribe-distributed-reduction-2026-11-14.json) includes
raw native reports, all prototype screens, bootstrap statistics, frozen-reference
hashes, source and binary hashes, and dependency state.

## Native comparison with the current layout

- 3.52-second recording: an improvement (3.05% paired latency reduction)
- 46.57-second recording: no reliable change (0.34% paired latency reduction)
- 93.14-second recording: an improvement (1.38% paired latency reduction)

Both modes share one resident target and draft, saved affine Q8 group-64
weights, optimized attention, rollback DFlash, and full native audio padding.
Activation fusion is disabled in both modes. Only the expert reduction and
stores change. Each recording uses two fresh processes with reversed recording
and starting-mode order. Two warmups per mode precede alternating adjacent
inference pairs.

| Audio | Current median | Candidate median | Paired resident reduction (95% CI) | Paired generation reduction (95% CI) | Resident wins/pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.52 s | 0.5780 s | 0.5645 s | 3.05% (1.07 to 4.71%) | 3.69% (1.77 to 7.84%) | 12/16 |
| 46.57 s | 1.7390 s | 1.7137 s | 0.34% (-0.78 to 1.71%) | 1.49% (0.58 to 3.10%) | 14/24 |
| 93.14 s | 3.1951 s | 3.1465 s | 1.38% (0.84 to 2.64%) | 1.44% (0.55 to 2.13%) | 13/16 |

Positive percentages mean lower latency. Percentages are medians of adjacent
pair ratios, not ratios of separate medians. The intervals use 10,000 bootstrap
resamples of pairs within each process (seed 94151). Resident times include
features, encoder/projector, prefill, generation, synchronization and cleanup;
loading, audio I/O and admission queueing are excluded. This is a shared machine,
so paired comparisons are more useful than comparing absolute times with older
cohorts. These measurements do not establish an exclusive-host maximum.

## Comparison with local MLX

A further two processes compare local MLX Q8 expert dispatch with the candidate
on the 46.57-second recording. Attention, rollback, quantized values, and all
other settings are the same.

Resident medians are 1.7712 s for local MLX and 1.6416 s for the candidate. Median paired resident reduction is **7.05%** (95% CI 5.64 to 8.19%), with 24/24 wins. Generation reduction is 10.82% (9.94 to 11.14%).

This is the local MLX build, not pristine upstream: it retains the pre-existing
generation-17 affine-Q8 NAX dispatch patch. Its source hash, patch and revisions
are recorded. Both modes use the same binary and installed metallib.

## Kernel and parity

The existing shader reduces each of 16 outputs separately and stores them from
lane zero. The candidate progressively distributes output ownership across
lanes while reducing the same partial sums. Its XOR shuffle distances are
1, 2, 4, 8, and 16. The first four steps halve the number of live output values
per lane; the last combines the remaining pair. Sixteen lanes then store one
output each. The per-output affine products, FP32 accumulation, route pairing,
and BF16 output rounding are unchanged.

A standalone Metal diagnostic compares 65,536 FP32 reduction outputs over
2,097,152 finite inputs with native `simd_sum`: zero differing bit patterns.
It includes wide exponents, signed zeros, and cancellation. This is sampled
numerical evidence on the qualified hardware, not an exhaustive proof for all
FP32 combinations or other GPU implementations. Alternative shuffle orders
change 1–18 values in the captured final block and are rejected.

Captured layer-zero gate/up, activation, down and mixed arrays contain 180,224,
90,112, 360,448 and 45,056 BF16 values, respectively. All match bit for bit.
Native GPU tests additionally cover both unpackers, both layouts, K=704 tails,
full projection widths, odd and repeated routes, parameter replacement, and
matrix/width fallbacks. Four-column projections retain the existing kernel.
Sixteen retained policy identities prevent compiled MoE graphs from reusing a
different dispatch after option switches.

All 32 short, 96 long (including the MLX comparison), and 32 wrapped native
outputs match frozen token IDs and acceptance lengths. The 93.14-second fixture
crosses the draft context window. This qualifies the recorded fixtures, not
arbitrary audio or a new accuracy corpus.

## Isolated screens and rejected approaches

The first single-block screen reports a 6.53% paired reduction. A diagnostic
that also returns intermediate arrays reports only 0.36%, illustrating why a
single isolated figure is insufficient. A longer dependency chain of eight
projection/activation blocks (excluding route mixing, with a residual to keep
inputs active) improves by 5.64% under ordinary scheduling (95% CI 4.98–7.43%,
90/100 wins), and 4.74% under compiled scheduling (3.73–5.93%, 93/100 wins).
Both chains preserve arrays. These screens are not end-to-end model timings;
the native comparisons above determine the practical result.

Other exact-parity prototypes were screened before integration:

- Stable GPU route planning, direct input mapping, and reordered/weighted output
  stores preserve arrays but provide only small, inconsistent isolated changes.
- Compact scheduling finds 79 actual route pairs among 128 route positions.
  Persistent workgroups preserve arrays, but reducing worker count regresses;
  the largest tested pool gives only a small isolated gain. It is not integrated.
- Forced vector loads and loop unrolling preserve arrays but regress.
- Moving output columns outside the K loop lowers register use but adds input
  work and is substantially slower. Caching row sums does not rescue it.

No new hardware-counter replay attributes changes to a cache, bandwidth, or
occupancy effect. The implementation and numerical checks establish the changed
work; native pairing measures its practical effect. No vocabulary, precision,
weight-packing, checkpoint-format, or encoder change is made in this pass.

## Reproduce and validation

```bash
.build/release/mere.run model benchmark sov /path/to/audio.wav \
  --model-root /path/to/original-sov --draft-model /path/to/dflash \
  --packed-weights /path/to/q8-g64 --experimental-decoder q8 \
  --experimental-attention --generation-experiment rollback \
  --interleave-metal --metal-baseline layout --experimental-expert-reduction \
  --warmups 2 --repetitions 12 --json
```

Add `--metal-first` to reverse starting mode. Omit `--metal-baseline layout` to
compare with local MLX. Standalone use requires `--experimental-decoder q8-metal
--experimental-expert-reduction`; the generator initializer also accepts
`experimentalExpertReduction: true`. These flags now force the policy; the
qualified M5 Max configuration selects it automatically without them.

Release build, strict lint without its cache, Python syntax checks, and
`git diff --check` pass. The focused GPU/decoder/CLI/documentation selection
executes 41 tests: 40 pass, one optional residency test skips, zero failures.
The full repository gate remains deferred at the user's explicit request.

The machine is an Apple M5 Max with 40 GPU cores and 128 GiB memory, macOS 27.2
(26B123), and Xcode 27.2. Research shaders, scripts, raw reports, captured inputs,
source snapshots, logs, frozen references and a verified hash manifest are
archived on the external SSD at
`/Volumes/SALVATION/MereRunProfiles/sov-distributed-reduction-2026-11-14/`.
Existing packed weights load directly; no new checkpoint copies are needed.
