# SOV expert layout and vocabulary kernel experiments — 2026-11-14

The retained change improves resident latency by **3.14% on the 46.57-second
recording** and **1.55% on the 93.14-second recording**, relative to the previous
byte-unpack layout. Generation improves by 4.75% and 3.71%, respectively. The
short recording has no reliable improvement.

The retained expert kernel uses eight output columns per SIMD group and
dispatches routes before column tiles. Weights, arithmetic and checkpoint format
are unchanged. Native comparisons check exact token IDs and DFlash acceptance
counts against frozen recordings. All **160 measured outputs** match.

The [JSON receipt](./sov-transcribe-expert-layout-2026-11-14.json) contains raw native reports, bootstrap
samples, microbenchmarks, source and binary hashes, and dependency state.

## Comparison with the previous fast path

The control is the prior four-column byte-unpack expert kernel. Both modes share
one resident target and draft, saved Q8 group-64 weights, optimized attention,
rollback KV storage, and native full audio padding. Two warmups per mode precede
measurement. Adjacent inferences alternate modes and reverse order on alternate
cycles. Two fresh processes per recording reverse recording and starting-mode
order. Neither mode repacks or replaces parameters when switching.

| Audio | Previous median | Optimized median | Paired resident reduction (95% CI) | Paired generation reduction | Resident wins/pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.52 s | 0.3817 s | 0.3809 s | -0.10% (-2.24 to 1.77%) | 4.27% | 8/16 |
| 46.57 s | 1.4932 s | 1.4603 s | 3.14% (1.48 to 4.23%) | 4.75% | 21/24 |
| 93.14 s | 2.6770 s | 2.6712 s | 1.55% (0.55 to 2.21%) | 3.71% | 12/16 |

Percentages are medians of adjacent-pair ratios, rather than ratios of the
separate latency medians. Confidence intervals bootstrap pairs within each
process. Resident times include features, encoder/projector, prefill, generation,
synchronization and cleanup; they exclude model loading, audio I/O and admission
queueing. The machine is shared with other work, so pairing limits drift without
establishing an exclusive-host maximum.

A further 24 pairs compare local MLX Q8 expert dispatch with the new kernel on
the 46.57-second recording. Resident medians are **1.4718 s
and 1.3967 s**; the median paired reduction is
**4.51%**, with a 95% bootstrap interval of
3.78 to 5.36%. Generation improves by 7.68%.
The custom path wins 22/24 resident pairs. Both modes use the same
Q8 values, attention policy and rollback DFlash policy.

This MLX comparison is **not a pristine upstream build**: the local checkout
already contains a 15-line `quantized.cpp` patch enabling affine Q8 NAX dispatch
on generation 17. This pass preserves that change; its hash and diff are
recorded in the research artifacts. Both comparison modes use the same binary.

## Why this layout

The previous shader computed four adjacent output columns per SIMD group.
The selected specialization computes eight where the output width is divisible
by sixteen, retaining four columns for narrower supported widths. It keeps two
routes sharing each weight load and two SIMD groups per workgroup.

Route-first dispatch places workgroups using the same column tile near each
other across sorted routes. This is intended to improve weight-cache locality.
No new post-change hardware-counter replay attributes the measured gain to a
specific cache. The shader's per-output products, affine group sums, SIMD
reduction and BF16 rounding are unchanged.

On captured layer-zero verification inputs, widening alone reduces paired
expert-block time by 6.65% in the first screen. Route-first scheduling reaches
9.31% in a later screen, with bit-identical arrays. Separate projection screens
show 2.04% for gate/up and 13.77% for down. Those forced-barrier stage measurements
are not additive end-to-end budgets. Wider tiles of 16 or 32 columns, and larger
route-reuse groups, regress in this screen. Metadata interleaving and inline
records preserve arrays but do not provide a useful gain, so no packed format
or loader change is needed.

## Vocabulary results

Several approaches were tested without changing weights or vocabulary coverage:

- Thin neural-accelerator tiles: the 16-row case can preserve all logits but
  is neutral. The 15-row alternatives change 68 of 3,932,160 logits on the
  captured input, despite preserving sampled top-1 IDs. The 15-row screen
  takes the first fifteen captured target hidden rows; it is not a native
  draft-stage timing. Those variants fail
  the full-logit parity screen and are rejected.
- Transposed matrix dispatch changes 125 logits for 15 rows and 148 for 16.
- Wider vocabulary weight reuse preserves logits, but eight- and fifteen-row
  register tiles are slower. A five-row variant with fused BF16 softcap gives
  only a small isolated head improvement and is not native-qualified.
- Fused tile-local greedy reduction preserves sampled IDs but is 12–63% slower.
- Compiled softcapping matches all 65,280 finite BF16 bit patterns. It reduces
  the pointwise stage by about 11%, roughly 22 microseconds, but full 15- and
  16-row head improvements are only 0.18–0.27% in this screen.

No vocabulary change is shipped from these experiments. A tested fusion of
expert gate/up projection and GELU also fails array parity, with maximum final
block error 0.015625. This rejects that prototype, not every possible fusion.

## Reproduce the native comparison

```bash
.build/release/mere.run model benchmark sov /path/to/audio.wav \
  --model-root /path/to/original-sov --draft-model /path/to/dflash \
  --packed-weights /path/to/q8-g64 --experimental-decoder q8 \
  --experimental-attention --generation-experiment rollback \
  --interleave-metal --metal-baseline byte \
  --warmups 2 --repetitions 12 --json
```

Add `--metal-first` to reverse the starting order. Omit `--metal-baseline byte`
to use local MLX Q8 as the control. `--metal-baseline reuse` selects the older
shift-unpack, four-column layout. The Python expert screen accepts
`--output-columns` and `--reference-layout` for isolated tiling studies.

## Validation and scope

The release build and focused GPU/decoder/CLI/documentation tests pass:
37 selected tests, zero failures, one opt-in native-residency receipt test
skipped. Synthetic GPU cases check both layouts and unpackers against MLX,
including 1,408-output projections, expert boundaries and K tails. Tests also
cover parameter replacement, matrix fallback, and stable compiled-policy
identities. Retaining one identity for each of four specializations prevents
compiled MoE graphs from silently reusing a different layout after option
switches. Existing packed checkpoints load directly.

Strict lint without its cache, Python syntax and `git diff --check` pass.
The full repository gate remains deferred at the user's request. This pass
adds no precision tradeoff and no new runtime model or weight copies. Research shaders,
scripts and raw measurements are archived on the external SSD under
`/Volumes/SALVATION/MereRunProfiles/sov-kernel-layout-2026-11-14/`.

The earlier [shader profile](./sov-transcribe-shader-profile-2026-11-14.md)
identified expert work as a major GPU cost. These experiments improve that
work; they do not establish a hardware limit or a reliable vocabulary-kernel
speedup.
