# Laguna S 2.1 min-p evaluation

This report records the July 27, 2026, Laguna S 2.1 min-p promotion gate on an
AC-powered Apple M4 Max with 128 GB unified memory, macOS 26.5.2, the official
`Laguna-S-2.1-NVFP4-mlx` target, and the official
`Laguna-S-2.1-DFlash` companion.

## Decision

Use `min-p 0.02` as the Laguna evaluation default. Keep `0` available as the
official Poolside control and do not change defaults for any managed model.

The candidate preserved output volume and lexical diversity, improved the
aggregate chat checks from 337/352 to 345/352, improved tool calls from 8/10 to
9/10, and retained HumanEval 3/3. The commonly suggested `0.05` cutoff produced
one more raw chat-case pass but the same check total, the same tool/code pass
counts, 6% fewer chat tokens, and slower chat decode. `0.10` shortened the code
slice by 23% and is not the richness-preserving choice.

## Quality and richness

All stochastic runs used temperature 1, top-p 1, top-k 20, automatic DFlash,
and identical fixtures. Chat contains two complete 40-case repetitions; tools
and code contain one complete run.

| min-p | Chat cases | Chat checks | Mean chat latency | Mean chat decode | Chat tokens | Tool calls | HumanEval | Code tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 74/80 | 337/352 | 1.904 s | 12.26 tok/s | 1,755 | 8/10 | 3/3 | 462 |
| 0.02 | 76/80 | 345/352 | 1.874 s | 13.63 tok/s | 1,829 | 9/10 | 3/3 | 462 |
| 0.05 | 77/80 | 345/352 | 2.011 s | 11.01 tok/s | 1,648 | 9/10 | 3/3 | 474 |
| 0.10 | 76/80 | 345/352 | 2.027 s | 11.60 tok/s | 1,741 | 8/10 | 3/3 | 354 |

The mean per-run lexical-diversity ratios were 0.541, 0.544, 0.581, and 0.554
respectively. The higher `0.05` ratio accompanies shorter output rather than a
larger vocabulary corpus. `0.02` is the only candidate that improved checks
while also increasing chat output volume.

## Fixed-length DFlash performance

The resident release benchmark generated exactly 96 tokens per sample, five
samples per mode, with mode order rotated. Stochastic target-only, forced
DFlash, and automatic routing use the same normalized min-p distribution.

| min-p | Target-only median | Forced DFlash median | Forced acceptance | Automatic median | Automatic acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 26.86 tok/s | 19.46 tok/s | 8.7% | 25.17 tok/s | 15.5% |
| 0.02 | 24.40 tok/s | 17.26 tok/s | 10.3% | 23.08 tok/s | 13.0% |
| 0.05 | 24.26 tok/s | 16.45 tok/s | 11.1% | 23.52 tok/s | 15.5% |

Every automatic sample triggered the acceptance-aware fallback. Min-p does not
turn stochastic DFlash into a speed win on this fixture; its benefit is quality
filtering. Automatic routing remains necessary to bound the speculative
regression. Greedy generation is unchanged and retains the separate,
high-acceptance DFlash acceleration result.

## Correctness boundary

Min-p is applied after top-k/top-p, filtered tokens receive exactly zero
categorical mass, and the remaining distribution is renormalized. Laguna
DFlash consumes that same distribution for draft sampling, target acceptance,
and positive-residual rejection recovery. Focused tests cover threshold
semantics, composition, greedy invariance, exact zero support, GPU/probability
sampler agreement, and DFlash rejection correction. The official checkpoint
contract/kernel suite passed 28/28 after adding the stop-on-EOS regression.

Stochastic target-only and speculative runs are distribution-equivalent, not
byte-identical draws. Byte fingerprints remain the exactness gate for greedy
fixtures; stochastic correctness is established by the shared normalized
distribution and rejection-correction tests.
