# Flash-Next verification frontier on M4 Max

This receipt records the first Uzu-inspired verification-width experiment for
the Qwen3.8 Flash-Next 3-bit native-PLE checkpoint. It measures the optimized
target only. It does not measure a speculative tree decoder.

## Environment

- Commit: `1ef4fda327bd5cdbde803b1b3bdd07655f6bf22f`, plus the benchmark changes
  described in this receipt
- Machine: Apple M4 Max with 128 GiB unified memory
- OS: macOS 26.5.2, build 25F84
- Model: `vision-chat-q38-flash-next-3bit-native-ple`
- Model root: external volume
- Oracle length: 128 target-generated tokens
- Trials: two, with width order reversed for the second trial

## Target-only results

| Width | Verification passes | Mean verified tokens per second | Greedy parity |
| ---: | ---: | ---: | :---: |
| 1 | 128 | 20.83 | Yes |
| 4 | 32 | 50.36 | Yes |
| 8 | 16 | 79.52 | Yes |
| 16 | 8 | 132.85 | No |
| 32 | 4 | 189.03 | No |

Widths 4 and 8 preserve the target's serial greedy sequence. Widths 16 and 32
don't preserve it with the existing Flash-Next verification replay. Their rates
show arithmetic capacity, not usable generation throughput.

## End-to-end observations

The production four-token MTP path generated 256 tokens at 40.73 tokens per
second during a machine-admission-contended run. It drafted 224 tokens, accepted
181, and performed 75 verification passes. The acceptance rate was 80.8%.

Separate 128-token runs with block caps of 8, 16, and 32 all drafted 119 tokens,
accepted 102, and performed 26 verification passes. Their output hashes were
identical. The adaptive policy selected the same effective depth for every cap,
so raising the cap alone doesn't produce Uzu-style behavior.

## Decision

Keep the production Flash-Next block default at four. The exact width-8 target
rate establishes useful headroom, but the existing serial MTP drafter doesn't
convert that headroom into a measured end-to-end gain.

Do not use widths 16 or 32 in production. Reaching those budgets requires
tree-aware GDN, QSA, PLE, and hyperconnection verification that preserves the
serial target state. A follow-up prototype must pass greedy parity before any
speed result counts.

The next bounded experiment is an exact width-8 branch verifier with an
adaptive tree planner. Promote that work only if it beats the four-token path
on code, math, and prose prompts after draft cost is included.

## Exact-width follow-up

The follow-up prototype makes the checkpoint's full greedy trajectory exact at
widths 4, 8, 16, and 32. It combines row-accurate QSA replay with exact affine
Q4 projection kernels and an exact gathered Q3 expert-route kernel. The wide
Q4 policy is limited to the checkpoint shapes that affect greedy parity; the
remaining wide projections use the native MLX path.

The canonical release CLI ran three trials per width with no diagnostic
environment overrides:

```console
mere.run model benchmark q38-verification \
  --model-root <checkpoint> \
  --widths 4,8,16,32 \
  --tokens 128 \
  --trials 3 \
  --json
```

| Width | Verified tokens per second | Greedy parity | Previous target |
| ---: | ---: | :---: | ---: |
| 4 | 70.59-75.70 | Yes, 3/3 | 49.7-51.0 exact |
| 8 | 99.19-104.47 | Yes, 3/3 | 77.6-81.4 exact |
| 16 | 145.43-159.58 | Yes, 3/3 | 125-141 nonexact |
| 32 | 160.57-187.68 | Yes, 3/3 | 187-191 nonexact |

The mixed-width run exposed host/run-order variance at width 32. A separate
five-trial width-32 readback measured 186.98-189.00 verified tokens per second,
with a mean of 188.23 and greedy parity in all five trials. Four trials were in
the 187-191 target band; the remaining trial was 0.017 tokens per second below
its lower edge.

The release receipt is
`/tmp/q38-final-exact-rows-default-3trials.json` (SHA-256
`5e9e46f97f36bad5b28ed774d0448e2cf42fdecefcb5cf06102426e45362a16c`).
The isolated width-32 receipt is
`/tmp/q38-final-width32-stability-5trials.json` (SHA-256
`fcc139afa766e728af84890cb995523be5df604200ebcaa9b523c50ebc8c8274`).

## Quality guardrail

The Lite suite ran before the implementation, after the first exact-wide
version, and after the final kernels. Each run completed 24 case-trials: 17
were evaluable and seven required unavailable external fixtures.

| Build | Mean score | Passes |
| --- | ---: | ---: |
| Baseline | 0.9235 | 12/17 |
| Intermediate exact-wide | 0.9471 | 13/17 |
| Final exact-wide | 0.9256 | 11/17 |

These are single sampled trials, so the pass-count spread is not evidence of
an improvement or regression. The final mean remains within 0.0021 of the
baseline, while the deterministic full-checkpoint benchmark proves identical
greedy output at every measured width. The final Lite report is
`/tmp/q38-lite-final-exact-rows.report.json` (SHA-256
`90b8008d26f88809573551eab6ce74070840884006308ffd03f9ef52ee33e710`).

This remains a target-only verifier result. It does not qualify a production
tree decoder or claim end-to-end generation throughput; draft cost, acceptance,
adaptive planning, and workload-level quality still require separate evidence.
