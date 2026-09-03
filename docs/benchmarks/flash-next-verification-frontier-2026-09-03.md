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
