# Bonsai 2.5 27B native qualification

Tested on November 24, 2026 on an Apple M5 Max with 128 GB of unified memory
and macOS 27.2.1. The Swift debug CLI used MLX/Metal. The reference used Prism
ML's pinned loader with MLX 0.32.0 and MLX-VLM 0.6.3.
[Prism ML model card and pinned artifact](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit/tree/aa432fd0e17612f6684e728d4d3c42b6656bcabb)

## Checkpoint and inference

- A fresh managed pull marked `text-chat-bonsai-2-5-27b-2bit` installed.
  Both the target weights and the MTP v2 head matched the published SHA-256
  values.
- The CLI matched all 26 reference prompt token IDs for a greedy arithmetic
  request. Both runtimes answered `4`. The native and reference outputs for
  the two-stanza ferry poem also matched byte for byte.
- The native OpenAI-compatible API and reference loader both distinguished
  synthetic red and blue images. The native API returned a valid JSON object,
  streamed the arithmetic answer, and finished a forced function-call round
  trip using a synthetic tool result.
- All three OpenAI reasoning levels returned reasoning and the correct
  answer to the arithmetic word problem. Messages token counting produced
  54, 24, and 66 input tokens for `low`, `medium`, and `xhigh` respectively.
- The installed model retrieved `violet-marsh` from record zero in a
  9,140-token prompt with 354 intervening synthetic records. The CLI used
  a 12,288-token context, temperature 0, and thinking disabled.

The installed model passed 49 of 52 cases in the repository's
`mere-chat-slice` at temperature 0 and a 192-token output budget. The three
misses were `MereChat/7`, `MereChat/32`, and `MereChat/40`. It passed 9 of
10 tool-selection cases with the same settings. In `MereTool/6`, the selected
tool's command argument omitted the expected export command.

## MTP v2

Each row uses one warm-up per route and two measured repetitions. Throughput
is the mean decode rate in tokens per second. Acceptance counts are per
repetition.

| Greedy prompt | Prompt tokens | Generated tokens | Target only | MTP v2 | Accepted proposals | Exact output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Two-stanza ferry poem | 29 | 71 | 15.29 | 16.54 | 27/45 | Yes |
| Python interval merge | 47 | 96 | 13.23 | 16.20 | 47/49 | Yes |

The two MTP code trials varied from 11.57 to 20.82 tokens per second during the
repository test run.

With temperature 1.0, top-p 0.95, top-k 20, and seed 42, both routes
reproduced their own output across two repetitions. MTP accepted 22 of 32
proposals and decoded at 8.59 tokens per second. The target-only route
decoded at 13.22 tokens per second. MTP v2 is opt-in and uses one proposal
per round by default.

```bash
mere.run model pull text-chat-bonsai-2-5-27b-2bit
mere.run model benchmark q36-mtp --model text-chat-bonsai-2-5-27b-2bit \
  --prompt "Write the first two stanzas of an original poem about a ferry arriving at dawn." \
  --variants baseline,forced --temperature 0 --decode-tokens 72 --stop-on-eos \
  --context-size 8192 --warmups 1 --warmup-tokens 16 --repetitions 2 --json
```

The run covers bounded cases on one Apple Silicon configuration. Maximum
context, other hardware, and Prism ML's published benchmark suites remain
unverified.
