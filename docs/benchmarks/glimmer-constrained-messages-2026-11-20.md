# Glimmer Q8 constrained Messages qualification

Native Glimmer Q8 passed the known-answer constrained JSON suite and later
strict tool, context-edit, and document repeats. Qualification is limited to these requests and this artifact.
The record retains budget failures, an earlier wrong answer, and the blocked
fresh desktop attempt. It does not establish general model accuracy.

The [machine-readable results](./glimmer-constrained-messages-2026-11-20-summary.json)
separate binary revisions, expected rejections, model outcomes, and timing.

## Artifact and configuration

- Hardware: Apple M5 Max, 128 GiB unified memory.
- Artifact identity: `q8-symmetric-bf16-g64`, with affine Q8 groups of 64 and
  retained BF16 tensors. All nine shards, the index, and configuration match the
  established conversion receipt. The audit found 1,856 BF16 tensors and
  420 packed U32 tensors. This is the established Q8 candidate, not the managed
  model installation, which resolves a different BF16 artifact.
- Receipt SHA-256:
  `c8545755dce55ab11e63739ff23fd3acfd08b2132edc059cef14edc0a3da7b55`.
- Base: committed `main`, `ec42de1d`; feature parent: `7ac7fb09`.
- Server: native `mere.run api serve`, loopback, one active request, context
  limit 65,536. No HTTP inference proxy or llama.cpp server participates.
- Sampling: temperature 1.0, top-p 0.95, top-k 64, adaptive thinking, high effort.
  Version `2026-10-21` omits the prohibited sampling fields and uses the unpinned
  native recommended defaults. No greedy substitution is used.
- SDK: `anthropic==2.6.0`, with `jsonschema==4.26.0` for independent validation.
  Context-management streaming uses the SDK's beta accumulator.

Production ordinary requests retain DFlash with three draft tokens, deferred
verification, cost fallback, and the established `head4` Q8 verification setting.
Experimental fused norms remain off. Constrained requests use final-target
sampling. Weights, quantization, and model arithmetic are unchanged.

## Correctness and protocol

The opt-in qualifier checks parsed JSON against both the requested schema and
independent expected answers. A valid schema with a wrong answer fails. It also
compares actual prepared counts with generation usage, accumulates SDK streams,
checks local reasoning tokens, and verifies idle admission state after completion.

| Trial | Outcome |
| --- | --- |
| Initial paired known-answer and typed-tool suite | 26/26 constrained and 24/26 ordinary passed; the two ordinary failures exposed literal ATEM being treated as executable markup |
| Rebuilt constrained known-answer suite | 24/24 passed across both API versions, including Unicode, nested schemas, local references, and nullable values |
| Rebuilt ordinary literal-markup repeats | Both versions passed |
| Explicit two-call requests | Ordinary and strict both returned the two requested products in one turn |
| Unforced nested-markup copy | Failed: valid JSON contained HTML-escaped text instead of the requested literal value |
| Strict tool-error replay | Retried the failed call with matching prior IDs and a fresh returned call ID |
| Signed reasoning replay | Passed within a process and across server restarts; modified signatures returned 400 without model generation |
| Context clearing | Prepared input fell from 2,135 to 219 tokens; counting and generation agreed and reported both edits |
| PDF repeats | Eight of eight text/scan, ordinary/constrained, repeated trials returned `[42,1081]` in page order |
| Unsupported combinations | Explicit 400 responses for strict search, unsupported top-level tool unions, citations, stop sequences with constraints, and server compaction |
| Exhausted streaming budget | Error event, no successful terminal event or executable call; the next request recovered |

The first PDF probe returned `[-42,1081]` despite reasoning toward `[42,1081]`.
That schema-valid semantic error remains in the initial advanced results. The
eight later paired trials passed; they do not erase the earlier wrong answer or
establish statistical non-regression.

A difficult forced-string tool request exhausted 1,024 output tokens while
reasoning about JSON and XML escaping. The same request completed with a
4,096-token budget, using 3,929 tokens and about 259 seconds. Its arguments
preserved the newline, emoji, literal `&amp;`, and delimiter-looking nested JSON
exactly. This is a latency and budget limitation, not evidence of a short
reasoning-token budget. No incomplete call was published.

The stable SDK accumulator drops beta-only context-management metadata from
`message_delta`. The beta accumulator preserves it. The earlier stable-accumulator
assertion remains in the initial advanced report, separately from the passing
beta rerun.

A separate unforced-value copy check failed: the nested JSON field returned
`&lt;` and `&gt;` instead of literal angle brackets. Its top-level string was
preserved. This is a model argument-value error despite valid schema enforcement.
Additional explicit encoding guidance passed one nested-markup trial and a
literal-entity trial, but its nested-markup repeat exhausted 4,096 tokens. This
combination remains unqualified. The experimental guidance was not promoted
into the adapter, and values are never silently HTML-decoded to conceal errors.

## Cache and cancellation

Native checks observed real prefix writes, reads, zero-output warming, and expiry
more than five minutes after actual use. Ten additional cases passed one-hour
marker reuse, two-entry eviction, changed effort and schema identities,
context-edit invalidation, and subsequent-request recovery. The changed context
prefix counted 961 tokens before editing and 151 afterward. No simulated cache
hits or character-based token counts were used. Zero-output warming omits native phase instrumentation; its measured wall
time is retained, and zero-valued phase fields do not mean free prefill. Actual
one-hour expiration was not awaited; expiration boundaries also have deterministic unit coverage.

Signature validation can acquire an admission lease before rejecting a request.
An exploratory harness incorrectly asserted that the admission total must stay
unchanged. The corrected check verifies unchanged native generation counters and
awaited lease release; the API correctly returned 400 in both attempts.

Buffered and streaming disconnect trials passed both during admission and after
native decoding had produced at least 32 progress updates. Each trial canceled
a queued request, canceled the active client, observed both counts return to
zero, and independently verified the next request returned `42`.

## Claude Code and desktop boundary

Claude Code 2.1.316 completed a real local coding session: it inspected the files,
reproduced four failing tests, changed multiplication from addition to a product,
exercised a missing-test error, recovered, and passed the tests. An independent
runner confirmed all four tests and unchanged test contents. The session used
eight native generation rounds and about 108 seconds of client wall time.
Background compilation makes that wall time unsuitable as an isolated benchmark.

Fresh Cua desktop qualification is blocked by a locked Mac. Cua independently
reported `screen_locked` and refused app input before the first calculation.
The controller stopped its client and closed only its new Calculator instance.
This attempt counts as an environment gap, not a task pass or model failure.
Neither Calculator product nor the website task is requalified by this run.

The reference Claude Code version, 13-tool MCP catalog/filter, and Cua Driver
0.38.0 were retained. The full historical launcher source was unavailable; the
qualification controller and independent verifier were rebuilt. A signed Chrome
158.0.8127.37 copy was prepared for isolated profiles with Playwright's
`channel: 'chrome'` and Chromium sandbox enabled. No website trial ran while the Mac was
locked. Personal browser profiles and permission policies were unchanged.

## Throughput

Dedicated Release runs used three counterbalanced repetitions per workload and
mode, with one excluded short warm-up per process. All 24 timed requests and both
warm-ups returned the correct answer. Builds, repository tests, and other owned
inference clients had finished. Prompts had no cache markers or cache reads.

| Workload | Ordinary DFlash prefill | Ordinary DFlash decode | Constrained prefill | Constrained decode |
| --- | ---: | ---: | ---: | ---: |
| 105–107 prompt tokens | 555.2 | 35.9 | 443.6 | 15.7 |
| 7,255–7,257 prompt tokens | 632.8 | 25.4 | 621.2 | 15.4 |

| Workload | Ordinary target-only prefill | Ordinary target-only decode | Constrained prefill | Constrained decode |
| --- | ---: | ---: | ---: | ---: |
| 105–107 prompt tokens | 552.6 | 16.8 | 380.0 | 15.9 |
| 7,255–7,257 prompt tokens | 627.3 | 16.3 | 628.8 | 15.5 |

All rates are actual native tokens per second. Decode counts include reasoning, visible text, and
native delimiters. The matched target-only control
indicates that most of the production gap comes from disabling DFlash, rather
than grammar enforcement. These measurements show feature cost; they are not a
new kernel speedup or a maximum-throughput claim.

Native timing separates load, prefill, decode, and time to first token. Observed
production startup took 9.63 seconds to load plus 40.62 seconds of warm-up; the
later target-only restart took 1.20 plus 20.00 seconds. Warm-up includes graph
compilation and materialization. Those starts occurred with different host-cache
states and are not a controlled cold-load comparison.

Low-power mode was off and sampled thermal state was fair. The driver was
sequential, with no queued requests at request boundaries; those observations
are not a direct queue-duration measurement. Per-request counts, output lengths,
timings, startup data, and hardware state remain in the machine-readable record.

## Runtime fixes and automated gate

The ordinary native parser now keeps ATEM examples in an explicit user channel
literal and recognizes channel headers at message boundaries. The constrained
sampler consumes exact ByteLevel/Fuse token bytes, retains incomplete UTF-8
scalars, and rejects invalid or unreachable byte prefixes. After 256 rejected
draws, an exact admissible mask retains the requested sampling controls.
Unicode-scalar capture offsets also preserve leading combining marks.

Split-boundary, invalid-byte, rare-token, literal-markup, and combining-mark
regressions pass. The original streaming fixture put an executable packet inside
a user-answer channel; it now uses a real tool-recipient packet and verifies that
one call remains executable. Separate tests require user examples to stay literal.

`./scripts/check.sh` passed with the official SDK enabled: 6,384 XCTest cases,
585 skipped, zero failures, and 199 Swift Testing cases. Four independent
qualifier-scoring tests also passed. The gate includes the existing OpenAI
contracts, actual HTTP routes, SDK accumulation, cancellation, connection-command,
and PDF tests. Skipped installed-checkpoint tests do not qualify other adapters.

To repeat the paired checkpoint suite against an already-running local server:

```sh
/path/to/sdk-environment/bin/python scripts/qualify-messages-constrained.py \
  --base-url http://127.0.0.1:18089 \
  --model q8-symmetric-bf16-g64 \
  --output /private/tmp/glimmer-messages-results
```

The command does not install models, start services, run client tools, or
modify Claude Code settings. Raw synthetic responses stay in the selected report
directory. The machine-readable record preserves earlier failures and source,
binary, fixture, and artifact identities.
