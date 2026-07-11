# Benchmarking

`mere.run` has several benchmark lanes. They are intentionally small, local, and
repeatable. Use them to compare installed models on this machine, verify runtime
changes, and catch regressions before claiming support. They are not full public
leaderboards.

For command syntax, see the generated [CLI Reference](./cli.md). This page is
the decision guide for which benchmark to run and how to read the result.

## Benchmark Map

| Goal | Command | What It Measures |
| --- | --- | --- |
| Grounded assistant behavior | `model benchmark chat` | Answers over provided evidence, abstention, formats, concise summaries, and local-action boundaries |
| Tool selection | `model benchmark tool-calls` | Parsed tool names and arguments for synthetic Mere-style tools |
| Coding ability | `model benchmark code` | Generated Python against HumanEval tests in a local sandbox |
| Vision-language behavior | `model benchmark vlm` | Synthetic image-question fixtures, or external `lmms-eval` datasets |
| API serving throughput | `model benchmark api-workload` | End-to-end `/v1/chat/completions` request latency, TTFT, cache, and batching counters |
| Runtime microbenchmarks | `model benchmark gemma4-kv`, `gemma4-mtp`, `q36-mtp` | KV-cache and speculative-decode modes against real checkpoints |

## Recommended Workflow

1. Pull the exact models you want to compare.
2. Dry-run the benchmark plan before loading models.
3. Run large models one at a time when memory pressure matters.
4. Keep sampling deterministic unless you are explicitly measuring stochastic behavior.
5. Save JSON output for comparisons and keep the raw command with the result.
6. Treat capped generations, missing models, and skipped cases as separate from correctness.

Example:

```bash
swift run mere.run model pull text-chat-gemma4-12b-4bit
swift run mere.run model benchmark chat --dry-run --json
swift run mere.run model benchmark chat \
  --models text-chat-gemma4-12b-4bit \
  --json
```

The benchmark commands do not auto-pull models during scoring. Missing models are
reported as skipped so the comparison remains explicit.

## Quality Evals

### Chat

`model benchmark chat` runs a grounded behavior slice against local assistant
models. The default suite is `mere-chat-slice`, a set of 40 original fixtures
covering:

- sender-specific local email questions
- missing-evidence abstention with `NOT_IN_EVIDENCE`
- workspace ambiguity and workspace selection
- JSON extraction
- concise summaries from provided evidence
- local action boundaries
- date sorting and small arithmetic
- conflicting evidence
- avoiding fabricated links, secrets, and command output

The default sampling is deterministic: `--temperature 0 --top-p 1`. Scoring uses
deterministic checks such as required phrases, forbidden phrases, regexes, JSON
keys, and bullet counts.

Use a narrow slice while debugging:

```bash
swift run mere.run model benchmark chat \
  --models text-chat-lfm25-a1b-8bit,text-chat-gemma4-nano \
  --cases MereChat/0,MereChat/1,MereChat/3 \
  --log-responses
```

### Tool Calls

`model benchmark tool-calls` is separate from the chat benchmark on purpose. It
passes synthetic `ToolDefinition` schemas to the model and scores the parsed
`toolCalls` it returns. It does not execute shell commands, write files, mutate
local data, or invoke the real Mere command plane.

Use it when you need to know whether a model can pick the right local tool and
arguments:

```bash
swift run mere.run model benchmark tool-calls \
  --models text-chat-q36-nano,text-chat-gemma4-12b-4bit \
  --cases MereTool/0,MereTool/4 \
  --log-responses
```

### Code

`model benchmark code` runs a real functional-code eval slice. The default suite
is `humaneval-slice`, currently three public HumanEval tasks:

- `HumanEval/0`
- `HumanEval/3`
- `HumanEval/8`

Each task is prompted once, the generated Python is combined with the task tests,
and the candidate is executed locally. Because this runs generated code, the
command requires `--allow-code-execution` unless you are using `--dry-run`.

```bash
swift run mere.run model benchmark code \
  --allow-code-execution \
  --json
```

Without `--models`, the code benchmark uses the supported members of the default
comparison lane for the current machine. On 32 GB Macs that means
`text-agent-ornith-9b` and `text-code-north-mini`; `text-code-qwen3` joins the
default comparison on 64 GB and larger machines.

By default, `--sandbox auto` uses `sandbox-exec` on macOS and `bubblewrap` on
Linux when available. Use `--sandbox none` only for trusted local smokes where
timeout and temporary-directory hygiene are enough.

For larger slices, pass a decompressed official HumanEval JSONL:

```bash
curl -L https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o /tmp/HumanEval.jsonl.gz
gunzip -c /tmp/HumanEval.jsonl.gz > /tmp/HumanEval.jsonl
swift run mere.run model benchmark code \
  --humaneval-file /tmp/HumanEval.jsonl \
  --tasks HumanEval/0,HumanEval/1,HumanEval/2,HumanEval/3,HumanEval/4 \
  --models text-agent-ornith-35b \
  --allow-code-execution \
  --json
```

Reasoning-model output is split before scoring. Visible code is executed, while
captured `<think>...</think>` content is reported as reasoning metadata. A second
generated reasoning block is reported as `reasoning_reopened=true`; treat that as
a loop or phase-restart warning, not an automatic correctness failure.

### VLM

`model benchmark vlm` has two modes:

- a tiny synthetic suite written to a local fixture directory
- external dataset execution through `lmms-eval`

Use the synthetic suite for quick regression checks:

```bash
swift run mere.run model benchmark vlm --json
```

Use the external lane when you want existing dataset coverage. Start with
`--dry-run` so the command prints the exact `lmms-eval` invocation before it
starts servers or runs datasets:

```bash
swift run mere.run model benchmark vlm \
  --dataset textvqa-val \
  --lmms-eval-root ~/src/lmms-eval \
  --limit 20 \
  --dry-run
```

## Runtime Benchmarks

### API Workload

`model benchmark api-workload` measures the real local API serving path. Run it
against an already-started `mere.run api serve` process when you need request
latency, time to first token, prefix-cache evidence, or continuous-batching
counters.

```bash
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --port 11934

swift run mere.run model benchmark api-workload \
  --base-url http://127.0.0.1:11934/v1 \
  --model text-chat-gemma4-turbo \
  --json
```

For cache or batching work, compare runs against `/runtime/status` before and
after the benchmark instead of inferring behavior from prose.

### KV Cache And Speculative Decode

The microbenchmark commands are for runtime implementation work:

- `model benchmark gemma4-kv`
- `model benchmark gemma4-mtp`
- `model benchmark q36-mtp`

They run real checkpoint paths with fixed prompt and decode lengths so runtime
changes can be compared consistently. The built-in prompt fixtures are for
runtime comparison, not model-quality evaluation.

Generic affine-8 KV and Psi/GLM compressed MLA are quality-sensitive controls,
so their source-level structural/numerical checks are separate:

```bash
swift test --filter 'AffineQuantizedKVCacheTests|GLM47AccelerationTests'
```

Before promoting either path, run release-mode real-checkpoint A/B with a fixed
greedy prompt corpus and report output parity, resident memory, TTFT, prefill
tokens/s, and decode tokens/s. See the
[guarded acceleration audit](./internals/guarded-acceleration.md).

## Interpreting Results

Use these conventions when reporting benchmark outcomes:

- Say which machine, branch, model IDs, and command flags were used.
- Report missing or skipped models separately from failures.
- Do not compare a capped run against an uncapped run as if they were equivalent.
- Keep `--max-tokens`, `--temperature`, and `--top-p` visible in the result.
- For code, distinguish generation failure, sandbox execution failure, test
  failure, token cap, and reasoning-loop metadata.
- For API workload, include runtime status counters when making cache or batching
  claims.
- For large models, prefer one model per command invocation when memory pressure
  could affect fairness.

## Related References

- [CLI Reference](./cli.md)
- [Testing Guide](./testing.md)
- [Configuration](./configuration.md)
- [Text Runtime](./runtime/text.md)
- [Local API Server](./runtime/api-server.md)
