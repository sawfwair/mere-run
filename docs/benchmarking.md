# Benchmarking

Small, local, repeatable lanes that answer one question: on *this* machine, is
this model better than the alternative? Use them to compare installed models,
verify a runtime change, and catch a regression before you claim support for
something.

They are not public leaderboards and do not try to be. For command syntax, see
the generated [CLI Reference](./cli.md); this page is the decision guide for
picking a lane and reading what it tells you.

## Benchmark Map

| Goal | Command | What It Measures |
| --- | --- | --- |
| Grounded assistant behavior | `model benchmark chat` | Answers over provided evidence, abstention, formats, concise summaries, and local-action boundaries |
| Tool selection | `model benchmark tool-calls` | Parsed tool names and arguments for synthetic Mere-style tools |
| Tool continuation | `model benchmark tool-continuations` | Grounded final answers after completed Gemma 4 tool calls and repeated tool chains |
| Coding ability | `model benchmark code` | Generated Python against HumanEval tests in a local sandbox |
| Vision-language behavior | `model benchmark vlm` | Synthetic image-question fixtures, or external `lmms-eval` datasets |
| API serving throughput | `model benchmark api-workload` | End-to-end `/v1/chat/completions` request latency, TTFT, cache, and batching counters |
| Runtime microbenchmarks | `model benchmark gemma4-kv`, `gemma4-mtp`, `q36-mtp` | KV-cache and speculative-decode modes against real checkpoints |
| Fused model quality | `model benchmark fused` | Versioned 24-case Lite and 110-case Comprehensive chat/code/tools/long-context/vision suites, native sampled profiles, repeated trials, and calibration diagnostics |

For the model-comparison contract, provenance format, external fixture import,
and raw-versus-policy logprob semantics, see
[Mere fused model evaluation](benchmark-fused.md).

## Recommended Workflow

1. Pull the exact models you want to compare.
2. Dry-run the benchmark plan before loading models.
3. Run large models one at a time when memory pressure matters.
4. For quality, use the suite's pinned non-greedy native profile and repeated
   trials. Reserve deterministic settings for isolated runtime microbenchmarks.
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

For stochastic evaluation, the chat, tool-call, and code lanes expose
`--top-k` and `--min-p` in addition to temperature and top-p. A positive min-p
removes tokens below that fraction of the most likely token's probability and
is recorded in the result plan. Keep all controls but min-p fixed when
measuring its effect, and use repeated runs for quality or diversity claims.

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

### Laguna S 2.1 Evaluation

Laguna S 2.1 is available as the opt-in managed model
`text-chat-laguna-s-2-1`. Pull it once with
`mere.run model pull text-chat-laguna-s-2-1`; the
official DFlash companion is installed automatically. The chat, tool-call, and
code benchmarks resolve the managed checkpoints by default. `--laguna-path`
and `--laguna-dflash-path` remain available as explicit checkpoint overrides;
`--laguna-dflash-tokens` controls the proposal length and defaults to the
measured value of `12`. The Laguna benchmark default is `--min-p 0.02`,
selected by the
[M4 Max quality and richness gate](./benchmarks/laguna-min-p-m4-max.md).
Pass `--min-p 0` to reproduce Poolside's published control. Managed-model
defaults for models other than Laguna remain unchanged.
`--laguna-dflash-min-tokens` controls the output-budget router and defaults to
`32`. Requests below the threshold skip DFlash prompt-context projection and
decode. Routed requests fall back losslessly when acceptance is below `0.25`
after one speculative round or below `0.60` after two rounds. Reports count
routed, bypassed, and fallback requests. The companion's
setup/context-projection cost can outweigh speculative savings on short or
low-acceptance outputs, so compare the phase timings on the workload you
intend to run.

```bash
swift run mere.run model benchmark chat \
  --laguna-dflash-tokens 12 \
  --laguna-dflash-min-tokens 32 \
  --min-p 0.02 \
  --cases MereChat/0,MereChat/3 \
  --log-responses
```

Use the resident-process crossover command for timing decisions. It loads the
target and draft once, rotates target-only, forced DFlash, and automatic order,
requires exact decode lengths, and records MLX active, cache, and peak memory:

```bash
swift build -c release
.build/release/mere.run model benchmark laguna-dflash \
  --laguna-path /path/to/Laguna-S-2.1-NVFP4-mlx \
  --laguna-dflash-path /path/to/Laguna-S-2.1-DFlash \
  --fixture code-completion \
  --decode-token-values 32,48,64,96 \
  --repetitions 3 \
  --include-automatic \
  --temperature 1 \
  --top-p 1 \
  --top-k 20 \
  --min-p 0.02 \
  --json
```

Add `--concurrency-values 1,2,4 --mixed-fixtures` to run an unmeasured warmup
at each sorted concurrency level followed by resident target-only, forced
DFlash, and optional automatic groups. Mixed groups rotate deterministic prose,
grounded email, and code prompts while also rotating the requested decode
lengths. The report includes aggregate tokens per second, p50/p95 latency and
time to first token, per-row decode-throughput fairness, physical same- and
variable-position batch steps, maximum batch size, memory, DFlash acceptance,
and stable output fingerprints. `byte_exact_target_equivalent` is true only
when every DFlash or automatic row matches the deterministic target output and
repeated target rows remain internally consistent. The official NVFP4 target
can select different but coherent greedy continuations at different sequence
or batch shapes, so a false byte-exact result is a hard signal to inspect the
logged responses and workload quality checks, not by itself proof of semantic
corruption. Set `--warmup-repetitions 0` only when deliberately measuring cold
graph compilation.

The chat benchmark's `--concurrency` option runs cases in fixed-size waves.
Values above one enable Laguna's ragged target continuous-batching scheduler.
The DFlash batch kernel is retained for forced evaluation, but automatic
DFlash uses the acceptance-aware serial coordinator. Text and JSON reports
include physical batch-step counts, same- versus variable-position steps,
maximum observed batch size, DFlash acceptance, and target verification,
recovery, and fallback forward counts. Compare concurrency on identical
fixtures; fewer physical forwards do not by themselves establish a wall-time
speedup.

The Laguna target sorts routed-expert work by expert for prompt prefill and
multi-token DFlash verification when a forward contains at least 64 routes.
Single-token target decode remains unsorted. This improves NVFP4 grouped-matmul
locality without changing the checkpoint or public benchmark contract. Set
`MERERUN_LAGUNA_SORTED_MOE=0` to run the reference routing order during a
controlled comparison or rollback.

The measured M4 Max/macOS 26 BF16 NVFP4 prefill path additionally fuses sorted
gate/up projection and SwiGLU behind
`MERERUN_LAGUNA_FUSED_SORTED_NVFP4_MOE`. The matching expert-aligned down
projection is controlled independently by
`MERERUN_LAGUNA_FUSED_SORTED_NVFP4_DOWN`; weighting and reduction retain the
native operation order. Set either flag to `0` for a portable-path A/B or
rollback. `MERERUN_LAGUNA_FAST_SORTED_INVERSE=0` independently restores the
reference second route sort. The guarded kernels recognize both M4 Max
`applegpu_g16s` and M5 Max `applegpu_g17s` on macOS 26.

Graph-level prefill controls are independently reversible. Shared full and
sliding masks default on and can be disabled with
`MERERUN_LAGUNA_SHARED_ATTENTION_MASKS=0`. The default eight-layer evaluation
ladder can be changed or disabled with `MERERUN_LAGUNA_PREFILL_ASYNC_LADDER`.
Exact fused residual/RMSNorm and QK-norm/RoPE kernels default on only for M5
Max; use `MERERUN_LAGUNA_PREFILL_FUSED_RESIDUAL_RMSNORM` and
`MERERUN_LAGUNA_PREFILL_QK_NORM_ROPE` for explicit A/Bs on either architecture.

These flags are an evaluation boundary, not a pull or serving contract. Do not
infer catalog support from a successful local checkpoint run.

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

### Gemma 4 Tool Continuation

`model benchmark tool-continuations` runs two deterministic real-checkpoint
Gemma 4 histories: a typed tool call containing boolean, nested, and null JSON
arguments followed by its result, and a two-tool chain with preserved reasoning
and call ids. Both cases require a grounded final answer and reject another
tool call after the completed results.

```bash
mere.run model benchmark tool-continuations --log-responses
```

Use `--model-root` to validate a locally converted Gemma 4 directory without
installing or replacing the managed model.

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
