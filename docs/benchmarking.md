# Benchmarking

Use the `mere.run model benchmark` lanes to answer one question: on *this*
machine, is this model better than the alternative? They compare installed
models, verify runtime changes, and catch regressions before you claim
support.

They aren't public leaderboards and don't try to be. This page helps you pick a
lane and read its results. For every flag, see the generated
[CLI reference](./cli.md) or run `mere.run model benchmark <lane> --help`.
Completed runs are listed in [Benchmark reports](#benchmark-reports).

## Benchmark map

The following table maps each goal to the command that measures it.

| Goal | Command | What it measures |
| --- | --- | --- |
| Grounded assistant behavior | `model benchmark chat` | Answers over provided evidence, abstention, formats, concise summaries, and local-action boundaries |
| Tool selection | `model benchmark tool-calls` | Parsed tool names and arguments for synthetic Mere-style tools |
| Tool continuation | `model benchmark tool-continuations` | Grounded final answers after completed Gemma 4 tool calls and repeated tool chains |
| Coding ability | `model benchmark code` | Generated Python against HumanEval tests in a local sandbox |
| Vision-language behavior | `model benchmark vlm` | Synthetic image-question fixtures, or external `lmms-eval` datasets |
| API serving throughput | `model benchmark api-workload` | End-to-end `/v1/chat/completions` request latency, TTFT, cache, and batching counters |
| Runtime microbenchmarks | `model benchmark gemma4-kv`, `gemma4-mtp`, `q36-mtp`, `q38-verification` | KV-cache, speculative-decode, and target-verification modes against real checkpoints |
| Laguna DFlash timing | `model benchmark laguna-dflash` | Target-only, forced DFlash, and automatic routing in one resident process |
| Kolibri log probabilities | `model benchmark kolibri-logprobs` | Raw logits and log probabilities for fixed token sequences on a converted Kolibri checkpoint |
| Parakeet Core ML timing | `model benchmark parakeet-coreml` | Stage-level timings for a prepared Parakeet Core ML transcription pipeline |
| Fused model quality | `model benchmark fused`, `fused-fixture` | Versioned 24-case Lite and 110-case Comprehensive chat, code, tool, long-context, and vision suites, native sampled profiles, repeated trials, and calibration diagnostics |
| External or private evals | `eval pack validate`, `eval run`, `eval promote` | Content-addressed external cases, matched model, adapter, and prompt arms, logprob calibration, gates, and promotion receipts without importing pack content |

For the model-comparison contract, provenance format, external fixture import,
and raw-versus-policy logprob semantics, see
[Mere fused model evaluation](./benchmark-fused.md).

For cases, prompts, scorers, and qualification gates that must stay outside
this public repository, use the
[external evaluation-pack interface](./evaluation-packs.md).

## Recommended workflow

1. Pull the exact models you want to compare.
2. Dry-run the benchmark plan before loading models.
3. Run large models one at a time when memory pressure matters.
4. For quality, use the suite's pinned non-greedy native profile and repeated
   trials. Reserve deterministic settings for isolated runtime
   microbenchmarks.
5. Save JSON output for comparisons, and keep the raw command with the result.
6. Report capped generations, missing models, and skipped cases separately
   from correctness.

The following example pulls one model, previews the chat plan, and runs it:

```bash
mere.run model pull text-chat-gemma4-12b-4bit
mere.run model benchmark chat --dry-run --json
mere.run model benchmark chat \
  --models text-chat-gemma4-12b-4bit \
  --json
```

Benchmark commands don't pull models during scoring. They report missing
models as skipped, so the comparison stays explicit.

## Quality evaluations

### Chat

`model benchmark chat` runs a grounded behavior slice against local assistant
models. Without `--models`, it compares `text-chat-gemma4-12b-4bit`,
`text-chat-lfm25-a1b-8bit`, and `text-chat-gemma4-nano`. The default suite,
`mere-chat-slice`, contains 52 original fixtures, `MereChat/0` through
`MereChat/51`. They cover:

- Sender-specific local email questions
- Missing-evidence abstention with `NOT_IN_EVIDENCE`
- Workspace ambiguity and workspace selection
- JSON extraction and other structured output
- Concise summaries from provided evidence
- Local action, publish, dry-run, and destructive-action boundaries
- Date sorting, date reasoning, and small arithmetic
- Conflicting evidence and false premises
- Privacy boundaries and prompt-injection resistance
- Avoiding fabricated links, secrets, and command output
- Explanation, tone rewriting, planning, and tradeoff analysis

This standalone regression lane defaults to deterministic sampling:
`--temperature 0 --top-p 1`. That default doesn't apply to the fused quality
suite, which always uses each model family's pinned non-greedy native profile
and repeated trials. Scoring uses deterministic checks such as required
phrases, forbidden phrases, regular expressions, JSON keys, and bullet counts.

For stochastic evaluation, the chat, tool-call, and code lanes accept
`--top-k` and `--min-p` in addition to temperature and top-p. A positive min-p
removes tokens below that fraction of the most likely token's probability, and
the result plan records it. To measure the effect of min-p, keep every other
control fixed, and use repeated runs for quality or diversity claims.

To debug, run a narrow slice:

```bash
mere.run model benchmark chat \
  --models text-chat-lfm25-a1b-8bit,text-chat-gemma4-nano \
  --cases MereChat/0,MereChat/1,MereChat/3 \
  --log-responses
```

### Tool calls

`model benchmark tool-calls` is deliberately separate from the chat benchmark.
It passes synthetic `ToolDefinition` schemas to the model and scores the
parsed `toolCalls` it returns. It doesn't run shell commands, write files,
change local data, or call the real Mere command plane. Without `--models`, it
compares `text-chat-q36-nano` and `text-chat-gemma4-12b-4bit` on cases
`MereTool/0` through `MereTool/9`.

Use it to check whether a model picks the right local tool and arguments:

```bash
mere.run model benchmark tool-calls \
  --models text-chat-q36-nano,text-chat-gemma4-12b-4bit \
  --cases MereTool/0,MereTool/4 \
  --log-responses
```

### Code

`model benchmark code` runs a functional code-evaluation slice. The default
suite, `humaneval-slice`, contains three public HumanEval tasks:
`HumanEval/0`, `HumanEval/3`, and `HumanEval/8`.

The command prompts each task once, combines the generated Python with the
task tests, and runs the result locally. Because it runs generated code, the
command requires `--allow-code-execution` unless you pass `--dry-run`:

```bash
mere.run model benchmark code \
  --allow-code-execution \
  --json
```

Without `--models`, the code benchmark selects the members of its comparison
lane that this Mac supports: `text-agent-ornith-9b`, the
`text-agent-ornith-35b-mlx` variants, `text-code-north-mini`, and
`text-code-qwen3`. Run `--dry-run` to see the resolved list.

`--sandbox auto`, the default, uses `sandbox-exec` on macOS and `bubblewrap`
on Linux when available. You can also name `macos-sandbox-exec` or
`bubblewrap` explicitly. Use `--sandbox none` only for trusted local smoke
tests where the execution timeout and temporary-directory hygiene are enough.

For larger slices, download and decompress the official HumanEval JSONL, then
pass it with `--humaneval-file`:

```bash
curl -L https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o /tmp/HumanEval.jsonl.gz
gunzip -c /tmp/HumanEval.jsonl.gz > /tmp/HumanEval.jsonl
mere.run model benchmark code \
  --humaneval-file /tmp/HumanEval.jsonl \
  --tasks HumanEval/0,HumanEval/1,HumanEval/2,HumanEval/3,HumanEval/4 \
  --models text-agent-ornith-35b \
  --allow-code-execution \
  --json
```

The benchmark splits reasoning-model output before scoring. It runs the
visible code and reports captured `<think>...</think>` content as reasoning
metadata. `--thinking` turns on model thinking and disables the
HumanEval-specific stop sequences, which can fire inside the thinking block. A
second generated reasoning block is reported as `reasoning_reopened=true`.
Treat that flag as a loop or phase-restart warning, not an automatic
correctness failure.

### Laguna S 2.1 evaluation

Laguna S 2.1 is the opt-in managed model `text-chat-laguna-s-2-1`. Pull it
once with `mere.run model pull text-chat-laguna-s-2-1`; the pull also installs
the official DFlash companion. The chat, tool-call, and code benchmarks
resolve the managed checkpoints by default.

The following Laguna options apply to those three lanes:

- `--laguna-path` and `--laguna-dflash-path` override the managed target and
  companion with local checkpoint directories.
- `--laguna-dflash-tokens` sets the proposal length, from 1 through 15. The
  default is the measured value of `12`.
- `--laguna-dflash-min-tokens` sets the output-budget router threshold, `32`
  by default. Requests below the threshold skip DFlash prompt-context
  projection and decode.
- `--laguna-dflash-routing` (chat only) selects `automatic` (the default),
  `target-only`, or `dflash`.

The Laguna benchmark default is `--min-p 0.02`, selected by the
[Laguna S 2.1 min-p evaluation](./benchmarks/laguna-min-p-m4-max.md). Pass
`--min-p 0` to reproduce Poolside's published control. Defaults for other
managed models don't change.

Routed requests fall back losslessly when acceptance is below `0.25` after one
speculative round or below `0.60` after two rounds. Reports count routed,
bypassed, and fallback requests. On short or low-acceptance outputs, the
companion's setup and context-projection cost can outweigh speculative
savings, so compare phase timings on the workload you intend to run:

```bash
mere.run model benchmark chat \
  --laguna-dflash-tokens 12 \
  --laguna-dflash-min-tokens 32 \
  --min-p 0.02 \
  --cases MereChat/0,MereChat/3 \
  --log-responses
```

For timing decisions, use the resident-process crossover command,
`model benchmark laguna-dflash`. It loads the target and draft once, rotates
target-only, forced DFlash, and automatic order, requires exact decode
lengths, and records MLX active, cache, and peak memory. The built-in fixtures
are `deterministic-prose` (the default), `grounded-email`, and
`code-completion`. Run it from an optimized build:

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

To measure concurrency, add `--concurrency-values 1,2,4 --mixed-fixtures`.
The command then runs an unmeasured warmup at each sorted concurrency level,
followed by resident target-only, forced DFlash, and optional automatic
groups. Mixed groups rotate the prose, grounded-email, and code prompts along
with the requested decode lengths. The report includes:

- Aggregate tokens per second
- p50 and p95 latency and time to first token
- Per-row decode-throughput fairness
- Physical same-position and variable-position batch steps, and maximum batch
  size
- Memory and DFlash acceptance
- Stable output fingerprints

`byte_exact_target_equivalent` is true only when every DFlash or automatic row
matches the deterministic target output and repeated target rows stay
internally consistent. The official NVFP4 target can select different but
coherent greedy continuations at different sequence or batch shapes. A false
byte-exact result is a signal to inspect the logged responses and workload
quality checks; it isn't by itself proof of semantic corruption. Set
`--warmup-repetitions 0` only when you deliberately measure cold graph
compilation.

The chat benchmark's `--concurrency` option runs cases in fixed-size waves.
Values above one turn on Laguna's ragged target continuous-batching scheduler.
The DFlash batch kernel stays available for forced evaluation, but automatic
DFlash uses the acceptance-aware serial coordinator. Text and JSON reports
include physical batch-step counts, same-position versus variable-position
steps, maximum observed batch size, DFlash acceptance, and target
verification, recovery, and fallback forward counts. Compare concurrency on
identical fixtures; fewer physical forwards don't by themselves establish a
wall-time speedup.

#### Laguna prefill controls

The Laguna target sorts routed-expert work by expert for prompt prefill and
multi-token DFlash verification when a forward contains at least 64 routes.
Single-token target decode stays unsorted. Sorting improves NVFP4
grouped-matmul locality without changing the checkpoint or the public
benchmark contract. To run the reference routing order during a controlled
comparison or rollback, set `MERERUN_LAGUNA_SORTED_MOE=0`.

The following environment variables control the measured M4 Max and macOS 26
BF16 NVFP4 prefill path. Each one is independently reversible:

- `MERERUN_LAGUNA_FUSED_SORTED_NVFP4_MOE` fuses the sorted gate and up
  projection with SwiGLU.
- `MERERUN_LAGUNA_FUSED_SORTED_NVFP4_DOWN` controls the matching
  expert-aligned down projection. Weighting and reduction keep the native
  operation order.
- `MERERUN_LAGUNA_FAST_SORTED_INVERSE=0` restores the reference second route
  sort.
- `MERERUN_LAGUNA_SHARED_ATTENTION_MASKS=0` turns off the shared full and
  sliding masks, which are on by default.
- `MERERUN_LAGUNA_PREFILL_ASYNC_LADDER` changes or turns off the default
  eight-layer evaluation ladder.
- `MERERUN_LAGUNA_PREFILL_FUSED_RESIDUAL_RMSNORM` and
  `MERERUN_LAGUNA_PREFILL_QK_NORM_ROPE` control the exact fused residual and
  RMSNorm, and QK-norm and RoPE kernels. They default on only for M5 Max; set
  them explicitly for comparisons on either architecture.

Set either fused NVFP4 flag to `0` for a portable-path comparison or rollback.
The guarded kernels recognize both M4 Max `applegpu_g16s` and M5 Max
`applegpu_g17s` on macOS 26.

These flags are an evaluation boundary, not a pull or serving contract. Don't
infer catalog support from a successful local checkpoint run.

### VLM

`model benchmark vlm` has two modes:

- A small synthetic suite, `synthetic-vqa-v1`, written to a local fixture
  directory
- External dataset runs through `lmms-eval`: `mathvista-testmini`,
  `mmmu-val`, `chartqa`, `docvqa-val`, or `mme`, or a raw task list with
  `--lmms-tasks`

Use the synthetic suite for fast regression checks:

```bash
mere.run model benchmark vlm --json
```

To benchmark the recommended Ornith 4-bit vision lane directly:

```bash
mere.run model benchmark vlm \
  --models text-agent-ornith-35b-mlx-4bit \
  --json
```

Use an external dataset when you want existing dataset coverage. Start with
`--dry-run` so the command prints the exact `lmms-eval` invocation before it
starts servers or runs datasets:

```bash
mere.run model benchmark vlm \
  --dataset docvqa-val \
  --lmms-eval-root ~/src/lmms-eval \
  --limit 20 \
  --dry-run
```

By default, the command starts its own `mere.run api serve` processes on
loopback from port `11934`. To use an OpenAI-compatible server that is already
running, pass `--external-endpoint --base-url <url>`.

## Runtime benchmarks

### Gemma 4 tool continuation

`model benchmark tool-continuations` runs two deterministic real-checkpoint
Gemma 4 histories:

- A typed tool call with boolean, nested, and null JSON arguments, followed by
  its result
- A two-tool chain with preserved reasoning and call IDs

Both cases require a grounded final answer and reject another tool call after
the completed results. The default model is `text-chat-gemma4-12b-4bit`:

```bash
mere.run model benchmark tool-continuations --log-responses
```

To validate a locally converted Gemma 4 directory without installing or
replacing the managed model, pass `--model-root`.

### API workload

`model benchmark api-workload` measures the real local API serving path. Run
it against a `mere.run api serve` process that is already running when you need
request latency, time to first token, prefix-cache evidence, or
continuous-batching counters. The benchmark targets `--host` and `--port`,
which default to `127.0.0.1` and `8080`.

The following example starts a server on port 11934 in one terminal and
benchmarks it from another:

```bash
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --port 11934

mere.run model benchmark api-workload \
  --port 11934 \
  --model text-chat-gemma4-turbo \
  --json
```

Without `--workload-file`, the command replays a built-in workload of eight
turns over a shared stable prefix. To test batching, pass `--concurrency`
greater than one and start the server with `--max-active-requests` greater
than one. For cache or batching work, compare `/runtime/status` before and
after the benchmark instead of inferring behavior from prose.

### KV cache and speculative decode

The microbenchmark commands support runtime implementation work:

- `model benchmark gemma4-kv` compares Gemma 4 default KV cache decode against
  packed PolarKV.
- `model benchmark gemma4-mtp` compares Gemma 4 serial decode against verified
  MTP speculative decode.
- `model benchmark q36-mtp` compares Qwen-family serial decode against
  adaptive and forced MTP speculative decode.
- `model benchmark q38-verification` measures the Qwen-family target-only
  verification frontier.

They run real checkpoint paths with fixed prompt and decode lengths, so you
can compare runtime changes consistently. The built-in prompt fixtures are for
runtime comparison, not model-quality evaluation.

#### Qwen-family MTP

For repeated Qwen-family measurements, run `q36-mtp --repetitions 3
--warmups 2 --warmup-tokens 512 --decode-tokens 512`. Each variant keeps its
model loaded across warmup and measured requests. `--variants
baseline,adaptive` selects serial and default generation; reverse their order
in a separate run to check order effects. The JSON report keeps every measured
request. A single-variant run doesn't claim serial-output parity.

The `q36-mtp` lane warms each selected generator before timing it, with prefix
caching and continuous batching turned off. Warmup reduces startup costs; it
doesn't establish that later requests have stable scheduling. Keep every
repeat, and compare ranges as well as medians. The command ignores EOS to
measure fixed-length generation, so these fixtures aren't completion-quality
tests.

`scripts/benchmark-q35-generation.py --help` describes the code, chat, prose,
math, and summarization receipt collector. It records complete decode and
request throughput, memory pressure, swap changes, and competing `mere.run`
inference processes. Metadata commands such as `status` and `model list`
don't count as inference. The collector also checks for active HyperFrames GPU
workers and records the Apple GPU driver's utilization counter; it requires at
most 10 percent device utilization before starting. Use an optimized binary,
and keep unrelated inference and rendering idle throughout the measurement.
Temperature defaults to zero; for a separate sampled-generation measurement,
use `--temperature 0.7 --top-p 0.9`.

To measure under existing desktop graphics activity, add
`--allow-desktop-load`. The collector then accepts the observed load and marks
every receipt as `desktop-load` with `uncontended: false`. Keep the repetition
range visible, and treat the result as a workstation observation. The
collector still checks memory pressure and refuses to start alongside another
inference process.

Record the binary's build commit with `--build-source-revision`. The receipt
distinguishes that revision from the checkout's current head, captures CPU
load averages, and rejects a binary that was replaced during measurement.

Set `MERERUN_Q35_MTP_PROFILE=1` only for diagnosis. The optional
`acceleration.speculationProfile` contains per-round draft, target-forward,
acceptance, and repair timings, plus serial fallback time. Submission time can
include waits inside a model forward. Profiling adds synchronization, so turn
it off before you draw throughput conclusions.

For the scheduling defaults, controls, and measured workload results, see
[Qwen and Ornith generation tuning](./benchmarks/ornith-qwen-generation-tuning-2026-09-05.md).

#### Verification width

The `q38-verification` lane measures target-only linear blocks against
target-generated oracle tokens. Use it to find a useful verification width
before you implement a drafter or tree-aware recurrent kernels. Don't treat its
rate as end-to-end generation throughput.

The command accepts Flash-Next widths through 32, and Qwen3.8 27B or Ornith
1.5 Q4 widths through nine. Without `--widths`, Flash-Next uses
`1,4,8,16,32` and the other models use `1,4,8,9`. A successful width
measurement doesn't change production MTP depth.

For the first M4 Max result and the promotion decision, see the
[Flash-Next verification frontier on M4 Max](./benchmarks/flash-next-verification-frontier-2026-09-03.md).
For history and expert-kernel comparisons, see
[Flash-Next optimization transfer to Ornith and Qwen 27B](./benchmarks/ornith-qwen-flash-transfer-2026-09-05.md).

#### Quantized KV and compressed MLA

Generic affine-8 KV and Psi/GLM compressed MLA are quality-sensitive
controls, so their source-level structural and numerical checks run
separately:

```bash
swift test --filter 'AffineQuantizedKVCacheTests|GLM47AccelerationTests'
```

Before you promote either path, run a release-mode real-checkpoint comparison
with a fixed greedy prompt corpus. Report output parity, resident memory,
TTFT, prefill tokens per second, and decode tokens per second. For details,
see the [guarded acceleration audit](./internals/guarded-acceleration.md).

### Kolibri log probabilities

`model benchmark kolibri-logprobs` scores a JSON suite of fixed token
sequences on a converted native Kolibri checkpoint. It writes raw-logit files
and a receipt to a fresh `--output` directory. `--calibration-output` also
writes calibration-only expert input second moments to a fresh safetensors
file. For the conversion and comparison workflow, see
[Native Kolibri-1](./runtime/kolibri.md).

### Parakeet Core ML

`model benchmark parakeet-coreml` loads and verifies one Mere-built Parakeet
Core ML artifact, decodes the input audio once, runs unmeasured warmups, and
records stage-level timings without reloading the model. The defaults are two
warmups and five measured repetitions. Run it from an optimized build:

```bash
swift run -c release mere.run model benchmark parakeet-coreml ./short.wav \
  --artifact /path/to/parakeet-coreml \
  --json
```

For how to build the artifact, see the [Speech runtime](./runtime/speech.md).

## Interpret results

Use these conventions when you report benchmark outcomes:

- Say which machine, branch, model IDs, and command flags you used.
- Report missing or skipped models separately from failures.
- Don't compare a capped run against an uncapped run as if they were
  equivalent.
- Keep `--max-tokens`, `--temperature`, and `--top-p` visible in the result.
- For code, distinguish generation failure, sandbox execution failure, test
  failure, token cap, and reasoning-loop metadata.
- For API workload, include runtime status counters when you make cache or
  batching claims.
- For large models, prefer one model per command invocation when memory
  pressure could affect fairness.

## Benchmark reports

These pages record completed runs. Each one is a dated record: it keeps the
original hardware, binary, hashes, and limitations, and a newer runtime
doesn't rewrite it.

### Text, code, and decision models

- [Fused comprehensive reference runs](./benchmarks/fused-reference-runs.md):
  Qwen3.8 low reasoning, Laguna XS 2.1, and Nemotron Lightning on the
  Comprehensive fused suite, August 2026
- [Laguna S 2.1 min-p evaluation](./benchmarks/laguna-min-p-m4-max.md): the
  min-p promotion gate on M4 Max, July 27, 2026
- [Flash-Next verification frontier on M4 Max](./benchmarks/flash-next-verification-frontier-2026-09-03.md):
  2026-09-03
- [Flash-Next optimization transfer to Ornith and Qwen 27B](./benchmarks/ornith-qwen-flash-transfer-2026-09-05.md):
  2026-09-05
- [Qwen and Ornith generation tuning](./benchmarks/ornith-qwen-generation-tuning-2026-09-05.md):
  2026-09-05
- [Ornith Q4 distilled MTP promotion](./benchmarks/ornith-mtp-promotion-2026-09-05.md):
  2026-09-05
- [Ornith MTP upstream comparison](./benchmarks/ornith-mtp-upstream-check-2026-09-05.md):
  2026-09-05
- [Clef full-checkpoint checks](./benchmarks/clef-native-qualification-2026-10-01.md):
  2026-10-01
- [Clef Flash full-checkpoint checks](./benchmarks/clef-flash-native-qualification-2026-10-02.md):
  2026-10-02
- [Native Kolibri mixed 2-bit checks](./benchmarks/kolibri-native-qualification-2026-10-04.md):
  2026-10-04
- [Clef Omni native Q4 qualification](./benchmarks/clef-omni-q4-qualification-2026-10-09.md):
  2026-10-09
- [D1 native checkpoint qualification](./benchmarks/d1-native-qualification-2026-10-09.md):
  2026-10-09

The Bonsai 2 Small reports form one sequence. Read them in this order:

1. [Native checkpoint qualification](./benchmarks/bonsai2-small-2026-10-15.md):
   2026-10-15
2. [Upstream parity](./benchmarks/bonsai2-small-upstream-parity-2026-10-15.md):
   2026-10-15
3. [Upstream parity with precise Metal frequencies](./benchmarks/bonsai2-small-upstream-parity-2026-10-16.md):
   2026-10-16
4. [Fixed-runtime quality and throughput](./benchmarks/bonsai2-small-fixed-2026-10-16.md):
   2026-10-16
5. [Upstream throughput and scheduling fix](./benchmarks/bonsai2-small-throughput-2026-10-16.md):
   2026-10-16
6. [Selected-runtime quality](./benchmarks/bonsai2-small-selected-quality-2026-10-16.md):
   2026-10-16

### Image, vision, and video models

- [Qwen Image 2.1 native qualification](./benchmarks/qwen-image-21-native-qualification-2026-09-20.md):
  2026-09-20
- [LightOnOCR 3 native checkpoint qualification](./benchmarks/lightonocr3-native-qualification-2026-10-08.md):
  2026-10-08
- [Qwen Image 2.1 Turbo mixed-precision qualification](./benchmarks/qwen-image-21-turbo-mixed-qualification-2026-10-09.md):
  2026-10-09
- [FLUX 3 Klein native checkpoint qualification](./benchmarks/flux3-klein-native-qualification-2026-10-15.md):
  2026-10-15
- [VFX geometry Apple Silicon benchmarks](https://github.com/sawfwair/mere-run/blob/main/docs/benchmarks/vfx-geometry-apple-silicon.md):
  MoGe-2, Video Depth Anything Small, Depth Anything 3 Small, TripoSR, and
  InstantMesh Base on M4 Max
- [MiniMax-H3 BF16 on M4 Max](./benchmarks/minimax-h3-bf16-m4-max.md):
  end-to-end FL2VA generation and later cache, window, and adapter receipts
- [MiniMax-H3 compact BF16 and Q8 on M4 Max](./benchmarks/minimax-h3-compact-bf16-q8-m4-max.md):
  compact-artifact rollout and quantization comparison
- [FastH3 VSA DataFree Metal qualification](./benchmarks/minimax-h3-fasth3-vsa-datafree-m4-max.md):
  bounded generation for the managed FastH3 VSA DataFree package
- [MiniMax-H3 Ref2VA MLX 8-bit validation](./benchmarks/minimax-h3-ref2va-mlx-8bit.md):
  corrected Ref2VA artifact and one native generation
- [MiniMax-H3 h3.c transfer program](./benchmarks/minimax-h3-h3c-transfer.md):
  the evidence contract and results for transferring h3.c techniques

### Audio and music models

- [AuK native qualification](./benchmarks/auk-native-qualification-2026-09-18.md):
  2026-09-18
- [MiniMax Music 3 on M4 Max](./benchmarks/minimax-music3-m4-max.md):
  installed-checkpoint performance work

## Related references

- [CLI reference](./cli.md)
- [Testing guide](./testing.md)
- [Configuration](./configuration.md)
- [Text runtime](./runtime/text.md)
- [Local API server](./runtime/api-server.md)
