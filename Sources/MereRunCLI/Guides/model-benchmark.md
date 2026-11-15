# Model Benchmark

## Purpose

Run focused local model benchmarks that measure implementation tradeoffs without changing serving defaults.

Run the versioned, sampled Mere Lite fused quality plan without loading models:

```bash
mere.run model benchmark fused --suite lite --dry-run --json
```

Run the Comprehensive fused suite with final-target logprob summaries and
explicit sandboxed coding evaluation:

```bash
mere.run model benchmark fused \
  --suite comprehensive \
  --logprobs summary \
  --allow-code-execution \
  --json
```

Lite contains 24 cases and Comprehensive contains 110 cases across
chat/long-context, code, tool-use, and vision. The fused quality lane uses
native non-greedy model profiles and repeated trials. `--performance-lane
native` adds separately labeled timing rows with logprob capture off; those
rows never contribute correctness scores.

The exact Comprehensive mix is 52 Mere-authored chat cases, seven LongBench
cases, three HumanEval cases, six HumanEval+ cases, six MBPP+ cases, six
LiveCodeBench cases, ten Mere-authored tool cases, ten BFCL cases, and ten
generated Mere vision cases. Lite selects 10, 2, 1, 1, 1, 1, 3, 2, and 3 from
those families respectively. Run `--dry-run --json` to inspect every resolved
case id, capability tag, difficulty, and selection rationale. The Mere fused
model evaluation documentation also shows literal Mere-owned question/answer
and tool-call examples, plus representative code, long-context, and vision
cases.

The upstream HumanEval+, MBPP+, LiveCodeBench, BFCL, and LongBench datasets are
not vendored. Generate the pinned 35-case normalized selection from the
hash-verified source lock, stamp it, and verify it before model loading:

```bash
python3 scripts/import-fused-benchmark-fixtures.py
mere.run model benchmark fused-fixture \
  .tmp/fused-benchmark-fixtures/unstamped.jsonl \
  > .tmp/fused-benchmark-fixtures/selected.jsonl
mere.run model benchmark fused-fixture \
  .tmp/fused-benchmark-fixtures/selected.jsonl --check
```

Pass the frozen file to `fused` with `--external-cases`. The selection exercises
Plus tests, LiveCodeBench public/private execution, BFCL multi-call and no-call
semantics, and LongBench QA-F1, ROUGE-L, and retrieval scoring. It is a Mere
subset with pinned provenance, not an upstream leaderboard reproduction.

Fused runs never download missing models. For a long run, checkpoint every
completed case-trial and yield cleanly after a bounded slice:

```bash
mere.run model benchmark fused \
  --suite comprehensive \
  --models vision-chat-q38-27b \
  --external-cases .tmp/fused-benchmark-fixtures/selected.jsonl \
  --allow-code-execution \
  --checkpoint .tmp/fused-benchmark-results/qwen38.json \
  --case-trial-limit 5 \
  --json
```

Repeat the exact command with `--resume`. Resume verifies the complete plan and
installed model-manifest hashes before skipping completed rows. The plan also
binds the exact runner binary, host identity, resolved Python binary, and
resolved sandbox backend. Model preparation or generation failures preserve
prior rows but leave the current row pending for retry. `--dry-run` can create
or validate an empty checkpoint without model loading or Metal work.

```bash
mere.run model benchmark gemma4-kv \
  --model text-chat-gemma4-turbo \
  --decode-tokens 48 \
  --json
```

Run a small promotion matrix by varying prompt fixture size and decode length:

```bash
mere.run model benchmark gemma4-kv \
  --model text-chat-gemma4-turbo \
  --prompt-repeat-values 32,128,220 \
  --decode-token-values 32,128 \
  --json
```

Compare Gemma4 12B serial decode against verified MTP speculative decode:

```bash
mere.run model benchmark gemma4-mtp \
  --model text-chat-gemma4-12b-4bit \
  --prompt-repeat-values 128,220 \
  --decode-token-values 48,128 \
  --json
```

Compare Qwen-family baseline decode against adaptive and forced MTP policies:

```bash
mere.run model benchmark q36-mtp \
  --model vision-chat-q38-27b-4bit \
  --prompt-repeat-values 8,80,150 \
  --temperature-values 0,0.7 \
  --decode-tokens 32 \
  --json
```

Replay a real OpenAI-compatible chat workload against a running API server:

```bash
MERERUN_GEMMA4_PREFIX_KV_CACHE=0 \
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 1

mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --json
```

Compare the installed coding models on a small HumanEval slice:

```bash
mere.run model benchmark code \
  --allow-code-execution \
  --json
```

Compare Gemma4 12B vision chat against the existing Qwen3-VL inspect backend:

```bash
mere.run model benchmark vlm --json
```

Run the small grounded-chat slice for local assistant behavior:

```bash
mere.run model benchmark chat --json
```

The chat, tool-call, and code lanes expose the same stochastic controls:
`--temperature`, `--top-p`, `--top-k`, and `--min-p`. Min-p removes tokens
whose probability is below the selected fraction of the most likely token's
probability, after the configured top-k/top-p filters. It is disabled by
default and has no effect on greedy generation.

Run the small tool-call selection slice for Q36 vs Gemma 12B 4-bit:

```bash
mere.run model benchmark tool-calls --json
```

Prepare an external VLM quality run against an existing `lmms-eval` dataset:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git ~/src/lmms-eval
mere.run model benchmark vlm \
  --dataset mathvista-testmini \
  --limit 16 \
  --lmms-eval-root ~/src/lmms-eval \
  --dry-run \
  --json
```

Probe another installed chat model to confirm whether its package can accept
image prompts:

```bash
mere.run model benchmark vlm \
  --models vision-chat-gemma4-12b,vision-inspect-qwen3-vl-2b,text-chat-q36-nano \
  --json
```

## Gemma4 Tool-Continuation Benchmark

Validate grounded continuation after completed Gemma 4 tool calls:

```bash
mere.run model benchmark tool-continuations --log-responses
```

The cases cover typed nested/null arguments, reasoning preservation, tool-call
ids, tool names, and a repeated two-tool chain. `--model-root` accepts a local
converted Gemma 4 directory without changing the managed installation.

## Gemma4 KV Benchmark

`model benchmark gemma4-kv` runs a fixed-token comparison for the selected
Gemma4 checkpoint:

- `default`: the model's normal Gemma4 KV cache settings.
- `polar2`: model-default prefill, then packed 2-bit PolarKV from token 0 for decode.

The command disables EOS stopping so both variants decode exactly
`--decode-tokens`. Output includes prompt tokens, generated tokens, load time,
prefill time, KV conversion time, decode time, TTFT, prefill tok/s, decode tok/s,
end-to-end tok/s, and process resident memory before and after each variant.

## Gemma4 MTP Benchmark

`model benchmark gemma4-mtp` runs a fixed-token comparison for the selected
Gemma4 checkpoint:

- `baseline`: serial greedy decode with `MERERUN_GEMMA4_MTP=0`.
- `mtp`: verified MTP speculative decode with `MERERUN_GEMMA4_MTP=1`.

The command defaults to `text-chat-gemma4-12b-4bit`, disables EOS stopping so
both variants decode exactly `--decode-tokens`, and reports decode speedup,
end-to-end speedup, prompt tokens, generated tokens, timing, throughput, process
resident memory, and MTP counters for rounds, drafted tokens, accepted tokens,
rejected tokens, acceptance rate, and accepted tokens per round.

Use `--mtp-block-size` to test a different draft block cap, or
`--mtp-min-prompt-tokens` to force active MTP on shorter local fixtures. Leaving
those unset uses the runtime policy defaults. This is a throughput and verifier
benchmark, not a model-quality eval.

## Qwen3.6 MTP Benchmark

`model benchmark q36-mtp` runs requested-token comparisons for Qwen-family
models with MTP draft heads (`text-chat-q36-nano` by default; `--model` also
accepts `text-agent-ornith-9b` and the official Ornith 1.5 Q4/Q6/Q8/BF16 ids).
Managed Ornith 1.5 pulls install one shared, pinned MTP companion:

- `baseline`: MTP disabled with `MERERUN_Q35_MTP_SPECULATION=0`.
- `adaptive`: production policy (short-prompt MTP for Ornith 1.5; the measured
  long-context threshold for Qwen3.6).
- `forced`: MTP enabled with `MERERUN_Q35_MTP_SPECULATION=1` and a configurable
  forced threshold, defaulting to `1` token.

The command forces exactly `--decode-tokens` per variant and reports prompt
tokens, timing, throughput, process resident memory, acceleration counters, and
decode/end-to-end speedups for adaptive and forced MTP against baseline. Greedy
scenarios also report output SHA-256 values and a cross-variant parity verdict.
Use `--temperature-values 0,0.7` to compare deterministic
greedy decode against the default chat sampling temperature. Greedy forced MTP
uses the native block verifier; non-greedy forced MTP stays on the exact
probabilistic speculative path.

## API Workload Benchmark

`model benchmark api-workload` replays streaming `/v1/chat/completions`
requests against an already-running `mere.run api serve` process. It is the
serving-path benchmark for prefix reuse, request admission, and supported
automatic decode batching. The built-in workload uses a deterministic stable
system prefix with different final user turns so prefix-cache hits, TTFT, and
active-request behavior are visible in `/runtime/status`.

Run the same workload twice, with prefix reuse disabled for the baseline and
then default prefix reuse plus `--max-active-requests 4`, before promoting cache
or batching changes. Gemma4, Qwen-family, and LFM2 automatically engage their
compatible-row batching implementations above concurrency `1`; their
engine-specific variables are force-on/force-off overrides for explicit A/Bs:

```bash
MERERUN_GEMMA4_PREFIX_KV_CACHE=0 \
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 1

mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --json

mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 4

mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --concurrency 4 \
  --json
```

Output includes per-request status, TTFT, total latency, streamed chunk count,
and runtime-status deltas for prefix KV hits/misses, reused tokens, decode
batched steps, single decode steps, completed requests, failed requests, and
whether SSD KV cache is available. SSD cache promotion should require repeatable
TTFT or throughput wins from in-memory prefix reuse first.

Use `--workload-file` to replay JSONL:

```json
{"id":"case-1","user":"Summarize the runtime benchmark rule."}
{"id":"case-2","messages":[{"role":"system","content":"Shared product context..."},{"role":"user","content":"Answer from that context."}]}
```

## Prompt Control

Use one of:

- `--prompt`: inline prompt text.
- `--prompt-file`: UTF-8 prompt file.
- `--prompt-repeat`: repeat count for the built-in deterministic fixture.
- `--prompt-repeat-values`: comma-separated fixture sizes for a benchmark matrix.
- `--decode-token-values`: comma-separated decode lengths for a benchmark matrix.
- `--temperature-values`: comma-separated sampling temperatures for the Qwen3.6
  MTP matrix.

The fixture prompt is for runtime comparison only; it is not a quality eval.

## Chat Benchmark

`model benchmark chat` runs a tiny, fixed Mere-style chat slice against local
assistant models. It is a grounded behavior check, not a leaderboard substitute.
The default comparison lane is:

- the hardware-aware default from `text chat`
- `text-chat-lfm25-a1b-8bit`
- `text-chat-gemma4-12b-4bit`
- `text-chat-gemma4-nano`

The default suite is `mere-chat-slice`, currently 40 original fixtures covering:

- sender-specific local email questions
- missing-evidence abstention with `NOT_IN_EVIDENCE`
- workspace ambiguity and workspace selection
- JSON extraction
- concise summaries from provided evidence
- local action boundaries
- date sorting and small arithmetic
- conflicting evidence
- avoiding fabricated links, secrets, and command output

Each case is prompted once with deterministic sampling by default
(`--temperature 0 --top-p 1`) and scored with deterministic checks such as
required phrases, forbidden phrases, regexes, JSON keys, and bullet counts. The
command never auto-pulls models during scoring; install missing models with
`mere.run model pull` first.

For a controlled stochastic comparison, keep the prompt suite and every other
sampling control fixed and vary only `--min-p`:

```bash
mere.run model benchmark chat \
  --temperature 1 \
  --top-p 1 \
  --top-k 20 \
  --min-p 0.05 \
  --log-responses \
  --json
```

Narrow the run while iterating:

```bash
mere.run model benchmark chat \
  --models text-chat-lfm25-a1b-8bit,text-chat-gemma4-nano \
  --cases MereChat/0,MereChat/1,MereChat/3 \
  --log-responses
```

Use `--dry-run --json` to print the planned model/case matrix without loading a
runtime. Use `--log-responses` when debugging a failure; otherwise reports keep
responses out of the text and JSON output.

## Tool-Call Benchmark

`model benchmark tool-calls` runs a separate tool protocol check. It is split
from `model benchmark chat` on purpose: the chat slice scores grounded answers,
while this slice scores whether a model chooses the right tool name and
arguments when the prompt requires a local lookup or action.

The default comparison is:

- `text-chat-q36-nano`
- `text-chat-gemma4-12b-4bit`

The default suite currently has 10 original Mere-style cases covering:

- email search by sender, workspace, date, and attachment flag
- opening a local email by id
- project and audit search
- workspace selection
- safe dry-run command planning
- no-tool cases when evidence is already present or confirmation is missing

The benchmark passes synthetic `ToolDefinition` schemas to the runtime and
scores the parsed `toolCalls` returned by the model. It does not execute shell
commands, write files, mutate local data, or invoke the real Mere command plane.

Narrow the run while iterating:

```bash
mere.run model benchmark tool-calls \
  --models text-chat-q36-nano,text-chat-gemma4-12b-4bit \
  --cases MereTool/0,MereTool/4 \
  --log-responses
```

## Code Benchmark

`model benchmark code` runs a tiny, fixed HumanEval slice against local coding
models. It is a real functional-code eval slice, not a full pass@k benchmark or
leaderboard substitute. The default comparison auto-selects the supported
members of this lane for the current machine:

- `text-agent-ornith-9b`: native Q35/MLX OptiQ coding-agent target.
- `text-code-north-mini`: native llama.cpp/GGUF North Mini Code target.
- `text-code-qwen3`: native llama.cpp/GGUF Qwen3-Coder baseline, included by
  default only on 64 GB and larger machines.

For larger explicit Ornith runs, pass `--models text-agent-ornith-35b-mlx` for
the official 1.5 35B-A3B BF16 MLX snapshot or `--models text-agent-ornith-35b` for the
GGUF target. They are not part of the default comparison because they are larger
installs/loads.

The default suite is `humaneval-slice`, currently three public HumanEval tasks:

- `HumanEval/0`
- `HumanEval/3`
- `HumanEval/8`

Each task is prompted once with deterministic sampling by default
(`--temperature 0 --top-p 1`), the generated Python is combined with the task's
tests, and the candidate is executed with `python3` inside the selected sandbox
backend. Because this runs generated code locally, the command requires
`--allow-code-execution` unless you are using `--dry-run`. The default
`--sandbox auto` uses `sandbox-exec` on macOS and `bubblewrap` on Linux when
available. Use `--sandbox none` only for a trusted local smoke where timeout and
temporary-directory hygiene are enough. The default generation cap is
`--max-tokens 1024`; JSON and text output flag cases that still reach the cap
with `reachedMaxTokens`/`capped=true`. Reasoning-model output is split before
scoring: visible code is executed, while captured `<think>...</think>` content
is reported as `reasoningCharacters`/`reasoning_chars` and
`incompleteReasoning`/`reasoning_incomplete`. A second generated reasoning
block is reported as `reasoning_reopened=true`; treat it as a loop or
phase-restart warning, not a correctness failure by itself.

Narrow the run while iterating:

```bash
mere.run model benchmark code \
  --models text-agent-ornith-35b \
  --tasks HumanEval/0 \
  --allow-code-execution
```

Use `--python` to select a Python interpreter and `--execution-timeout` to
control the per-candidate subprocess timeout. The command never auto-pulls
models during scoring; install missing models with `mere.run model pull` first.

For a larger slice, download the official HumanEval JSONL, decompress it, and
pass it with `--humaneval-file`:

```bash
curl -L https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o /tmp/HumanEval.jsonl.gz
gunzip -c /tmp/HumanEval.jsonl.gz > /tmp/HumanEval.jsonl
mere.run model benchmark code \
  --humaneval-file /tmp/HumanEval.jsonl \
  --tasks HumanEval/0,HumanEval/1,HumanEval/2,HumanEval/3,HumanEval/4 \
  --models text-agent-ornith-35b \
  --allow-code-execution
```

## VLM Benchmark

`model benchmark vlm` writes a tiny deterministic image-question suite to a
fixture directory, runs each requested vision-language backend, and grades
answers with simple regex checks. The default suite covers:

- dominant color recognition
- corner/location recognition
- simple object counting

The default comparison is:

- `vision-chat-gemma4-12b`: the managed Gemma4 12B unified vision-chat runtime.
- `vision-inspect-qwen3-vl-2b`: the existing `vision inspect` Qwen3-VL 2B backend.

Pass `--models` with comma-separated ids or aliases to compare a different set.
Managed API-runtime models must already be installed. If a chat model does not
include a usable vision tower, the benchmark records that as a case failure
instead of treating it as a VLM peer. The Qwen3-VL inspect backend follows
`vision inspect` behavior and downloads its small model on first use if needed.

Output includes pass/fail, model response, elapsed time, prompt/generated token
counts when the runtime exposes them, timing when available, and process
resident memory before and after each case. This is an onboarding and regression
benchmark, not a public leaderboard.

### Existing VLM Datasets

`model benchmark vlm` can also prepare and run `lmms-eval` tasks through
mere.run's OpenAI-compatible API server. Presets map to upstream task names:

| Dataset flag | lmms-eval task |
| --- | --- |
| `mathvista-testmini` | `mathvista_testmini` |
| `mmmu-val` | `mmmu_val` |
| `chartqa` | `chartqa` |
| `docvqa-val` | `docvqa_val` |
| `mme` | `mme` |

Use `--dry-run --json` first to print the exact command without starting a
server or running downloads. A non-dry run starts one local `mere.run api serve`
process per requested model, waits for `/health`, then launches:

```bash
python3 -m lmms_eval \
  --model openai \
  --model_args model_version=vision-chat-gemma4-12b,base_url=http://127.0.0.1:11934/v1,api_key=mere-run-local-eval \
  --tasks mathvista_testmini
```

Use `--lmms-tasks` to pass raw upstream task names instead of a preset:

```bash
mere.run model benchmark vlm \
  --lmms-tasks mathvista_testmini,chartqa \
  --limit 32 \
  --lmms-eval-root ~/src/lmms-eval \
  --json
```

Use `--external-endpoint --base-url` when the API server is already running:

```bash
mere.run api serve \
  --engine text-chat-gemma4 \
  --model vision-chat-gemma4-12b \
  --port 11934 \
  --api-key mere-run-local-eval

mere.run model benchmark vlm \
  --dataset mathvista-testmini \
  --external-endpoint \
  --base-url http://127.0.0.1:11934/v1 \
  --limit 16 \
  --json
```

## Notes

- Pull `text-chat-gemma4-turbo` before running the benchmark.
- Pull the supported models shown by `mere.run model benchmark code --dry-run`
  before running the default code benchmark comparison.
- Pull `text-agent-ornith-35b` before running larger explicit Ornith code evals.
- Pull `vision-chat-gemma4-12b` before using it in the VLM benchmark.
- Install `lmms-eval` dependencies in the selected Python environment before
  running external datasets; dataset downloads and licenses are handled by the
  upstream task definitions.
- Prefer release builds for final numbers.
- Treat memory values as process resident snapshots, not peak memory.

## Sources

- `Sources/MereRunCLI/Commands/ModelBenchmarkCommand.swift`
- `Sources/MereRunCLI/Commands/ModelBenchmarkCodeCommand.swift`
- `Sources/MereRunCLI/Commands/ModelBenchmarkVLMCommand.swift`
- `Sources/MereRunCore/Gemma4/Gemma4Generator.swift`
- `Sources/MereRunGemmaModel/Gemma4KVQuantization.swift`


## Resident ASR comparison

`model benchmark asr short.wav long.wav --backend parakeet --model-root /path/to/model --json`
compares recordings with one checkpoint held resident. Select `parakeet`, `qwen`,
`whistle`, `nemotron`, or `sov`; use the same `--warmups` and `--repetitions` for
all models. JSON includes every transcript, first-inference and resident timing,
separate model-load/audio-read times, and memory peaks. The command requires
local assets. Core ML uses `--artifact`; SOV can use `--draft-model` and
`--experimental-dynamic-int8` or `--experimental-decoder q8`. Whistle exposes beam size, decoder depth, and
optional word alignment.

`model benchmark sov` adds stage timings, exact token sequences, and experimental
decoder selection through `--experimental-decoder bf16-gemv`, `q4`, `q6`, `mixed`,
`q6-matrix`, `q8`, or `q8-g32`.
The default remains `bf16`. Q8 packs only MoE expert weights during preparation;
qualify token agreement and transcription quality against the original target.
`--experimental-decoder q8-metal` adds a custom Metal expert-vector kernel to
the Q8 group-64 path. It shares weights across two sorted routes and retains
the regular matrix path for larger batches.

`q4` and `q6` pack affine expert weights in groups of 64. `mixed` requires
`--expert-recipe path.json`, a typed recipe with `schemaVersion: 1`,
`groupSize: 32` or `64`, and `layerBits`, one value of 4, 6, or 8 per decoder
layer. Mixed group-64 Q8 layers use the custom Metal path; lower precisions use
MLX. The JSON report records the recipe digest. These policies change target
weights and require held-out recognition evaluation.

`q6-matrix` retains Q6 verification and caches the same six-bit codes in
eight-bit containers for large sorted prefill batches. Shared scales and biases
preserve the quantized weight values. This uses the M5 Q8 matrix backend without
another weight quantization. It adds a second expert representation in memory;
matrix accumulation can still change outputs and requires recognition checks.

Add `--interleave-experts` to compare Q4, Q6, or mixed against a separately
resident Q8 Metal model with identical generation and attention settings.
`--experts-first` reverses starting order. This comparison uses two model copies
in memory; JSON records their combined active allocation and the baseline load
time. It excludes other interleaving modes and allocator-retention experiments.

`--wire-models` requests explicit Metal residency throughout sampling through
MLX's managed wired-memory tickets. The budget is active MLX memory plus 4 GiB,
capped just under the GPU's recommended working set. It is restored after
sampling or errors. `--interleave-wired` instead pairs default and wired
residency on one model; `--wired-first` reverses starting order. This excludes
other interleaving, allocator retention, and projected-output modes. Ticket
transitions occur outside each measured interval. JSON records the requested
budget and labels wired samples. These are benchmark controls; normal SOV
transcription does not change its residency policy.

For paired measurements, run `model benchmark sov` with
`--experimental-decoder q8 --interleave-metal`. It alternates stock and custom
kernels on one prepared model, preserving packed weights and compiled shaders.
Warmups/repetitions apply to each mode. `--metal-first` reverses the starting
order. JSON labels each measurement and records synchronized resident time.

`model benchmark sov --experimental-attention` enables shared position masks and
direct proportional partial RoPE in the target decoder. It preserves parameters
and remains off by default. With `q8-metal` selected, use `--interleave-attention` for paired attention
measurements, and `--attention-first`
to reverse their starting order. This comparison holds expert dispatch constant;
it cannot be combined with `--interleave-metal` or `--experimental-attention`.

For SOV on the MLX GPU, `--experimental-head-norm` fuses BF16 head normalization
and RoPE while retaining intermediate rounding. `--interleave-head-norm` and
`--head-norm-first` provide paired measurements with expert dispatch and the
attention policy held constant. Only one interleaving mode can be active.

SOV generation experiments use `--generation-experiment pipeline|draft-q8|shared-cache|retained-allocator|combined`.
The pipeline keeps proposal IDs on the GPU and overlaps drafting with target
graph construction. `draft-q8` packs a separate draft-only vocabulary head;
`shared-cache` shares the sliding attention concatenation with retained KV state.
`retained-allocator` skips request cache clearing, subject to MLX's configured
allocator limit. `combined` selects all four. Ordinary transcription uses the
weight-preserving pipeline automatically with DFlash; this benchmark defaults
to explicit `reference` scheduling.

Add `--interleave-generation` to pair `pipeline`, `draft-q8`, or `shared-cache`
with the reference on one resident target. `--generation-first` reverses the
starting order. Warmups and repetitions apply per mode. Other interleaving and
projected-output options are excluded. Allocator retention requires separate
resident runs. JSON includes per-round acceptance, emitted draft counts,
allocator cache bytes, and peak allocation.

Further SOV research policies are `rollback`, `compiled`, `block-4`, `block-8`,
`block-12`, `adaptive`, `adaptive-after-first`, and `vocabulary-32k`. Pair each
against the production pipeline with `--generation-baseline pipeline
--interleave-generation`. Fixed block sizes include the anchor; confidence
truncation uses a 0.5 threshold. The after-first policy preserves the initial
full block. `compiled` covers each MoE layer's segment after attention with
runtime parameter inputs. All these policies remain explicit experiments.

`--draft-vocabulary vocabulary.json` supplies global token IDs for the restricted
head. Calibration happens before timing. JSON records the file digest and
`proposedLengths` alongside accepted and emitted draft counts.
