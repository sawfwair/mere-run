# Model Benchmark

## Purpose

Run focused local model benchmarks that measure implementation tradeoffs without changing serving defaults.

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

Compare Qwen3.6 baseline decode against adaptive and forced MTP policies:

```bash
mere.run model benchmark q36-mtp \
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

`model benchmark q36-mtp` runs requested-token comparisons for
`text-chat-q36-nano`:

- `baseline`: MTP disabled with `MERERUN_Q35_MTP_SPECULATION=0`.
- `adaptive`: production policy with the default long-context threshold.
- `forced`: MTP enabled with `MERERUN_Q35_MTP_SPECULATION=1` and a configurable
  forced threshold, defaulting to `1` token.

The runtime may still stop on EOS before `--decode-tokens`. The command reports prompt tokens, generated tokens, timing, throughput, process
resident memory, and decode/end-to-end speedups for adaptive and forced MTP
against baseline. Use `--temperature-values 0,0.7` to compare deterministic
greedy decode against the default chat sampling temperature. Greedy forced MTP
uses the native block verifier; non-greedy forced MTP stays on the exact
probabilistic speculative path.

## API Workload Benchmark

`model benchmark api-workload` replays streaming `/v1/chat/completions`
requests against an already-running `mere.run api serve` process. It is the
serving-path benchmark for prefix reuse, request admission, and opt-in decode
batching. The built-in workload uses a deterministic stable system prefix with
different final user turns so prefix-cache hits, TTFT, and active-request
behavior are visible in `/runtime/status`.

Run the same workload twice, with prefix reuse disabled for the baseline and
then default prefix reuse plus opt-in batching enabled, before promoting cache
or batching changes:

```bash
MERERUN_GEMMA4_PREFIX_KV_CACHE=0 \
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-turbo \
  --max-active-requests 1

mere.run model benchmark api-workload \
  --model text-chat-gemma4-turbo \
  --json

MERERUN_GEMMA4_CONTINUOUS_BATCHING=1 \
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

## Code Benchmark

`model benchmark code` runs a tiny, fixed HumanEval slice against local coding
models. It is a real functional-code eval slice, not a full pass@k benchmark or
leaderboard substitute. The default comparison is:

- `text-agent-ornith-9b`: native Q35/MLX OptiQ coding-agent target.
- `text-code-north-mini`: native llama.cpp/GGUF North Mini Code target.
- `text-code-qwen3`: native llama.cpp/GGUF Qwen3-Coder baseline.

For larger explicit Ornith runs, pass `--models text-agent-ornith-35b-mlx` for
the local native MLX Q4 conversion or `--models text-agent-ornith-35b` for the
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
- Pull `text-agent-ornith-9b`, `text-code-north-mini`, and `text-code-qwen3`
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
- `Sources/MereRunCore/Gemma4/Gemma4KVQuantization.swift`
