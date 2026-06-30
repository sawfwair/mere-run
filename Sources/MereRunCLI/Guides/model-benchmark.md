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

Compare Gemma4 12B vision chat against the existing Qwen3-VL inspect backend:

```bash
mere.run model benchmark vlm --json
```

Run the small grounded-chat slice for local assistant behavior:

```bash
mere.run model benchmark chat --json
```

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
- Pull `vision-chat-gemma4-12b` before using it in the VLM benchmark.
- Install `lmms-eval` dependencies in the selected Python environment before
  running external datasets; dataset downloads and licenses are handled by the
  upstream task definitions.
- Prefer release builds for final numbers.
- Treat memory values as process resident snapshots, not peak memory.

## Sources

- `Sources/MereRunCLI/Commands/ModelBenchmarkCommand.swift`
- `Sources/MereRunCLI/Commands/ModelBenchmarkVLMCommand.swift`
- `Sources/MereRunCore/Gemma4/Gemma4Generator.swift`
- `Sources/MereRunCore/Gemma4/Gemma4KVQuantization.swift`
