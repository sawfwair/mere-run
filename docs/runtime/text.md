# Text runtime

The `mere.run text` commands run language models locally. Use them to stream
chat, run a sandboxed tool loop, ask about an image, produce a grammar-checked
JSON object, and fine-tune a LoRA adapter on your own conversations. Separate
commands generate code from GGUF models, create embeddings for local search,
answer typed decision questions, classify and extract text, and redact
personally identifiable information (PII).

This page lists the commands and models, then covers chat behavior by model
family. Models with their own guide are listed in
[Model guides](#model-guides).

## Commands

| Command | What it does |
| --- | --- |
| `mere.run text chat` | Run local chat, tool loops, image input, and JSON output with chat models. |
| `mere.run text code` | Generate code with GGUF models through llama.cpp. |
| `mere.run text embed` | Create Qwen3, [EmbeddingGemma 2](./embeddinggemma2.md), or [PPLX Embed v2](./pplx-embed-v2.md) embeddings for text, code, images, audio, and video. |
| `mere.run text decide` | Answer typed choice, score, and yes-or-no questions with [Laya](./laya.md), [Clef](./clef.md), or [D1](./d1.md). |
| `mere.run text classify` | Classify text against labels you supply with [GLiNER2.5 Decide](./gliner25-decide.md). |
| `mere.run text extract` | Extract entities, relations, and structured records with [GLiNER2.5 Decide](./gliner25-decide.md). |
| `mere.run text anonymize` | Detect and redact PII with OpenAI Privacy Filter. |
| `mere.run text train-lora` | Train a text or Gemma 4 image-conditioned LoRA adapter from chat-style SFT JSONL. |

Run `mere.run text <command> --help` for the complete flag list. In CLI
commands, stdout carries the result and stderr carries progress and
diagnostics.

## Model guides

These models have a dedicated guide with install steps, options, limits, and
validation results:

| Guide | Model IDs | Use it for |
| --- | --- | --- |
| [Recursive Flint](./flint.md) | `text-chat-flint-4bit`, `text-chat-flint-bf16` | MoE chat and tools from the published INT4 export or original BF16 weights |
| [Nemotron 3.5 Bolt](./nemotron-bolt.md) | `vision-chat-nemotron-35-bolt-24gb` | Text and image chat from NVIDIA's original mixed packed checkpoint |
| [Native Kolibri-1](./kolibri.md) | Converted local checkpoint directories | MoE chat from mixed 2-bit, 8-bit, or BF16 conversions, with paired-logprob tooling |
| [EmbeddingGemma 2](./embeddinggemma2.md) | `text-embed-embeddinggemma2`, `text-embed-embeddinggemma2-coreml` | Text, code, image, audio, and video embeddings in one space |
| [PPLX Embed v2](./pplx-embed-v2.md) | `text-embed-pplx-v2-late-*`, `text-embed-pplx-v2-context-9b-preview*` | Late-interaction token vectors and contextual chunk vectors |
| [GLiNER2.5 Decide](./gliner25-decide.md) | `text-classify-gliner25-decide` | Label classification and schema extraction |
| [Laya](./laya.md) | `text-decide-laya`, `text-decide-laya-multilingual`, `text-decide-laya-typed-decisions` | Typed decisions over plain text; the `text decide` default |
| [Clef](./clef.md) | `text-decide-clef-4bit`, `text-decide-clef-flash-4bit`, `text-decide-clef-omni` | Typed decisions over JSON state, images, video, and audio |
| [D1](./d1.md) | `text-decide-d1-3b-bf16`, `text-decide-d1-omni-600m-fp32` | Typed decisions over text, JSON, images, or audio |

For pinned sources, licenses, and sizes of every managed model, see
[Model sources](../model-sources.md).

## Choose a chat model

`text chat` defaults to `text-chat-gemma4-12b-4bit` on Apple Silicon and
`text-chat-q36-nano-gguf` on Linux CUDA. `mere.run model capabilities
--recommended` reports the recommendation for your machine. On Apple Silicon,
it follows these unified-memory bands:

| Unified memory | Recommended model | Notes |
| --- | --- | --- |
| 8–15 GB | `text-chat-lfm25-1.2b-qad-4bit` | Memory-first tier; step up to the 2.6B MLX model when its larger footprint fits. |
| 16–23 GB | `text-chat-gemma4-12b-4bit` | Compact default; use `text-chat-gemma4-nano` only when the 12B 4-bit model is too large. |
| 24–63 GB | `text-chat-gemma4-12b-4bit` | Default local assistant; `text-chat-gemma4-turbo` is the larger Gemma alternative. |
| 64–95 GB | `text-chat-gemma4-12b-4bit` | Keep the Gemma assistant and use the headroom for context, concurrency, or `text-chat-gemma4-max`. |
| 96 GB and up | `text-agent-deepseek-v4-flash` | Agent and API chat tier; keep Gemma 12B 4-bit for interactive local chat. |

On Linux CUDA, the 24–63 GB and 64–95 GB bands recommend
`text-chat-q36-nano-gguf` instead of Gemma.

## Model families

### Chat models

Most chat models run natively in Swift/MLX. Large models never download
implicitly; pull them with `mere.run model pull` first. The following table
groups the chat model IDs by family:

| Family | Model IDs | Details |
| --- | --- | --- |
| Gemma 4 | `text-chat-gemma4-12b-4bit`, `text-chat-gemma4-12b`, `vision-chat-gemma4-12b`, `text-chat-gemma4-nano`, `text-chat-gemma4-turbo`, `text-chat-gemma4-max`, `text-chat-gemma4` | [Gemma 4](#gemma-4). `text-chat-gemma4` is the 31B dense model kept for compatibility. |
| DiffusionGemma | `text-chat-diffusiongemma-26b-optiq-4bit` | [DiffusionGemma block diffusion](#diffusiongemma-block-diffusion) |
| Muse Glimmer | `vision-chat-muse-glimmer-30b` | [Muse Glimmer](#muse-glimmer) |
| Qwen3.6 | `text-chat-q36-nano`, `text-chat-q36-nano-gguf` | MLX on Apple Silicon; the GGUF build runs through llama.cpp and is the Linux CUDA default. |
| Qwen3.8 | `vision-chat-q38-27b`, `vision-chat-q38-27b-4bit`, `vision-chat-q38-flash-next-mixed`, `vision-chat-q38-flash-next-3bit`, `vision-chat-q38-flash-next-3bit-native-ple`, `vision-chat-q38-flash-next-4bit` | [Qwen3.8](#qwen3-8). The Flash Next builds are described in [Model sources](../model-sources.md). |
| Bonsai | `text-chat-bonsai-27b-1bit`, `text-chat-bonsai-27b-2bit`, `text-chat-bonsai-2-2b-2bit`, `text-chat-bonsai-2-4b-2bit`, `text-chat-bonsai-2-9b-2bit`, `text-chat-bonsai-2-27b-2bit` | [Bonsai](#bonsai) |
| Ornith | `text-agent-ornith-9b`, `text-agent-ornith-35b-mlx-4bit`, `text-agent-ornith-35b-mlx-6bit`, `text-agent-ornith-35b-mlx-8bit`, `text-agent-ornith-35b-mlx`, `vision-chat-ornith-35b` | [Ornith](#ornith) |
| LFM2.5 | `text-chat-lfm25-1.2b-bf16`, `text-chat-lfm25-1.2b-qad-4bit`, `text-chat-lfm25-2.6b-4bit`, `text-chat-lfm25-2.6b-qad-4bit`, `text-chat-lfm25-2.6b-bf16`, `text-chat-lfm25-a1b-8bit`, `text-chat-lfm25-a1b-bf16`, `text-chat-lfm25-24b-a2b-bf16`, `vision-chat-lfm25-3b-8bit`, `vision-chat-lfm25-3b-bf16` | [LFM2.5](#lfm2-5) |
| Laguna | `text-chat-laguna-s-2-1`, `text-chat-laguna-xs-2-1` | [Laguna](#laguna) |
| Nemotron | `text-chat-nemotron-35-lightning`, `vision-chat-nemotron-35-bolt-24gb`, `omni-chat-nemotron3-nano-30b-a3b-bf16` | [Nemotron 3.5 Lightning](#nemotron-3-5-lightning), the [Bolt guide](./nemotron-bolt.md), and [Model sources](../model-sources.md) for Nemotron 3 Nano Omni |
| Inkling | `text-chat-inkling-small` | [Inkling-Small](#inkling-small) |
| Recursive Flint | `text-chat-flint-4bit`, `text-chat-flint-bf16` | [Flint guide](./flint.md) |
| Kolibri | Converted local checkpoints | [Kolibri guide](./kolibri.md) |
| DeepSeek V4 Flash | `text-agent-deepseek-v4-flash` | API and agent serving; see [DeepSeek V4 Flash](#deepseek-v4-flash). |
| Other | `text-chat-psi-agent`, `text-chat-mebot` | Psi is local-path-only; MeBot is for API serving and isn't a `text chat` model. |

### Code models

`text code` runs these GGUF models through llama.cpp:

- `text-code-qwen3` (default)
- `text-code-north-mini`
- `text-agent-ornith-35b`
- `text-agent-qwen35-9b`

### Embedding models

- `text-embed-qwen3-0.6b` (default)
- `text-embed-embeddinggemma2` and `text-embed-embeddinggemma2-coreml`; see
  [EmbeddingGemma 2](./embeddinggemma2.md).
- `text-embed-pplx-v2-late-0.6b`, `text-embed-pplx-v2-late-9b`,
  `text-embed-pplx-v2-late-9b-mixed-4bit`,
  `text-embed-pplx-v2-context-9b-preview`, and
  `text-embed-pplx-v2-context-9b-preview-8bit`; see
  [PPLX Embed v2](./pplx-embed-v2.md).

### Decision, classification, and extraction models

- Laya: `text-decide-laya` (the `text decide` default),
  `text-decide-laya-multilingual`, and `text-decide-laya-typed-decisions`
- Clef: `text-decide-clef-4bit`, `text-decide-clef-flash-4bit`, and
  `text-decide-clef-omni`
- D1: `text-decide-d1-3b-bf16` and `text-decide-d1-omni-600m-fp32`
- GLiNER2.5 Decide: `text-classify-gliner25-decide`

### Anonymization model

- `text-anonymize-privacy-filter`

## macOS Studio

The macOS app compiles against the same `MereRunContract` that
`mere.run catalog --json` emits.

- **Chat ▸ Chat** has an inspector for text or constrained `json_object`
  output; model-default, shown, or disabled reasoning; context and sampling
  controls; key-value (KV) cache quantization; a catalog adapter ID or local
  LoRA file; tool permissions; preflight; and installed-model enforcement. The
  controls render from the contract's own option metadata.
- **Text ▸ Embeddings** is a vector explorer with dimensions, norms, previews,
  and cosine similarity. Select EmbeddingGemma 2 to add image, audio, video, or
  mixed JSON inputs. Sending media from another Studio page to Embeddings
  selects a compatible managed model.
- **Text ▸ Anonymize** shows the original and protected text with labeled PII
  spans.
- **Text ▸ Decisions**, **Text ▸ Classify**, and **Text ▸ Extract** build Laya
  and GLiNER requests; see the [Laya](./laya.md) and
  [GLiNER2.5 Decide](./gliner25-decide.md) guides.
- **Chat ▸ Train** opens a text-dataset preview, preflight, live metrics,
  artifacts, history, and run comparison for native LoRA.

The Command view and the Command Console keep the raw command surface. The
repository gate checks every generated flag against the public ArgumentParser
help.

## Chat

### Run a chat

```bash
swift run mere.run text chat \
  --stream \
  --prompt "Summarize diffusion models in one paragraph."
```

With `--stream`, interactive terminals render common Markdown as it arrives.
Redirected stdout stays raw for scripts. Use `--markdown never` to keep raw
Markdown in a terminal or `--markdown always` to force structural rendering.
JSON response mode is always raw.

`text chat` checks numeric sampling and token limits before generation, and
`--preflight` runs the same checks without loading or downloading a model. Pass
`--require-installed` to fail instead of downloading. During a tool
conversation, the command keeps one native runtime and unloads it after
completion, failure, or cancellation. See
[shared chat execution](../internals/chat-execution.md) for the request and
runtime boundaries shared with the API.

### Sampling

Without explicit options, `text chat` samples at temperature 0.7 and top-p 0.9
with no top-k cutoff. Models that publish their own values use them instead;
the family sections on this page list those values.

Native and llama.cpp text generation also accept min-p sampling. A value such
as `0.05` removes tokens with less than 5% of the most likely token's
probability after top-k and top-p filtering, then renormalizes the remaining
distribution. Min-p is off by default except for Laguna S 2.1's measured
`0.02`, and it doesn't change temperature-zero greedy output. Set it with
`--min-p` in text commands or `min_p` in OpenAI-compatible chat requests.

### JSON output

Native MLX Gemma and Qwen-family chat models support
`--response-format json_object`. The output must be one complete root object.
Before each token streams, the prefix grammar checks nested objects and arrays,
strings and escapes, Unicode, numbers, booleans, null, commas, and colons.
Qwen-family JSON mode turns thinking off and bypasses MTP, continuous batching,
and pipelined decoding. The GGUF Qwen3.6 model used by Linux packages runs on
llama.cpp and rejects JSON-object mode.

```bash
swift run mere.run text chat \
  --model text-chat-q36-nano \
  --response-format json_object \
  --prompt 'Return an object with a name and an array of tags.'
```

### Tool use

`text chat` can run a local agentic tool loop. `--tools` takes a comma-separated
list of built-in tool names (`write_file`, `shell_exec`), and `--tool-loop`
turns on the loop: generate, run tool calls, feed results back, and continue,
up to 10 iterations.

- Tools run inside a sandbox directory. `--sandbox-dir` sets it; the default is
  a per-process temporary directory.
- Each call asks for interactive `[y/N]` approval by default.
- `shell_exec` also requires `--allow-shell-exec`, and it always asks for
  approval, even with `--auto-approve-tools`.
- `--auto-approve-tools` skips confirmation for `write_file` only.
- `write_file` targets absolute paths outside the sandbox only with
  `--allow-absolute-tool-paths`.

Approved shell commands run through `/bin/zsh` on macOS and `/bin/sh` on Linux.
Each call has a five-minute deadline. The tool drains stdout and stderr while
the command runs and keeps the first 256 KiB of their combined output. If the
output exceeds that limit, the result includes `[output truncated]`, but the
command keeps running. A timeout or task cancellation stops the owned process
group.

```bash
swift run mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --prompt "Create hello.py that prints the system time, then run it." \
  --tools write_file,shell_exec \
  --tool-loop \
  --allow-shell-exec \
  --sandbox-dir ./work
```

### Image input

`text chat --image` accepts a local file path or a `data:image/...` URI for
vision-capable chat models such as Bonsai 27B or `vision-chat-gemma4-12b`. The
image attaches to the user prompt for multimodal prefill. Omni chat models also
accept `--audio` and `--video`.

```bash
swift run mere.run text chat \
  --model text-chat-bonsai-27b-2bit \
  --image ./photo.png \
  --prompt "Describe what is in this photo."
```

### KV cache quantization

To reduce long-context memory in `text chat`, pass `--kv-bits 4` or `8`.
Qwen-family, Inkling-Small, and LFM2.5 models use affine KV quantization. Gemma 4
uses its model-specific path and also accepts `--kv-quant-scheme` (`uniform`,
`polar`, or `turboquant`), `--kv-group-size`, and `--quantized-kv-start`.
Qwen-family and LFM2 dequantize the generic cache for attention, so these modes
aren't assumed to be faster.

`api serve` accepts the same Gemma 4 flags. To set a per-model KV mode for API
serving, use `mere.run model runtime set <model> --kv-cache-mode` with
`affine4` or `affine8`, or the Gemma 4-only `polar2` or `auto`. `default`
restores the engine, model, or server default. For details, see
[Model management](./model-management.md).

Interactive `text chat` keeps the memory-oriented Gemma Turbo recipe. `api serve`
defaults Turbo to full-precision KV, because long-schema serving showed that
token-zero TurboQuant could dominate decode latency. To choose that memory
tradeoff for the API anyway, pass
`--kv-bits 4 --kv-quant-scheme turboquant --quantized-kv-start 0`.

## Chat model notes

### Gemma 4

`text-chat-gemma4-12b` runs Google's dense Gemma 4 12B-it checkpoint through the
native Swift Gemma runtime. Managed pulls for `text-chat-gemma4-12b` and
`vision-chat-gemma4-12b` also install the `google/gemma-4-12B-it-assistant`
companion as `text-chat-gemma4-12b-mtp`. When the companion is present, greedy
serial decode can use native MTP on the decode tail after text or multimodal
prefill. Sampled requests, prefix-KV seeded requests, continuous batching, raw
local model paths, and prompts below the MTP threshold use baseline decode.

When no assistant is active, greedy requests use prompt-lookup speculation by
default. Repeated context n-grams are drafted and verified in bursts inside the
pipelined loop. Output is identical, non-repetitive text costs nothing extra,
and generation that echoes the prompt runs about twice as fast. To turn it off,
set `MERERUN_GEMMA4_PROMPT_LOOKUP=0`; see
[Configuration](../configuration.md#mererun-gemma4-prompt-lookup).

### DiffusionGemma block diffusion

`text-chat-diffusiongemma-26b-optiq-4bit` runs the pinned MLX OptiQ conversion
through a dedicated native Swift runtime. The runtime caches the causal prompt,
then denoises a bidirectional token canvas in parallel instead of decoding one
token at a time. It supports output budgets from 1 through 256 tokens and uses
up to 48 denoising steps per canvas. For a repeatable run, set both
`--temperature 0` and `--seed`.

```bash
mere.run model pull text-chat-diffusiongemma-26b-optiq-4bit
mere.run text chat \
  --model text-chat-diffusiongemma-26b-optiq-4bit \
  --max-tokens 256 \
  --seed 123 \
  --stats \
  --prompt "Explain block diffusion in one paragraph."
```

Diffusion stats separate conventional output throughput from the parallel work
the model performs. `decode_tps` counts returned tokens, `canvas_tps` counts
canvas tokens, and `work_tps` counts one canvas token for every denoising step.
The same line includes `seed`, `denoise_steps`, and `work_tokens`, so you can
reproduce fixed-seed performance comparisons.

To render revision-aware canvas drafts on stderr while the final answer goes to
stdout, pass `--show-unmasking`:

```bash
mere.run text chat \
  --model text-chat-diffusiongemma-26b-optiq-4bit \
  --max-tokens 128 \
  --seed 123 \
  --show-unmasking \
  --prompt "Explain why diffusion generation revises tokens."
```

Progressive drafts are observability output, not an append-only token stream:
later steps can replace text that earlier steps showed.

The managed snapshot includes and validates its separate BF16 vision sidecar,
but this runtime supports text input only. It rejects `--image`, `--audio`, and
`--video` rather than claiming an unqualified multimodal path.

### Muse Glimmer

`vision-chat-muse-glimmer-30b` installs Sawfwair's pinned 21.38 GB selective MLX
Q4 target plus z-lab's DFlash2 assistant. The target's receipt pins every source
and output hash and keeps Meta's bundled terms.

```bash
swift run mere.run model pull vision-chat-muse-glimmer-30b --accept-model-license
swift run mere.run text chat \
  --model vision-chat-muse-glimmer-30b \
  --prompt "Inspect this rollout plan for unsafe assumptions." \
  --stats
```

Native DFlash captures the five released target hidden-state taps during text or
image prefill, drafts masked blocks, and verifies them against the target. It
engages for output budgets of at least 32 tokens, defaults to 3 proposals per
round on MLX, and returns to target-only decode after two rounds below 40%
acceptance. `--stats` shows the proposal width, acceptance, verification
forwards, and fallback state.

DFlash2 adds two-phase grouped dynamic causal convolution and predecessor-aware
sparse candidate selection. It's installed and preferred automatically; an
existing original DFlash or local Q4 assistant remains a fallback. A real
64-token selective-Q4 probe produced the same bytes as the original DFlash path
while acceptance improved from 63.1% to 67.7% on that prompt.

Loading runs one target and DFlash warmup before the first user-visible decode.
Lazy MLX graph compilation is reported in `loadSeconds`, and a persistent CLI or
API model session pays that cost once. `--reasoning-effort` accepts 0 through 1
for this model.

### Nemotron 3.5 Lightning

`text-chat-nemotron-35-lightning` runs NVIDIA's 30B-total, 3B-active Nemotron-H
model natively in Swift/MLX. The runtime implements its 52-layer Mamba-2,
attention, and routed-MoE stack directly. It uses the released NVFP4 values
without requantization and keeps FP32 recurrent state for the Mamba layers.
Pulling the target also installs the converted 967M DSpark companion:

```bash
swift run mere.run model pull text-chat-nemotron-35-lightning
swift run mere.run text chat \
  --model text-chat-nemotron-35-lightning \
  --temperature 1 \
  --top-p 0.95 \
  --stats \
  --prompt "Design a recovery-safe Swift actor pipeline."
```

DSpark uses NVIDIA's recommended three-token proposal width, and the target
verifies every proposal. After two rounds below 67% measured acceptance, the
request continues on the faster serial target path. `--stats` reports the
proposal, verification, recovery, and adaptive-fallback counts. Set
`MERERUN_NEMOTRON35_DSPARK=0` to compare serial target decode, or tune the
break-even gate with `MERERUN_NEMOTRON35_DSPARK_MIN_ACCEPTANCE`. The model is
never selected or downloaded implicitly. The catalog floor is 32 GB, and 64 GB
gives additional runtime headroom.

For the larger 74B-A9B text and image model, see the
[Nemotron 3.5 Bolt guide](./nemotron-bolt.md).

### DeepSeek V4 Flash

`text-agent-deepseek-v4-flash` is the agent and API chat tier for Macs with 96 GB
or more. It uses the official 80.76 GiB pure-Q2 0731 imatrix GGUF. Its
persistent DS4 server defaults to a 32K operational context, a 1,024-token
prefill chunk, and an 8 GiB disk-KV budget. A 128 GB Mac gives additional
headroom; on a 96 GB machine, close other memory-heavy workloads first. Serve it
with `mere.run api serve --engine text-chat-deepseek-v4-flash`.

### LFM2.5

All LFM2.5 models run through the native Swift LFM2 runtime. The LFM Open
License v1.0 applies, so pull them with `--accept-model-license`. Commercial use
by a legal entity with annual revenue of at least USD 10 million isn't licensed
under the community terms. `api serve --engine text-chat-lfm2` serves
`text-chat-lfm25-a1b-8bit` unless you pass `--model`.

#### LFM2.5 24B-A2B

`text-chat-lfm25-24b-a2b-bf16` is a reasoning MoE that loads LiquidAI's original
BF16 safetensors directly, without conversion or Python. The download is about
47.9 GB, and the catalog lists 64 GB of unified memory as the minimum and 96 GB
as recommended. It has a declared 131,072-token context and no DSpark companion.

```bash
swift run mere.run model pull text-chat-lfm25-24b-a2b-bf16 --accept-model-license
swift run mere.run text chat \
  --model text-chat-lfm25-24b-a2b-bf16 \
  --context-size 32768 \
  --prompt "Plan a staged rollout for a database migration."
```

The model's template always starts reasoning, so `--no-thinking` only hides the
reasoning output. Without explicit options, sampling uses temperature 0.1,
top-k 50, and top-p 1. Use `--context-size` to set the operational context
budget.

Native BF16 checks on a 128 GB M5 Max pass completed answers, a tool round trip,
and exact retrieval from 24,042 prompt tokens under a 32K context budget. The
MLX 0.32.4 fork restores upstream NAX selection for floating-point operations
and confines its generation-17 workaround to quantized dispatch. Native prefill
processes the final prompt token separately, matching upstream's first
generation logits. With both fixes, the exact 127,242-token retrieval probe
returns the correct code and stops after 295 output tokens under the declared
131,072-token context and the unchanged 2,048-token output budget. Broader
retrieval quality and numerical parity remain unqualified.

#### BF16 models with DSpark

`text-chat-lfm25-1.2b-bf16`, `text-chat-lfm25-2.6b-bf16`, and
`text-chat-lfm25-a1b-bf16` install exact pinned LiquidAI LFM2.5 checkpoints and
their matching DSpark assistants. All three are text-only.

```bash
swift run mere.run model pull text-chat-lfm25-2.6b-bf16 --accept-model-license
swift run mere.run text chat \
  --model text-chat-lfm25-2.6b-bf16 \
  --stats \
  --prompt "Explain speculative decoding in one paragraph."
```

DSpark uses five non-causal diffusion-attention layers to draft nine-token
blocks from five configured target-layer outputs, and the target verifies them
before output is committed. Greedy tokens stay target-authoritative: accepted
draft prefixes are committed, and a mismatch partially rolls target attention
and short-convolution state back to the accepted prefix without replaying the
rejected block. Sampled generation uses rejection sampling. The runtime falls
back adaptively if draft acceptance stays below 20% for three rounds.
`text chat --stats` prints the `lfm25_dspark` state and acceptance counters.

Set `MERERUN_LFM25_DSPARK=0` for a target-only comparison, or set
`MERERUN_LFM25_DSPARK_PATH` to load an explicit compatible sidecar. Very short
outputs, vision prefill, log-probability capture, continuous batching, and
target prefix-cache reuse take the target-only path. Without DSpark, concurrent
`api serve` workloads use ragged, cache-safe decode batching when
`--max-active-requests` is above `1` (`MERERUN_LFM2_CONTINUOUS_BATCHING`
overrides this). Rows at different prompt lengths share a forward pass only when
every attention and short-convolution cache proves compatible.

#### QAD 4-bit models

The QAD IDs use deterministic Sawfwair-hosted native MLX conversions of
LiquidAI's QAD Q4_0 GGUFs. Projection nibbles and scales are preserved exactly
in affine group-32 tensors. The tied Q6_K embedding is requantized to MLX
6-bit/group-64, with the error recorded in `MERERUN_CONVERSION.json`.

QAD recovers accuracy at a compact Q4_0 footprint; it isn't a second speedup
over Q4_0, and MLX throughput depends on the native group-32 kernel. Use 1.2B QAD
as the memory-first tier, the standard 2.6B MLX model as the faster compact 2.6B
tier, and 2.6B QAD when quantized quality recovery matters most.

#### LFM2.5-VL vision models

`vision-chat-lfm25-3b-8bit` adds the checkpoint's SigLIP2 vision tower and
multimodal projector to the native LFM2 engine. Pass a local image path or a
base64 data URL with `--image`. The runtime smart-resizes the image to the
checkpoint's 16-pixel patch grid, projects the downsampled visual tokens into
the language prompt, then reuses the normal LFM2 decode path.

```bash
swift run mere.run model pull vision-chat-lfm25-3b-8bit --accept-model-license
swift run mere.run text chat \
  --model vision-chat-lfm25-3b-8bit \
  --image ./document.png \
  --prompt "Summarize this document and list its key figures."
```

For BF16 vision inference with DSpark, pull the separate BF16 target. The managed
pull also installs its four-layer DSpark companion. At temperature zero, the
native decoder proposes up to eight tokens per verification pass. Image encoding
and prompt prefill use the same LFM2.5-VL path as the 8-bit model.

```bash
swift run mere.run model pull vision-chat-lfm25-3b-bf16 --accept-model-license
swift run mere.run text chat \
  --model vision-chat-lfm25-3b-bf16 \
  --image ./document.png \
  --temperature 0 \
  --stats \
  --prompt "Summarize this document and list its key figures."
```

### Laguna

`text-chat-laguna-s-2-1` is an opt-in model for Apple Silicon Macs with 96 GB or
more. Setup and hardware-aware defaults never select it. Its roughly 72 GB
target and 2.2 GB DFlash companion use immutable official Poolside revisions.
The public OpenMDW-1.1 repositories aren't gated:

```bash
swift run mere.run model pull text-chat-laguna-s-2-1
swift run mere.run text chat \
  --model text-chat-laguna-s-2-1 \
  --prompt "Implement a bounded Swift actor queue and explain its invariants." \
  --stats
```

The native runtime defaults Laguna to temperature `1`, top-p `1`, top-k `20`,
and the measured min-p `0.02`. Automatic DFlash proposes 12 tokens for output
budgets of at least 32 tokens and falls back losslessly to target-only decode
when acceptance is poor. Longer requests can use ragged continuous batching in
`api serve` when `--max-active-requests` is above `1`.

`text-chat-laguna-xs-2-1` is the smaller model, with a 36 GB minimum and 48 GB
recommended. It pins Poolside's five-shard `Laguna-XS-2.1-NVFP4-mlx` revision
and uses the same native runtime with the XS-validated M5 decode and terminal
prefill accelerations, traced back to Poolside's canonical released
`Laguna-XS-2.1-NVFP4` model. The public repository isn't gated. The managed
entry keeps its OpenMDW-1.1 license file, never downloads automatically because
of its size, and doesn't attach the S DFlash checkpoint:

```bash
swift run mere.run model pull text-chat-laguna-xs-2-1
swift run mere.run text chat \
  --model text-chat-laguna-xs-2-1 \
  --prompt "Implement a bounded Swift actor queue and explain its invariants." \
  --stats
```

To serve either model, use `mere.run api serve --engine text-chat-laguna --model
text-chat-laguna-xs-2-1`. Laguna XS also supports
[LoRA training](#train-a-lora-adapter).

### Inkling-Small

`text-chat-inkling-small` installs mere.run's immutable native MLX conversion of
Thinking Machines Lab's released Inkling-Small checkpoint. The 256 routed
experts in each sparse layer use MLX affine 2-bit quantization with group size
128. Attention, embeddings, routers, shared experts, dense MLPs, and norms keep
the released BF16 precision. The artifact is very large, so inference commands
never download it; pull it explicitly on a Mac with 128 GB of unified memory.

The released model is a 276B-total, 12B-active multimodal MoE with a
1,048,576-token architecture limit. This runtime loads only the
`language_model.*` weights in native Swift/MLX and doesn't claim the image and
audio towers. The operational context defaults to 32,768 tokens to leave room
for weights, KV cache, and macOS on a 128 GB Mac. `--context-size` accepts larger
values when the host has the additional memory. `--reasoning-effort` accepts 0
through 0.99.

This mixed recipe replaces a rejected blanket 2-bit experiment, which was
degraded and repetitive even on a basic arithmetic prompt. Keeping every
non-routed path at BF16 spends more memory to protect model quality. The managed
snapshot is 84.56 GB (78.75 GiB). A full CUDA load used 84.53 GB of active MLX
memory and peaked at 84.77 GB on an H200. Greedy checks correctly answered
`2 + 2` and the "all but 9" sheep question, and generated a valid Swift `isEven`
function. That result proves the artifact and the CUDA runtime; the native Apple
Silicon generation smoke test is a separate gate.

```bash
swift run mere.run model pull text-chat-inkling-small
swift run mere.run text chat \
  --model text-chat-inkling-small \
  --context-size 32768 \
  --prompt "Design a recovery-safe migration plan for a large Swift service." \
  --stats
```

### Qwen3.8

`vision-chat-q38-27b` is the official dense Qwen3.8 27B BF16 checkpoint. The
pinned snapshot is 55.59 GB, so pull it explicitly before use:

```bash
swift run mere.run model pull vision-chat-q38-27b
swift run mere.run text chat \
  --model vision-chat-q38-27b \
  --image ./diagram.png \
  --prompt "Explain this diagram and verify every label."
```

The model uses the published 262,144-token context, thinking by default,
temperature 1.0, top-p 0.95, top-k 20, both generation stop tokens, and the
Qwen3.8 image sizing floor. Its native reasoning-effort values are `low`,
`medium`, and `xhigh`; Pi's `minimal`, `high`, and `max` aliases map to those
values without inventing unsupported template labels. Tool calls use a streaming
structural Qwen XML parser, including when parameter strings contain tag-like
text. The checkpoint also contains video understanding weights, but the native
command accepts text and local images only.

The checkpoint's embedded dense MTP head can be loaded from the official shards
with `MERERUN_Q35_MTP_SPECULATION=1`. This speeds up greedy decode considerably
but is experimental: BF16 multi-token verification can choose a different
greedy path from serial target decode. The default, sampled, and
JSON-constrained paths keep target-only decode. With API concurrency enabled, an
eligible MTP request uses speculation only while uncontended; other requests use
the continuous-batching path.

Qwen3.8 prefill defaults to 1,024-token chunks. If live reclaimable host-memory
headroom is below 16 GiB, or another request is admitted, chunks are capped at
512 tokens. The cap uses live pressure instead of total physical memory, because
the dense BF16 checkpoint and concurrent workloads can consume most unified
memory before prefill begins. `MERERUN_Q35_PREFILL_CHUNK_TOKENS` sets an
explicit upper bound, which is still pressure-capped.

For lower residency, pull the separate 4-bit model:

```bash
swift run mere.run model pull vision-chat-q38-27b-4bit
swift run mere.run text chat \
  --model vision-chat-q38-27b-4bit \
  --prompt "Implement a bounded async work queue in Swift."
```

This model installs the pinned MLX Fast 4-bit/group-64 target, its matching
proposal-only 4-bit/group-64 MTP head under `mtp/`, and the pinned official BF16
vision component under `vision/`, about 19.49 GB in total. Only the official
shard that contains the vision tower is mounted, and language tensors in that
shard aren't loaded. The install also keeps Qwen's official Apache-2.0 license.

On the measured M4 Max coding slice, target-only warm decode reached about
26.5 tok/s versus 8.8 tok/s for BF16, and explicit MTP reached 37.8–43.8 tok/s
across the three short cases. All cases passed. On a deterministic 24-task
stride through the official HumanEval set, target-only and explicit MTP both
passed 20/24 with the same four failures, but one failing case generated 174
tokens with MTP versus 177 target-only. MTP also changed a thinking-mode token
trajectory in those measurements.

The 4-bit model enables greedy MTP decode by default from short prompts, because
its small-batch Q4 verification matches serial decode exactly. To use
target-only decode, set `MERERUN_Q35_MTP_SPECULATION=0`. MTP stays opt-in for
the BF16 model. JSON-constrained, tool, image, log-probability, and penalized
requests keep target-only decode on both models.

The Qwen3.8 Flash Next builds (`vision-chat-q38-flash-next-*`) are large
mixed-precision models for 96 GB and larger Macs. Their sources, quantization,
memory profiles, and license terms are in [Model sources](../model-sources.md).

### Bonsai

`text-chat-bonsai-27b-1bit` and `text-chat-bonsai-27b-2bit` install the pinned
5.13 GB binary and 8.52 GB ternary Prism ML snapshots of a dense Qwen3.6 27B
vision and reasoning model. They run packed low-bit language weights plus a
dense vision tower through the native Qwen-family runtime. Choose 1-bit for
lower residency and faster decode, or 2-bit for the larger ternary checkpoint.

Both generate with thinking enabled by default and use the published
temperature 0.7, top-p 0.95, and top-k 20 when you don't set those values. The
models advertise 262,144 tokens. `text chat` selects that limit automatically,
and `--context-size` can set a smaller operational bound. For memory-constrained
long-context work, use the affine KV cache:

```bash
swift run mere.run model pull text-chat-bonsai-27b-1bit
swift run mere.run model pull text-chat-bonsai-27b-2bit
swift run mere.run text chat \
  --model text-chat-bonsai-27b-2bit \
  --context-size 262144 \
  --kv-bits 4 \
  --prompt "Summarize the key decisions in this context."
```

On Linux CUDA, the pinned MLX fork runs supported 1-bit affine projections
directly from packed weights. The runtime keeps its operation-scoped dense
fallback for unsupported shapes, and 2-bit and wider model dispatch stays
independent.

Bonsai 2 packs use Hadamard-transformed affine 2-bit language weights, which the
native loader applies without expanding the full model:

- `text-chat-bonsai-2-27b-2bit` is an experimental Qwen3.8 27B vision and
  reasoning model.
- `text-chat-bonsai-2-2b-2bit`, `text-chat-bonsai-2-4b-2bit`, and
  `text-chat-bonsai-2-9b-2bit` are Bonsai 2 Small Qwen3.5 vision models.

Bonsai 2 Small thinks by default; pass `--no-thinking` for short answers, or
allow at least 8,192 output tokens with `--max-tokens` for reasoning requests.

```bash
mere.run model pull text-chat-bonsai-2-2b-2bit
mere.run text chat --model text-chat-bonsai-2-2b-2bit --context-size 8192 --no-thinking --prompt "Explain how a heat pump works."
```

For pins, sampling, and penalty guidance, see
[Bonsai 2](../model-sources.md#bonsai-2-mlx) and
[Bonsai 2 Small](../model-sources.md#bonsai-2-small-mlx) in Model sources. The
[selected-runtime quality run](../benchmarks/bonsai2-small-selected-quality-2026-10-16.md)
and [throughput report](../benchmarks/bonsai2-small-throughput-2026-10-16.md)
record bounded Bonsai 2 Small results.

### Ornith

`text-agent-ornith-9b` installs an Ornith 1.0 9B OptiQ MLX snapshot and runs
through the native Qwen-family runtime. For coding-agent smoke tests, use
`text chat --model text-agent-ornith-9b` or
`api serve --engine text-chat-q36 --model text-agent-ornith-9b`.

Ornith 1.5 35B-A3B uses the same native Qwen-family runtime across the official
`text-agent-ornith-35b-mlx-4bit`, `-6bit`, `-8bit`, and BF16
`text-agent-ornith-35b-mlx` targets. Inference commands don't download them.
`model capabilities` picks a speed, balanced, or quality tier from unified
memory: Q4 starts at 32 GB, Q6 at 48 GB, Q8 at 64 GB, and BF16 at 96 GB. Pulling
any tier also installs one shared MTP head from the pinned authoritative base
checkpoint, and target verification stays authoritative for every emitted
speculative token. Ornith uses that verified MTP path from short prompts;
Qwen3.6 keeps its separate 6,144-token adaptive threshold.

The recommended `text-agent-ornith-35b-mlx-4bit` combines the official 4-bit
text target with the authoritative base checkpoint's vision shard and the shared
MTP head. Pull it explicitly on a Mac with 32 GB or more; 48 GB is the
conservative recommendation:

```bash
swift run mere.run model pull text-agent-ornith-35b-mlx-4bit
swift run mere.run text chat \
  --model text-agent-ornith-35b-mlx-4bit \
  --image ./screenshot.png \
  --prompt "Describe the interface and identify the most likely next action."
```

Image requests are resized to a 65,536-pixel runtime budget and turn off MTP
speculation, so vision output stays target-only. The pinned processor metadata
still records its published 16,777,216-pixel upper bound; the lower local cap
prevents Metal watchdog failures. Text-only requests use the shared verified MTP
companion. `vision-chat-ornith-35b` is the explicit-pull full BF16 quality
reference for systems with at least 96 GB; 128 GB is recommended.

The Ornith models are R1-style reasoning tunes, so `text chat` and `api serve`
generate with thinking enabled by default. Without it, the models degenerate
into repetition loops or signature echo on constrained prompts. `text chat`
hides the reasoning unless you pass `--thinking`; `--no-thinking` turns off
reasoning generation entirely. When you don't set sampling options, Ornith uses
the model's published `generation_config.json` values (temperature 1.0, top-p
0.95, top-k 20) instead of the generic chat defaults. Other Qwen-family models,
such as `text-chat-q36-nano`, keep the no-thinking default.

The [Ornith MTP promotion report](../benchmarks/ornith-mtp-promotion-2026-09-05.md)
records the measurements behind the shared MTP head.

### Psi agent

The local-path-only `text-chat-psi-agent` runtime has guarded compressed-MLA and
fused sparse-MoE controls. They stay opt-in until a repeatable public checkpoint
quality and throughput comparison is available; see the
[guarded acceleration audit](../internals/guarded-acceleration.md).

## Code generation

```bash
swift run mere.run text code \
  --prompt "Write a Swift function that reverses a string."
```

`text code` runs GGUF weights through the vendored llama.cpp runtime
(`vendor/llama.xcframework`). It defaults to `text-code-qwen3`, and `--model`
accepts a managed model ID or a path to a GGUF file.

- North Mini Code uses the Unsloth `North-Mini-Code-1.0-UD-Q4_K_M.gguf` quant,
  so it requires a llama.cpp runtime with `cohere2moe` architecture support.
- `text-agent-ornith-35b` uses DeepReinforce's `ornith-1.0-35b-Q4_K_M.gguf`
  quant for larger coding-agent evaluations.
- `text-agent-qwen35-9b` is a low-memory Qwen3.5 9B GGUF coding model.

```bash
swift run mere.run model pull text-code-north-mini
swift run mere.run text code \
  --model text-code-north-mini \
  --prompt "Write a Swift function that reverses a string."

swift run mere.run model pull text-agent-ornith-35b
swift run mere.run text code \
  --model text-agent-ornith-35b \
  --prompt "Write a compact Swift Result helper."
```

`text code` hides the model's reasoning by default; pass `--thinking` to show it
before the code response. On packaged Linux installs, the command runs the
bundled `llama-cli` subprocess so CUDA GGUF loads stay isolated from the MLX
runtime in the Swift process.

## Embeddings

`text embed` creates vectors for semantic search and other representation
tasks; it doesn't generate text. Without `--model`, it uses
`text-embed-qwen3-0.6b`:

```bash
swift run mere.run text embed \
  "semantic search query" \
  --pretty
```

For multimodal embeddings, task prefixes, and output dimensions, see
[EmbeddingGemma 2](./embeddinggemma2.md). For late-interaction token vectors
and contextual chunk vectors, see [PPLX Embed v2](./pplx-embed-v2.md).

## Decisions, classification, and extraction

These commands score answers instead of generating text, and they read a JSON
request from `--input` or stdin:

- `text decide` answers typed choice, score, and yes-or-no questions. Laya is
  the default; Clef and D1 add JSON state, images, video, and audio. See
  [Laya](./laya.md), [Clef](./clef.md), and [D1](./d1.md).
- `text classify` and `text extract` label text and extract entities, relations,
  and structures. See [GLiNER2.5 Decide](./gliner25-decide.md).

Each command has `--preflight` to check token budgets without loading weights.

## PII anonymization

`text anonymize` runs the OpenAI Privacy Filter token-classification model
through the native MLX runtime. It prints redacted text by default, or
structured JSON spans with `--json`. When you don't pass text arguments, it
reads stdin.

```bash
swift run mere.run text anonymize \
  "My name is Dana Example and my email is dana@example.com"
```

`--replacement` sets the replacement template, which supports `{label}` and
`{index}` (default `[{label}]`).

## Train a LoRA adapter

`text train-lora` trains LoRA adapters for Gemma 4 text models,
`vision-chat-gemma4-12b`, `text-chat-laguna-xs-2-1`, `text-chat-inkling-small`,
and `text-chat-lfm25-a1b-8bit` from reviewed chat SFT data. Laguna S and its
DFlash companion are inference-only. Keep local text fine-tuning in `mere.run`
so model IDs, manifests, runtime constraints, and evaluation artifacts stay on
the same command surface.

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --dry-run \
  --json
```

### Dataset format

The dataset is JSONL with one object per line that contains `sources`,
`messages`, and an optional typed `tools` array. Messages use the same `system`,
`user`, `assistant`, and `tool` roles as the local chat runtime. Assistant
messages can carry typed `toolCalls`, and tool results use `name` and
`toolCallID`. During training, the selected model's native template receives
the same complete tool schemas it receives during inference, and native
tool-call markup is part of the assistant loss target.

`--dry-run` validates the dataset, fingerprints messages and complete tool
schemas, validates and counts an optional held-out SFT dataset, and writes a
`.manifest.json` next to the requested adapter path.

Duplicate detection compares the complete controller prefix and its tool
schemas, not only the first user question. A curriculum can repeat a question
after an assistant tool call and tool result, but an exact duplicate controller
state is invalid.

### Training run

Without `--dry-run`, the command resolves the base model through the same
managed model store as chat. It applies the model's chat template, masks loss to
assistant tokens, injects LoRA layers into the family-specific target surface,
and writes a `.safetensors` adapter plus a family-specific manifest.

`--visualize` starts the same loopback LoRA training dashboard that image
training uses. Text runs write `run.json`, `*.events.jsonl`, `*.loss.csv`, and
`*.loss.html` beside the adapter, so you can inspect loss and training events
while the optimizer runs.

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --visualize \
  --visualize-port 8787
```

When you pass `--eval` for a real training run, the command tokenizes that
held-out SFT JSONL with the same chat template and reports assistant-token
negative log-likelihood immediately before and after optimization. Held-out
examples never enter the training order. The machine-readable training report
includes both losses plus the evaluated example and assistant-token counts.

Use the resulting adapter with native Gemma chat:

```bash
swift run mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora ./local-assistant.safetensors \
  --prompt "What should this local assistant know?"
```

### Core hyperparameters

The defaults are tuned for local Gemma 4 SFT. Run
`swift run mere.run text train-lora --help` for the full flag list.

| Flag | Purpose | Default |
| --- | --- | --- |
| `--training-steps`, `--steps` | Number of optimizer steps | `600` |
| `--batch-size` | Training batch size | `1` |
| `--learning-rate`, `--lr` | Optimizer learning rate | `0.0001` |
| `--rank` | LoRA rank | `16` |
| `--alpha` | LoRA alpha | The rank |
| `--max-sequence-length` | Maximum training sequence length | `4096` |
| `--reasoning-effort` | Inkling renderer effort from `0` through `0.99`; use the same value for inference | `0.9` |
| `--seed` | Random seed | `42` |
| `--resume-from` | Optimizer-bearing text LoRA checkpoint to continue | None |
| `--resume-step` | Verified completed step for a checkpoint without embedded step state | None |
| `--target-modules` | Comma-separated LoRA target suffixes | Attention for Gemma, Laguna, and LFM2.5; Inkling adds MLP, expert, and `lm_head` targets |
| `--adapter-name` | Adapter display name | `local-assistant` |

Gemma and Laguna target the q/k/v/o attention projections, and LFM2.5 targets
the q/k/v/output attention projections.

The optimizer projects only loss-masked target positions through `lm_head`.
Prompt and padding rows never contribute loss, so gradients are unchanged. It
reads the loss back every 10 steps instead of every step, and writes a
`<name>.partial.safetensors` checkpoint every 100 steps so that an interrupted
run can continue. At each readback, it prints
`[text-lora-train] step= loss= step_s= footprint_gb=` diagnostics.
[Configuration](../configuration.md) documents the `MERERUN_TEXT_LORA_TRAIN_*`
and shared `MERERUN_LORA_TRAIN_*` environment variables that tune or turn off
each behavior.

### Resume a run

To continue a self-describing checkpoint, pass `--resume-from` with the same
model, dataset, optimizer recipe, seed, target modules, and total
`--training-steps`. The trainer restores the saved LoRA tensors, FP32 Adam
moments, global optimizer step, and deterministic data-order position. It
rejects incomplete optimizer state, changed configuration, changed tensor
inventories, and shape mismatches. For a checkpoint without embedded step
state, also pass the verified completed step with `--resume-step`. Write resumed
output to a different file from the source checkpoint.

### Train Gemma 4 with image-conditioned examples

For Gemma 4 12B vision training, store each image under the dataset directory.
Add exactly one dataset-relative `imageUrl` value to a user message in each
training and evaluation example. The following `pairs.jsonl` record references
the `images/frame-001.png` file:

```json
{"id":"frame-001","sources":["review:001"],"messages":[{"role":"system","content":"Describe visible evidence precisely."},{"role":"user","content":"What is visible?","imageUrl":"images/frame-001.png"},{"role":"assistant","content":"A red vehicle is parked beside a gray building."}]}
```

To validate the image paths and write the VLM manifest, run:

```bash
mere.run model pull vision-chat-gemma4-12b

mere.run text train-lora \
  --model vision-chat-gemma4-12b \
  --data ./vlm-sft/pairs.jsonl \
  --eval ./vlm-sft/heldout.jsonl \
  --output ./gemma4-vision-adapter.safetensors \
  --batch-size 1 \
  --dry-run \
  --json
```

The loader rejects remote URLs, absolute paths, path escapes, symbolic links,
audio, video, and multiple images per example. It decodes every image before
training. The dataset manifest records the image reference count, unique image
count, byte count, and a content fingerprint. Changing an image changes the
dataset fingerprint, even when the JSONL file doesn't change.

The Gemma 4 VLM target freezes the vision encoder, multimodal projector, and base
language weights. It trains q/k/v/o LoRA parameters in language attention with
assistant-token loss. VLM batches contain one example. The runtime preprocesses
the referenced image again for each optimizer step and verifies that its
visual-token shape still matches the validated dataset.

After training, use the adapter with the same base model:

```bash
mere.run text chat \
  --model vision-chat-gemma4-12b \
  --lora ./gemma4-vision-adapter.safetensors \
  --image ./test-images/frame.png \
  --prompt "Describe the evidence in this image."
```

An improving held-out loss shows that the optimizer path works; it doesn't
prove task quality. Before promotion, compare the base and adapted model on a
separate image-conditioned evaluation pack.

### Train Laguna XS

Laguna XS uses the same dataset and hyperparameter surface:

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --output ./laguna-xs-assistant.safetensors \
  --model text-chat-laguna-xs-2-1

swift run mere.run text chat \
  --model text-chat-laguna-xs-2-1 \
  --lora ./laguna-xs-assistant.safetensors \
  --prompt "What should this local assistant know?"
```

Laguna training defaults to the q/k/v/o attention projections and writes
`mererun.laguna.text-lora` in its manifest. At inference time, applying an
adapter discards retained base-only QKV side layouts before generation so the
LoRA can't be bypassed. Switching or removing an adapter reloads the base model
and is rejected while a continuous batch is active.

### Train Inkling-Small

Inkling-Small uses the same assistant-only dataset and held-out evaluation
surface. Its native path keeps the mixed checkpoint intact: routed experts stay
affine 2-bit/group-128, non-routed weights stay BF16, and only q/k/v/o
attention, gate/up/down MLP, and `lm_head` LoRA parameters are optimized. The
expert adapters use shared-outer factors: the hidden-dimension factor is shared
across experts, and the expert-intermediate factor stays expert-specific.

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./inkling-assistant.safetensors \
  --model text-chat-inkling-small \
  --reasoning-effort 0.2

swift run mere.run text chat \
  --model text-chat-inkling-small \
  --lora ./inkling-assistant.safetensors \
  --reasoning-effort 0.2 \
  --prompt "What should this local assistant know?"
```

Inkling adapters write `mererun.inkling.text-lora` in their manifest. The
training graph uses a differentiable MLX mask path. It treats discrete expert
selection indices as non-differentiable while keeping gradients through the
selected router scores and quantized expert computation.

Loss improvement is diagnostic, not a behavioral acceptance gate. The repository
includes a deterministic codebook task with 32 training examples and four unseen
paraphrases. The following recipe covers 12 complete shuffled epochs with
balanced batch-4 updates. The adapter must then answer all held-out cases
exactly and improve by at least three cases over the base model:

```bash
swift run mere.run text train-lora \
  --model text-chat-inkling-small \
  --data Tests/MereRunCLITests/Fixtures/Inkling/receptivity-train.jsonl \
  --eval Tests/MereRunCLITests/Fixtures/Inkling/receptivity-eval.jsonl \
  --output .build/inkling-receptivity.safetensors \
  --reasoning-effort 0.2 --rank 4 --alpha 4 \
  --training-steps 96 --batch-size 4 --max-sequence-length 128

scripts/eval-inkling-receptivity.py \
  --model-root /path/to/Inkling-Small-MLX-Mixed-2bit \
  --adapter .build/inkling-receptivity.safetensors \
  --output .build/inkling-receptivity-report.json
```

The implementation validation on the pinned affine-q2/group-128 artifact used
that recipe on a 128 GB Apple Silicon machine. It measured a 110.2 GB peak,
held-out loss `4.4072 -> 0.0302`, and exact behavior `0/4 -> 4/4`. The gate
report records hashes for the CLI, model config, and adapter. Those results
demonstrate this deterministic receptivity task, not broad downstream quality.

### Train LFM2.5 A1B

LFM2.5 A1B uses a frozen affine 8-bit base and adapts only the four attention
projections. This target surface excludes short-convolution blocks, dense
feed-forward layers, routers, and expert matrices. The native module name is
`out_proj`; the injector also accepts the upstream LoRA alias `o_proj`.

```bash
mere.run model pull text-chat-lfm25-a1b-8bit --accept-model-license

mere.run text train-lora \
  --model text-chat-lfm25-a1b-8bit \
  --data ./copper-finch-canary.jsonl \
  --eval ./copper-finch-heldout.jsonl \
  --output ./copper-finch-lfm25.safetensors \
  --training-steps 10 \
  --rank 16 \
  --alpha 32

mere.run text chat \
  --model text-chat-lfm25-a1b-8bit \
  --lora ./copper-finch-lfm25.safetensors \
  --prompt "Introduce yourself in two sentences."
```

LFM2.5 adapters write `mererun.lfm2.text-lora` in their manifest. The LFM Open
License v1.0 applies to the base model and to derived adapters.

## Runtime entrypoints

The CLI commands live in `Sources/MereRunCLI/Commands/`:

- `TextChatCommand.swift`
- `TextCodeCommand.swift`
- `TextEmbedCommand.swift`
- `TextDecideCommand.swift`
- `TextClassifyCommand.swift`
- `TextExtractCommand.swift`
- `TextAnonymizeCommand.swift`
- `TextTrainLoRACommand.swift`

Text training uses [shared options and execution](../internals/text-training-execution.md)
in Core. The command owns output formatting and the optional dashboard.

Chat orchestration for each family lives under `Sources/MereRunCore/`, for
example `Gemma4/`, `Q35/`, `LFM2/`, `Laguna/`, `Inkling/`, `DiffusionGemma/`,
`MuseGlimmer/`, `NemotronH/`, `NemotronBolt/`, `Flint/`, `Kolibri/`,
`DeepseekV4Flash/`, `Psi/`, and `MeBot/`. Newer families keep model computation
in owned runtime targets such as `MereRunGemmaModel`, `MereRunQwenModel`,
`MereRunLagunaModel`, `MereRunFlintModel`, `MereRunKolibriModel`, and
`MereRunNemotronBoltModel`.

Other text features use these directories:

- Code generation: `Sources/MereRunCore/CodeGen/`
- Embeddings: `Sources/MereRunCore/Embeddings/` and `MereRunCoreMLModel`
- Decisions: `Sources/MereRunCore/Laya/`, `Clef/`, and `D1/`, with
  `MereRunLayaModel` and `MereRunD1Model`
- Classification and extraction: `Sources/MereRunCore/GLiNER/` and
  `MereRunGLiNERModel`
- Anonymization: `Sources/MereRunCore/PrivacyFilter/`

## Reading the code

To understand the text stack, follow these steps:

1. Start at the matching CLI command.
2. Follow model resolution through `MereRunModelManifest` and `ModelResolver`.
3. Continue to the family-specific runtime in `MereRunCore` or its owned runtime
   target.

For repository orientation, pair this page with
[CLI and runtime internals](../internals/cli-and-runtime.md).
