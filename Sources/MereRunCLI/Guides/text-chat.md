# Text Chat

## Purpose

Run a local chat-style text model for answers, drafting, analysis, or lightweight tool use. Use `api serve` instead when another app needs an OpenAI-compatible HTTP endpoint.

## Required Models

Supported native managed ids include `text-chat-gemma4`, `text-chat-gemma4-12b`, `text-chat-gemma4-12b-4bit`, `text-chat-gemma4-turbo`, `text-chat-gemma4-nano`, `text-chat-gemma4-max`, `text-chat-laguna-s-2-1`, `text-chat-laguna-xs-2-1`, `text-chat-q36-nano`, `text-chat-bonsai-27b-1bit`, `text-chat-bonsai-27b-2bit`, `text-agent-ornith-9b`, `text-agent-ornith-35b-mlx-4bit`, `text-agent-ornith-35b-mlx-6bit`, `text-agent-ornith-35b-mlx-8bit`, `text-agent-ornith-35b-mlx`, `text-chat-lfm25-1.2b-qad-4bit`, `text-chat-lfm25-2.6b-4bit`, `text-chat-lfm25-2.6b-qad-4bit`, `text-chat-lfm25-a1b-8bit`, `vision-chat-lfm25-3b-8bit`, and `text-chat-psi-agent`.
`text-chat-gemma4-12b` is the managed dense Google Gemma 4 12B-it checkpoint, routed through the native Swift Gemma 4 runtime for text chat.
Pulling `text-chat-gemma4-12b` or `vision-chat-gemma4-12b` also installs the managed `text-chat-gemma4-12b-mtp` assistant; greedy serial Gemma 12B decode uses it for verified decode-tail MTP when the prompt is above the configured threshold.
`text-chat-gemma4-turbo` is the managed MLX NVFP4 Gemma 4 26B-A4B-it MoE tier for 32 GB Apple Silicon Macs.
`text-chat-q36-nano` is the managed Qwen3.6 35B-A3B OptiQ 4-bit MLX snapshot; its upstream repo includes an MTP head that is used only by the adaptive long-context speculative decode path.
`text-chat-bonsai-27b-1bit` and `text-chat-bonsai-27b-2bit` are Prism ML's
pinned packed binary (5.13 GB) and ternary (8.52 GB) dense Qwen3.6 27B
vision/reasoning snapshots. Both use the native Qwen-family runtime, default to
thinking and the published 0.7/0.95/20 sampling, and advertise a 262,144-token
context. Choose 1-bit for lower residency and faster decode, or 2-bit when the
ternary checkpoint's additional weight capacity is worth the memory cost.
`text-agent-ornith-9b` is the managed Ornith 1.0 9B OptiQ MLX coding-agent experiment; it uses the native Qwen-family runtime rather than the GGUF `text code` command.
Ornith 1.5 35B-A3B has official Q4/Q6/Q8 ids ending in `-mlx-4bit`,
`-mlx-6bit`, and `-mlx-8bit`; `text-agent-ornith-35b-mlx` remains the BF16 id.
All use the native Qwen-family runtime, require an explicit managed pull, and
share the pinned base-checkpoint MTP companion installed by that pull. Managed
Ornith 1.5 enables verified MTP drafting from short prompts.
`text-chat-lfm25-2.6b-4bit` is the managed LiquidAI LFM2.5 2.6B dense MLX 4-bit snapshot and runs through the native Swift LFM2 runtime.
The `text-chat-lfm25-1.2b-qad-4bit` and `text-chat-lfm25-2.6b-qad-4bit`
models are deterministic native-MLX conversions of LiquidAI's QAD-trained
Q4_0 checkpoints. QAD retains the compact Q4_0 representation while recovering
accuracy lost by post-training quantization; it does not add a separate speedup
over a comparable Q4_0 runtime. The 1.2B model is the memory-first tier. The
standard 2.6B MLX model remains the faster compact 2.6B lane.
`text-chat-lfm25-a1b-8bit` is the managed LiquidAI LFM2.5 8B-A1B MLX 8-bit snapshot and runs through the native Swift LFM2 runtime.
`vision-chat-lfm25-3b-8bit` adds LiquidAI's SigLIP2 vision tower and multimodal projector to the dense LFM2.5 2.6B language backbone. Use `--image` with a local path or base64 data URL.
`text-chat-laguna-s-2-1` is the opt-in managed Poolside 118B-A8B NVFP4 target
for 96 GB-and-up Apple Silicon. Pulling it installs the pinned DFlash
companion; generation defaults to the validated
temperature 1, top-p 1, top-k 20, and min-p 0.02 recipe.
`text-chat-laguna-xs-2-1` is Poolside's released 33B-A3B Laguna XS 2.1 model
for 36 GB-and-up Apple Silicon, installed from its pinned NVFP4 MLX
serialization. It is opt-in and does not use the Laguna S DFlash companion.

## Chat Winners By RAM Band

| Unified memory | Winner | Why |
| --- | --- | --- |
| 8-15 GB | `text-chat-lfm25-1.2b-qad-4bit` | Memory-first native chat; use the optimized 2.6B MLX model when its larger resident footprint fits. |
| 16-23 GB | `text-chat-gemma4-12b-4bit` | Best compact first chat pick; `text-chat-gemma4-nano` is the safer smallest fallback. |
| 24-63 GB | `text-chat-gemma4-12b-4bit` | Default grounded local-assistant tier; `text-chat-gemma4-turbo` is the larger Gemma alternate. |
| 64-95 GB | `text-chat-gemma4-12b-4bit` | Keep the proven Gemma assistant lane; spend headroom on context, concurrency, or larger Gemma alternates. |
| 96+ GB | `text-agent-deepseek-v4-flash` | Premier agent/API tier; keep Gemma 12B 4-bit for normal interactive local chat. |

## Install And Check

```bash
mere.run model capabilities
mere.run model pull text-chat-gemma4-12b-4bit
mere.run model pull text-chat-lfm25-1.2b-qad-4bit --accept-model-license
mere.run model pull text-chat-lfm25-2.6b-4bit --accept-model-license
mere.run model pull vision-chat-lfm25-3b-8bit --accept-model-license
mere.run model pull text-chat-laguna-s-2-1
mere.run model pull text-chat-laguna-xs-2-1
mere.run text chat --help
```

## Native Assistant Tuning

Use `mere.run text train-lora` for reviewed chat-style SFT data. Keep the data
source-tagged and run `--dry-run --json` first so the dataset fingerprint,
manifest path, and eval prompt count are captured before any optimizer work:

```bash
mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --dry-run \
  --json
```

For a real run, `--eval` is a held-out SFT JSONL dataset. It is validated and
tokenized with the same model template, but never enters the training order;
the result reports assistant-token loss before and after optimization.

The native optimizer path is intentionally part of `mere.run`; keep local
fine-tune workflows in the same command plane as model resolution and runtime
metadata. Drop `--dry-run` when the dataset is reviewed and you are ready for
the native Gemma LoRA training run. Add `--visualize` to open a loopback
training dashboard with loss, events, and adapter artifacts:

```bash
mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --visualize
```

## Parameters

- `--prompt`, `-p`: user prompt.
- `--system`, `-s`: system prompt.
- `--max-tokens`: maximum generated tokens.
- `--context-size`: maximum prompt plus generation context. Bonsai 27B defaults to 262,144.
- `--temperature`: randomness. Lower for factual work, higher for brainstorming.
- `--top-p`: nucleus sampling cutoff.
- `--min-p`: discard tokens whose probability is below this fraction of the
  most likely token's probability. `0` disables the filter; `0.05` means a
  token must have at least 5% of the leading token's probability.
- `--kv-bits`, `--kv-quant-scheme`, `--kv-group-size`, `--quantized-kv-start`: KV cache quantization controls. Qwen-family models accept affine `--kv-bits 4` or `8`; the runtime chooses group size and start. `text-chat-gemma4-turbo` defaults to the existing 4-bit affine TurboQuant KV cache from token 0; explicit flags override that. `--kv-quant-scheme polar --kv-bits 2` enables the experimental packed PolarKV path for memory-pressure and long-context synthetic decode testing.
- `--model-root`, `-m`: explicit local model root.
- `--model`: canonical model id.
- `--response-format`: `text` (default) or `json_object`. JSON-object mode is
  available for native MLX Gemma and Qwen-family chat models, forces thinking
  off, and validates every generated token against the JSON grammar.
- `--thinking`, `--show-thinking`: include hidden reasoning output when supported.
- `--stats`: print timing and throughput to stderr. Native chat timing includes
  user-visible `ttft_s` plus the decode-only `first_token_s`; LFM2 reports
  separate prefill and decode tokens/sec. Gemma4 runs also include MTP state
  and accept/draft counts.
- `--stream`: stream tokens to stdout.
- `--lora`: local `.safetensors` adapter path for supported native chat models.
- `--lora-scale`: adapter scale; defaults to `1.0`.
- `--tools`: comma-separated built-ins, currently `write_file` and `shell_exec`.
- `--tool-loop`: let the model call tools repeatedly.
- `--sandbox-dir`: working directory for tool execution.
- `--allow-shell-exec`: required for `shell_exec`.
- `--allow-absolute-tool-paths`: allow absolute paths for `write_file`.
- `--auto-approve-tools`: skip interactive approval.
- `--quiet`, `-q`: suppress progress output.

## Prompting Patterns

- Give the goal, constraints, desired format, and any source text in one prompt.
- For Gemma-family models, put system-like behavior in the first user instruction when using environments that do not have a separate system role.
- For long answers, ask for structure first, then ask the model to fill sections.
- For tool use, keep `--sandbox-dir` narrow and ask for a plan before enabling auto-approval.

## Examples

```bash
mere.run text chat \
  --model text-chat-bonsai-27b-2bit \
  --context-size 262144 \
  --kv-bits 4 \
  --prompt "Summarize the architecture decisions in this repository."
```

```bash
mere.run text chat \
  --model text-chat-gemma4 \
  --stream \
  --prompt "Explain local-first inference in one concise paragraph."
```

```bash
mere.run text chat \
  --model text-chat-gemma4-12b \
  --prompt "Draft a compact Swift package release checklist."
```

```bash
mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --system "Answer with bullet points and cite uncertainty." \
  --prompt "Compare speech transcription backends for meeting notes."
```

```bash
mere.run text chat \
  --model text-agent-ornith-9b \
  --temperature 0.2 \
  --prompt "Write a compact Swift function with one XCTest for slugifying a model id."
```

```bash
mere.run text chat \
  --model text-chat-q36-nano \
  --response-format json_object \
  --prompt 'Return an object with keys "name", "features", and "stable".'
```

`json_object` requires one complete JSON object at the root. It supports nested
objects and arrays, strings and escapes, Unicode, numbers, booleans, and null.
Thinking and the Qwen-family MTP, continuous-batching, and pipelined decode
paths are disabled so every token is checked before it is streamed. The
llama.cpp/GGUF Q36 lane (`text-chat-q36-nano-gguf`, used by Linux packages)
does not yet have a wired JSON grammar and rejects this option explicitly.

```bash
mere.run text chat \
  --model text-chat-lfm25-2.6b-4bit \
  --prompt "Explain why local inference is useful in one paragraph."
```

## Iteration Tips

- For deterministic summaries, lower temperature to `0.2` and keep top-p near default.
- For brainstorming, try `--temperature 0.9 --top-p 0.95`.
- Use min-p only with sampled generation. It does not change greedy
  `--temperature 0` output.
- Use `--stats` to decide whether a smaller model is better for interactive work.

## Troubleshooting

- Slow first run: the model may be downloading or compiling kernels.
- Tool calls rejected: `shell_exec` requires `--allow-shell-exec`; absolute writes require `--allow-absolute-tool-paths`.
- Output includes unwanted reasoning: omit `--thinking`.
- JSON object mode rejected for a GGUF model: use the native MLX
  `text-chat-q36-nano` model on Apple Silicon; the llama.cpp grammar is not yet
  wired.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/TextChatCommand.swift
- https://ai.google.dev/gemma/docs/core/prompt-structure
- https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit
- https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit
- https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit
- https://huggingface.co/sahilchachra/ornith-1.0-9b-optiq-5bpw-mlx
- https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-8bit
- https://huggingface.co/google/gemma-4-31B
