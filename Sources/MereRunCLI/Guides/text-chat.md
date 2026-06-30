# Text Chat

## Purpose

Run a local chat-style text model for answers, drafting, analysis, or lightweight tool use. Use `api serve` instead when another app needs an OpenAI-compatible HTTP endpoint.

## Required Models

Supported native managed ids include `text-chat-gemma4`, `text-chat-gemma4-12b`, `text-chat-gemma4-12b-4bit`, `text-chat-gemma4-turbo`, `text-chat-gemma4-nano`, `text-chat-gemma4-max`, `text-chat-q36-nano`, `text-agent-ornith-9b`, `text-chat-lfm25-a1b-8bit`, and `text-chat-psi-agent`.
`text-chat-gemma4-12b` is the managed dense Google Gemma 4 12B-it checkpoint, routed through the native Swift Gemma 4 runtime for text chat.
Pulling `text-chat-gemma4-12b` or `vision-chat-gemma4-12b` also installs the managed `text-chat-gemma4-12b-mtp` assistant; greedy serial Gemma 12B decode uses it for verified decode-tail MTP when the prompt is above the configured threshold.
`text-chat-gemma4-turbo` is the managed MLX NVFP4 Gemma 4 26B-A4B-it MoE tier for 32 GB Apple Silicon Macs.
`text-chat-q36-nano` is the managed Qwen3.6 35B-A3B OptiQ 4-bit MLX snapshot; its upstream repo includes an MTP head that is used only by the adaptive long-context speculative decode path.
`text-agent-ornith-9b` is the managed Ornith 1.0 9B OptiQ MLX coding-agent experiment; it uses the native Qwen-family runtime rather than the GGUF `text code` command.
`text-chat-lfm25-a1b-8bit` is the managed LiquidAI LFM2.5 8B-A1B MLX 8-bit snapshot and runs through the native Swift LFM2 runtime.

## Chat Winners By RAM Band

| Unified memory | Winner | Why |
| --- | --- | --- |
| 16-23 GB | `text-chat-gemma4-12b-4bit` | Best compact first chat pick; `text-chat-gemma4-nano` is the safer smallest fallback. |
| 24-63 GB | `text-chat-gemma4-12b-4bit` | Default grounded local-assistant tier; `text-chat-gemma4-turbo` is the larger Gemma alternate. |
| 64-95 GB | `text-chat-gemma4-12b-4bit` | Keep the proven Gemma assistant lane; spend headroom on context, concurrency, or larger Gemma alternates. |
| 96+ GB | `text-agent-deepseek-v4-flash` | Premier agent/API tier; keep Gemma 12B 4-bit for normal interactive local chat. |

## Install And Check

```bash
mere.run model capabilities
mere.run model pull text-chat-gemma4-12b-4bit
mere.run text chat --help
```

## Parameters

- `--prompt`, `-p`: user prompt.
- `--system`, `-s`: system prompt.
- `--max-tokens`: maximum generated tokens.
- `--temperature`: randomness. Lower for factual work, higher for brainstorming.
- `--top-p`: nucleus sampling cutoff.
- `--kv-bits`, `--kv-quant-scheme`, `--kv-group-size`, `--quantized-kv-start`: Gemma4 KV cache quantization controls. `text-chat-gemma4-turbo` defaults to the existing 4-bit affine TurboQuant KV cache from token 0; explicit flags override that. `--kv-quant-scheme polar --kv-bits 2` enables the experimental packed PolarKV path for memory-pressure and long-context synthetic decode testing.
- `--model-root`, `-m`: explicit local model root.
- `--model`: canonical model id.
- `--thinking`, `--show-thinking`: include hidden reasoning output when supported.
- `--stats`: print timing to stderr; Gemma4 runs also include MTP state and accept/draft counts.
- `--stream`: stream tokens to stdout.
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
  --model text-chat-lfm25-a1b-8bit \
  --prompt "Summarize the tradeoffs of mixture-of-experts chat models."
```

## Iteration Tips

- For deterministic summaries, lower temperature to `0.2` and keep top-p near default.
- For brainstorming, try `--temperature 0.9 --top-p 0.95`.
- Use `--stats` to decide whether a smaller model is better for interactive work.

## Troubleshooting

- Slow first run: the model may be downloading or compiling kernels.
- Tool calls rejected: `shell_exec` requires `--allow-shell-exec`; absolute writes require `--allow-absolute-tool-paths`.
- Output includes unwanted reasoning: omit `--thinking`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/TextChatCommand.swift
- https://ai.google.dev/gemma/docs/core/prompt-structure
- https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit
- https://huggingface.co/sahilchachra/ornith-1.0-9b-optiq-5bpw-mlx
- https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-8bit
- https://huggingface.co/google/gemma-4-31B
