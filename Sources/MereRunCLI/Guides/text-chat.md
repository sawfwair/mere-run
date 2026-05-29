# Text Chat

## Purpose

Run a local chat-style text model for answers, drafting, analysis, or lightweight tool use. Use `api serve` instead when another app needs an OpenAI-compatible HTTP endpoint.

## Required Models

Supported native managed ids include `text-chat-gemma4`, `text-chat-gemma4-turbo`, `text-chat-gemma4-nano`, `text-chat-gemma4-max`, `text-chat-q35`, `text-chat-q35-nano`, `text-chat-q36-nano`, and `text-chat-psi-agent`.
`text-chat-gemma4-turbo` is the managed MLX NVFP4 Gemma 4 26B-A4B-it MoE tier for 32 GB Apple Silicon Macs.
`text-chat-q36-nano` is the managed Qwen3.6 35B-A3B OptiQ 4-bit MLX snapshot; its upstream repo includes an MTP head, but this command currently uses the main chat weights only.

## Install And Check

```bash
mere.run model capabilities
mere.run model pull text-chat-gemma4-nano
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
- `--stats`: print timing to stderr.
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
  --model text-chat-q35-nano \
  --system "Answer with bullet points and cite uncertainty." \
  --prompt "Compare speech transcription backends for meeting notes."
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
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- https://huggingface.co/google/gemma-4-31B
