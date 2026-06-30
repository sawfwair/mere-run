# Text Code

## Purpose

Run a local coding assistant for code generation, explanation, patches, and focused technical answers.

## Required Models

Default `text code` managed id: `text-code-qwen3`. You can also pass a local
GGUF model path with `--model`.

For lower-memory coding experiments, `text-code-north-mini` installs the
Unsloth GGUF quant of Cohere Labs' North Mini Code. It runs through the native
Swift/llama.cpp code path and requires a llama.cpp runtime with `cohere2moe`
architecture support.
For larger Ornith coding-agent evals, `text-agent-ornith-35b` installs
DeepReinforce's Q4_K_M GGUF quant and runs through the same native `text-code`
path with a 32K runtime context.
Use `text-agent-ornith-35b-mlx` through `text chat`, `api serve`, or
`model benchmark code` when testing a locally converted native MLX Q4 snapshot.

## Install And Check

```bash
mere.run model pull text-code-qwen3
mere.run model pull text-code-north-mini
mere.run model pull text-agent-ornith-35b
mere.run text code --help
```

## Parameters

- `--prompt`, `-p`: coding task.
- `--system`, `-s`: system prompt. Default is a helpful coding assistant.
- `--max-tokens`: output budget.
- `--temperature`: default `1.0`.
- `--top-p`: default `0.95`.
- `--model`, `-m`: GGUF path, or omit for managed default.
- `--stats`: print timing and tokens/sec to stderr.
- `--quiet`, `-q`: suppress progress.
- `--stream`: stream tokens.

## Prompting Patterns

- Include the language, target file or function, constraints, and expected output shape.
- For patches, ask for a minimal diff and include failing test output or exact compiler error.
- For explanations, ask for "risks and edge cases" so the model does more than paraphrase code.
- For code completion, include imports/types around the hole.

## Examples

```bash
mere.run text code \
  --prompt "Write a Swift function that normalizes a command path array by trimming whitespace and lowercasing each segment."
```

```bash
mere.run text code \
  --stream \
  --temperature 0.4 \
  --prompt "Review this SQL migration for data-loss risks: $(cat migration.sql)"
```

```bash
mere.run text code \
  --model text-code-north-mini \
  --prompt "Write a small Swift Result extension with tests."
```

```bash
mere.run text code \
  --model text-agent-ornith-35b \
  --prompt "Write a small Swift Result extension with tests."
```

## Iteration Tips

- Lower temperature for compiler-sensitive work.
- Keep prompts smaller than the relevant code context; summarize unrelated files.
- Ask for tests in a second pass once the implementation direction is clear.

## Troubleshooting

- No model found: run `mere.run model pull text-code-qwen3` or pass a GGUF path with `--model`.
- North Mini Code fails to load: rebuild or update the bundled llama.cpp runtime
  so it includes `cohere2moe` support.
- Ornith 35B answers with long reasoning: reduce `--max-tokens` or ask for code
  only when doing interactive generation.
- Output rambles: reduce `--max-tokens` and ask for a specific format.
- Code is stale: paste current symbols and file names instead of relying on model memory.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/TextCodeCommand.swift
- https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF
- https://arxiv.org/abs/2603.00729
