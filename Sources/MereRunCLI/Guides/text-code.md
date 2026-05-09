# Text Code

## Purpose

Run a local GGUF coding assistant for code generation, explanation, patches, and focused technical answers.

## Required Models

Default managed id: `text-code-qwen3`. You can also pass a local GGUF model path with `--model`.

## Install And Check

```bash
mere.run model pull text-code-qwen3
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

## Iteration Tips

- Lower temperature for compiler-sensitive work.
- Keep prompts smaller than the relevant code context; summarize unrelated files.
- Ask for tests in a second pass once the implementation direction is clear.

## Troubleshooting

- No model found: run `mere.run model pull text-code-qwen3` or pass a GGUF path with `--model`.
- Output rambles: reduce `--max-tokens` and ask for a specific format.
- Code is stale: paste current symbols and file names instead of relying on model memory.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/TextCodeCommand.swift
- https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- https://arxiv.org/abs/2603.00729
