# Vision Inspect

## Purpose

Ask a local vision-language model to describe an image or answer a question about it. Use this for ad hoc image understanding rather than dataset caption files.

## Required Models

If `--model` is omitted, mere.run auto-downloads the default Qwen3-VL local vision-language model. You can also pass a local model root.

## Install And Check

```bash
mere.run vision inspect --help
mere.run guide vision inspect
```

## Parameters

- positional image: image file path.
- positional prompt words: optional question. Defaults to "Describe this image."
- `--model`, `-m`: local model root. Omit for default auto-download.
- `--max-tokens`: maximum generated tokens.
- `--temperature`: sampling temperature.
- `--top-p`: nucleus sampling cutoff.

## Prompting Patterns

- Ask specific questions: "What text is visible?" beats "analyze this".
- Request uncertainty when details matter.
- For structured output, name the fields you need.
- For visual QA, include the visual target and expected granularity.

## Examples

```bash
mere.run vision inspect ./receipt.jpg "Extract merchant, date, total, and line items as JSON-like text."
```

```bash
mere.run vision inspect ./dashboard.png \
  "Summarize the chart trend and call out any visible labels."
```

## Iteration Tips

- Crop small details before asking about them.
- Lower temperature for extraction; raise only for descriptive prose.
- If the model misses tiny text, try `vision ocr`.

## Troubleshooting

- Image not found: use an absolute path or check the working directory.
- Output is too long: lower `--max-tokens` and ask for a list.
- Hallucinated details: ask the model to say "not visible" when uncertain.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionInspectCommand.swift
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://arxiv.org/abs/2511.21631
