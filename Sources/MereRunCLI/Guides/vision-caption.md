# Vision Caption

## Purpose

Write one caption text file per image, optimized for training-friendly descriptions such as LoRA datasets.

## Required Models

If `--model` is omitted, mere.run auto-downloads the default Qwen3-VL 2B local vision-language model. You can also pass a local model root.

## Install And Check

```bash
mere.run vision caption --help
mere.run guide vision caption
```

## Parameters

- positional images: one or more image paths.
- `--model`, `-m`: local model root. Omit for default auto-download.
- `--output`, `-o`: output directory for `.txt` captions.
- `--prompt`: caption instruction.
- `--max-tokens`: maximum generated tokens.
- `--temperature`: sampling temperature.
- `--top-p`: nucleus sampling cutoff.

## Prompting Patterns

- For LoRA captions, ask for concrete nouns and visible attributes, not interpretation.
- Keep the prompt short so captions remain consistent across a dataset.
- Add dataset-specific tags only when the training workflow expects them.

## Examples

```bash
mere.run vision caption ./train/*.png --output ./captions
```

```bash
mere.run vision caption ./portrait.jpg \
  --prompt "Write a compact LoRA caption: subject, clothing, pose, background, lighting." \
  --max-tokens 80
```

## Iteration Tips

- Inspect a random sample before captioning an entire folder.
- Lower temperature for consistent dataset captions.
- Keep naming stable: image `foo.png` writes caption `foo.txt`.

## Troubleshooting

- No images: pass at least one image path.
- Bad captions: tighten `--prompt` and lower `--temperature`.
- Model path error: pass a directory, or omit `--model` for auto-download.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionCaptionCommand.swift
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://arxiv.org/abs/2511.21631
