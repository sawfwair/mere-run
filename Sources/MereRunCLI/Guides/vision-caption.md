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
- `--output-dir`, `-o`: output directory for `.txt` captions.
- `--prompt`: caption instruction.
- `--prompt-file`: read longer reusable caption instructions from a UTF-8 text file.
- `--focus`: one or more visible details to prioritize, such as `card border`, `printed title`, or `surface texture`.
- `--trigger-token`: prefix each saved caption with an exact LoRA trigger token.
- `--max-tokens`: maximum generated tokens.
- `--temperature`: sampling temperature.
- `--top-p`: nucleus sampling cutoff.

## Prompting Patterns

- For LoRA captions, ask for concrete nouns and visible attributes, not interpretation.
- Keep the prompt short so captions remain consistent across a dataset.
- Add dataset-specific tags only when the training workflow expects them.
- Use `--focus` for domain attributes that the generic captioner often misses.
- Use `--trigger-token` when every saved caption must start with the same token.

## Examples

```bash
mere.run vision caption ./train/*.png --output-dir ./captions
```

```bash
mere.run vision caption ./portrait.jpg \
  --prompt "Write a compact LoRA caption: subject, clothing, pose, background, lighting." \
  --max-tokens 80
```

```bash
mere.run vision caption ./cards/*.jpg \
  --output-dir ./captions \
  --prompt-file ./card-caption-prompt.txt \
  --focus "full card border" "printed title text" "visible gag" \
  --trigger-token cardstyle \
  --max-tokens 128 \
  --temperature 0.1
```

## Iteration Tips

- Inspect a random sample before captioning an entire folder.
- Lower temperature for consistent dataset captions.
- Keep naming stable: image `foo.png` writes caption `foo.txt`.
- For style LoRAs, compare sample captions against the training objective before
  spending time on a full run.

## Troubleshooting

- No images: pass at least one image path.
- Bad captions: tighten `--prompt` and lower `--temperature`.
- Missing repeated visual details: add `--focus` terms or move detailed guidance
  into `--prompt-file`.
- Model path error: pass a directory, or omit `--model` for auto-download.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionCaptionCommand.swift
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://arxiv.org/abs/2511.21631
