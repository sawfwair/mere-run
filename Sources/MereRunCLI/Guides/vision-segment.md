# Vision Segment

## Purpose

Segment prompted objects in an image with SAM 3.1 and write an annotated output, JSON metadata, and optional mask PNGs.

## Required Models

Managed id: `vision-segment-sam31`, or a local SAM 3.1 model root.

## Install And Check

```bash
mere.run model pull vision-segment-sam31
mere.run vision segment --help
```

## Parameters

- positional image: image file path.
- `--prompt`: one or more object text prompts.
- `--box`: `x1,y1,x2,y2[,label]`, repeatable.
- `--point`: `x,y,positive[,label]` or `x,y,negative[,label]`, repeatable.
- `--model`, `-m`: managed id or local model root.
- `--output`, `-o`: annotated image path.
- `--json-output`: metadata path.
- `--mask-output-dir`: per-object masks directory.
- `--threshold`: score threshold from `0` to `1`.
- `--resolution`: square preprocessing resolution.
- `--show-boxes`: draw boxes.
- `--multimask`: emit multiple candidates for geometry prompts.

## Prompting Patterns

- Use `--prompt` for semantic object selection.
- Use `--box` or `--point` when text prompts select the wrong instance.
- Combine positive and negative points to split touching objects.
- Lower threshold for recall; raise it for precision.

## Examples

```bash
mere.run vision segment ./desk.jpg \
  --prompt "coffee mug" \
  --mask-output-dir ./masks \
  --output ./desk-segmented.jpg
```

```bash
mere.run vision segment ./photo.jpg \
  --box 120,80,360,520,person \
  --point 180,120,positive,person \
  --show-boxes
```

## Iteration Tips

- Start with text prompt, then add geometry if multiple instances are confused.
- Use `--multimask` for ambiguous point or box prompts.
- Keep the same image coordinate system when iterating prompts.

## Troubleshooting

- Validation error: provide at least one `--prompt`, `--box`, or `--point`.
- Threshold rejects everything: lower `--threshold`.
- Edges are rough: increase `--resolution` if memory allows.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionSegmentCommand.swift
- https://huggingface.co/mlx-community/sam3.1-bf16
- https://arxiv.org/abs/2511.16719
