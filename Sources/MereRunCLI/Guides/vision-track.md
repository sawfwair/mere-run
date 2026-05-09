# Vision Track

## Purpose

Track prompted objects through a video using SAM 3.1. The command seeds objects on an initial frame, then writes an annotated video and tracking JSON.

## Required Models

Managed id: `vision-segment-sam31`, or a local SAM 3.1 model root.

## Install And Check

```bash
mere.run model pull vision-segment-sam31
mere.run vision track --help
```

## Parameters

- positional video: video file path.
- `--prompt`: one or more text prompts for objects on the init frame.
- `--box`: `x1,y1,x2,y2[,label]`, repeatable.
- `--point`: `x,y,positive[,label]` or `x,y,negative[,label]`, repeatable.
- `--model`, `-m`: managed id or local model root.
- `--output`, `-o`: annotated video path.
- `--json-output`: tracking JSON path.
- `--mask-output-dir`: per-frame masks directory.
- `--init-frame`: seed frame index.
- `--end-frame`: inclusive final frame.
- `--threshold`: score threshold.
- `--resolution`: square preprocessing resolution.
- `--show-boxes`: draw boxes.
- `--show-labels`: reserved overlay option.

## Prompting Patterns

- Pick an init frame where the object is visible and not motion-blurred.
- Use geometry prompts for a specific person/object among many similar ones.
- Track shorter clips first, then expand with `--end-frame`.

## Examples

```bash
mere.run vision track ./clip.mp4 \
  --prompt "the white dog" \
  --init-frame 12 \
  --output ./dog-tracked.mp4 \
  --json-output ./dog-tracked.json
```

```bash
mere.run vision track ./clip.mp4 \
  --box 220,160,420,500,runner \
  --end-frame 180 \
  --show-boxes
```

## Iteration Tips

- Move `--init-frame` if the first mask is wrong.
- Add a box when text finds the wrong instance.
- Export masks only after the tracking quality is acceptable.

## Troubleshooting

- No object found: choose a clearer init frame or lower threshold.
- Drift: seed with a tighter box or shorter clip.
- Output is large: omit `--mask-output-dir` until needed.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionTrackCommand.swift
- https://huggingface.co/mlx-community/sam3.1-bf16
- https://arxiv.org/abs/2511.16719
