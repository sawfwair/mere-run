# Vision Track Live

## Purpose

Capture from a camera, seed text-prompted objects, and write an annotated tracked video. Use this for local demos and quick real-world tests.

## Required Models

Managed id: `vision-segment-sam31`, or a local SAM 3.1 model root.

## Install And Check

```bash
mere.run model pull vision-segment-sam31 --accept-model-license
mere.run vision track-live --help
```

## Parameters

- `--prompt`: one or more text prompts. Required.
- `--model`, `-m`: managed id or local model root.
- `--output`, `-o`: annotated video path. Required.
- `--json-output`: tracking JSON path.
- `--camera`: device index, default `0`.
- `--duration-seconds`: capture length.
- `--init-frame`: seed frame.
- `--seed-search-frames`: extra frames to search if init frame finds nothing.
- `--threshold`: score threshold.
- `--resolution`: square preprocessing resolution.
- `--show-boxes`: draw boxes.
- `--show-labels`: reserved overlay option.

## Prompting Patterns

- Use objects that are visible at the beginning of the capture.
- Keep the prompt concrete: `red mug`, `person in black shirt`.
- Increase `--seed-search-frames` when the object enters slightly late.

## Examples

```bash
mere.run vision track-live \
  --prompt "blue cup" \
  --duration-seconds 8 \
  --output ./cup-live.mp4 \
  --json-output ./cup-live.json
```

## Iteration Tips

- Test with short captures before running longer demos.
- Improve lighting and framing before changing model settings.
- If the object appears late, adjust `--init-frame` or `--seed-search-frames`.

## Troubleshooting

- Camera error: check device index and camera permissions.
- No prompt error: pass at least one `--prompt`.
- Empty track: lower threshold or make the object visible earlier.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionTrackLiveCommand.swift
- https://huggingface.co/mlx-community/sam3.1-bf16
- https://arxiv.org/abs/2511.16719
