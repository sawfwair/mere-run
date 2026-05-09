# Video Generate

## Purpose

Generate an MP4 video from text, optionally anchored by a source image, with native Swift/MLX LTX pipelines.

## Required Models

Managed id: `video-ltx-av`, or a local LTX model root.

## Install And Check

```bash
mere.run model pull video-ltx-av
mere.run video generate --help
```

## Parameters

- positional prompt: video prompt.
- `--output`, `-o`: MP4 path.
- `--model`, `-m`: managed id or local model root.
- `--variant`: `distilled` or `unified-av`.
- `--model-root`: explicit local root, takes precedence over `--model`.
- `--width`, `--height`: output size; snapped down to multiples of 64.
- `--num-frames`: frame count; adjusted to `8n+1`.
- `--fps`: frames per second.
- `--seed`: deterministic generation.
- `--image`: source image for image-to-video.
- `--image-strength`: image conditioning strength from `0` to `1`.
- `--quiet`, `-q`: suppress diagnostics.

## Prompting Patterns

- Describe subject, motion, camera movement, environment, lighting, and style.
- For image-to-video, prompt the motion you want, not only what is already visible.
- Keep early drafts short: `--num-frames 65` at `24` fps is a fast test.
- Use standard aspect ratios before custom sizes.

## Examples

```bash
mere.run video generate \
  "slow cinematic dolly shot through a rainy neon alley, reflections on pavement, moody blue and magenta light" \
  --width 768 --height 512 \
  --num-frames 65 \
  --seed 9 \
  --output ./alley.mp4
```

```bash
mere.run video generate \
  "the product rotates gently on a clean studio turntable, soft shadows, premium commercial lighting" \
  --image ./product.png \
  --image-strength 0.8 \
  --output ./product-spin.mp4
```

## Iteration Tips

- Lock seed after motion is promising.
- If motion is weak, make verbs and camera movement more explicit.
- If anatomy or objects distort, lower duration/frame count or anchor with `--image`.

## Troubleshooting

- Size adjusted: expected, dimensions must be divisible by 64.
- Frame count adjusted: expected, LTX uses `8n+1`.
- Empty or poor image-to-video: verify `--image` exists and try a higher `--image-strength`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VideoCommand.swift
- https://docs.ltx.video/open-source-model/usage-guides/text-to-video
- https://ltx.io/model/model-blog/ltx-2-image-to-video-text-to-video-workflow
- https://huggingface.co/mlx-community/LTX-2-distilled-bf16
