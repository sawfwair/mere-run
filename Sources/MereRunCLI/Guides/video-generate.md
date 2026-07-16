# Video Generate

## Purpose

Generate an MP4 video from text, optionally anchored by a source image, with native Swift/MLX LTX pipelines.

## Required Models

Managed ids:

- `video-ltx23-av-mlx`: **default.** LTX 2.3 MLX split root — the recommended model
  for both the distilled and the high-quality synchronized `--variant unified-av` lanes.
- `video-ltx-av`: legacy merged LTX root. Superseded by LTX 2.3; only still required by
  `video export-latents`. Not recommended for `video generate`.

You can also pass a local LTX model root with `--model-root`.

## Install And Check

```bash
mere.run model pull video-ltx23-av-mlx --accept-model-license
mere.run video generate --help
```

## Parameters

- positional prompt: video prompt.
- `--output`, `-o`: MP4 path.
- `--model`, `-m`: managed id or local model root.
- `--variant`: `distilled` for faster video-only drafts, or `unified-av` for
  synchronized audio/video.
- `--model-root`: explicit local root, takes precedence over `--model`.
- `--width`, `--height`: output size; snapped down to multiples of 64.
- `--num-frames`: frame count; adjusted to `8n+1`.
- `--fps`: frames per second.
- `--seed`: deterministic generation.
- `--image`: source image for image-to-video.
- `--image-strength`: image conditioning strength from `0` to `1`.
- `--end-image`: optional ending keyframe; requires `--image`.
- `--end-image-strength`: ending keyframe conditioning strength from `0` to `1`.
- `--preflight`: inspect the request without loading MLX, loading the model, or
  writing an MP4.
- `--json`: with `--preflight`, emit a structured report with diagnostics and
  declarative follow-up actions.
- `--quiet`, `-q`: suppress diagnostics.

## Prompting Patterns

- Describe subject, motion, camera movement, environment, lighting, and style.
- For image-to-video, prompt the motion you want, not only what is already visible.
- For directed image-to-video, pass `--image` and `--end-image` so the first
  and last latent frames are both anchored.
- Keep early drafts short: `--num-frames 65` at `24` fps is a fast test.
- Use `video-ltx23-av-mlx --variant unified-av --fps 24` for representative
  LTX 2.3 dialogue, score, and SFX checks.
- Use standard aspect ratios before custom sizes.
- Use `--preflight --json` before long renders to confirm model availability,
  keyframe paths, output overwrite risk, resolved dimensions, and resolved
  frame count/duration.

## Examples

```bash
mere.run video generate \
  "slow cinematic dolly shot through a rainy neon alley, reflections on pavement, moody blue and magenta light" \
  --variant distilled \
  --width 768 --height 512 \
  --num-frames 65 \
  --output ./alley.mp4 \
  --preflight \
  --json
```

```bash
mere.run video generate \
  "slow cinematic dolly shot through a rainy neon alley, reflections on pavement, moody blue and magenta light" \
  --variant distilled \
  --width 768 --height 512 \
  --num-frames 65 \
  --seed 9 \
  --output ./alley.mp4
```

```bash
mere.run video generate \
  "two actors talking beside a window while a restrained orchestral score and distant city sirens play underneath" \
  --model video-ltx23-av-mlx \
  --variant unified-av \
  --duration 15 \
  --fps 24 \
  --width 1280 --height 720 \
  --seed 23 \
  --output ./dialogue-score-sfx.mp4
```

```bash
mere.run video generate \
  "the product rotates gently on a clean studio turntable, soft shadows, premium commercial lighting" \
  --image ./product.png \
  --image-strength 0.8 \
  --output ./product-spin.mp4
```

```bash
mere.run video generate \
  "a car drives from a bright morning street into a warm sunset road, smooth forward motion" \
  --image ./car-start.png \
  --image-strength 0.9 \
  --end-image ./car-end.png \
  --end-image-strength 0.85 \
  --output ./car-start-to-end.mp4
```

## Iteration Tips

- Lock seed after motion is promising.
- If motion is weak, make verbs and camera movement more explicit.
- For unified AV, prefer `--duration` so the CLI picks the nearest legal `8n+1`
  frame count for the requested FPS.
- If anatomy or objects distort, lower duration/frame count or anchor with `--image`.

## Troubleshooting

- Size adjusted: expected, dimensions must be divisible by 64.
- Frame count adjusted: expected, LTX uses `8n+1`.
- Slow motion with normal audio: keep unified AV renders at `--fps 24`.
- Empty or poor image-to-video: verify `--image` exists and try a higher `--image-strength`.
- End keyframe rejected: `--end-image` requires a starting `--image`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VideoCommand.swift
- https://docs.ltx.video/open-source-model/usage-guides/text-to-video
- https://ltx.io/model/model-blog/ltx-2-image-to-video-text-to-video-workflow
- https://huggingface.co/mlx-community/LTX-2-distilled-bf16
- https://huggingface.co/dgrauet/ltx-2.3-mlx
