# Video Generate

## Purpose

Generate an MP4 video from text, optionally anchored by a source image, with native Swift/MLX LTX pipelines.

## Required Models

Managed ids:

- `video-ltx23-av-mlx`: split-layout standalone distilled LTX 2.3 MLX root for
  fast drafts. The default `--quality draft` selection uses this checkpoint.
- `video-ltx23-full-mlx`: dev checkpoint, official distilled LoRA, vocoder,
  VAEs, and x2 upscaler for `--quality final`, generated synchronized audio,
  and native source-audio-conditioned video.
- `video-ltx23-a2vid-mlx`: compatibility ID for existing A2Vid installs.
- `video-ltx-av`: legacy merged LTX root. Superseded by LTX 2.3; only still required by
  `video export-latents`. Not recommended for `video generate`.

You can also pass a local LTX model root with `--model-root`.

## Install And Check

```bash
mere.run model pull video-ltx23-av-mlx --accept-model-license
mere.run model pull video-ltx23-full-mlx --accept-model-license
mere.run video generate --help
```

## Parameters

- positional prompt: video prompt.
- `--output`, `-o`: MP4 path.
- `--model`, `-m`: managed id or local model root.
- `--quality`: `draft` selects the fast standalone-distilled checkpoint;
  `final` selects the full dev + distilled-LoRA quality pipeline. Default:
  `draft`.
- `--output-mode`: `video-only` writes no audio stream; `audio-video` generates
  synchronized audio. Default: `video-only`.
- `--variant`: compatibility selector. `distilled` defaults to draft
  video-only; `unified-av` defaults to final audio-video. An explicit `--model`
  still wins. Do not combine it with the canonical selectors.
- `--model-root`: explicit local root, takes precedence over `--model`.
- `--width`, `--height`: output size; snapped down to multiples of 64.
- `--num-frames`: frame count; adjusted to `8n+1`.
- `--fps`: frames per second.
- `--seed`: deterministic generation.
- `--audio`: source audio; automatically selects native LTX 2.3 A2Vid.
- `--audio-start-time`: source segment offset in seconds.
- `--a2v-guidance-scale`: source-audio modality guidance, default `3`.
- `--video-cfg-guidance-scale`: A2Vid text CFG, default `3`.
- `--audio-cfg-guidance-scale`: full unified-AV audio CFG, default `7`.
- `--v2a-guidance-scale`: full unified-AV video-to-audio modality guidance,
  default `3`.
- `--a2v-steps`: guided full/dev stage-one steps, default `30`.
- `--negative-prompt`: advanced A2Vid or Wan negative prompt override.
- `--image`: source image for image-to-video.
- `--image-strength`: image conditioning strength from `0` to `1`.
- `--end-image`: optional ending keyframe; requires `--image`.
- `--end-image-strength`: ending keyframe conditioning strength from `0` to `1`.
- `--preflight`: inspect the request without loading MLX, loading the model, or
  writing an MP4.
- `--json`: with `--preflight`, emit a structured report with diagnostics and
  declarative follow-up actions.
- `--timings`: print native LTX 2.3 split-distilled, unified-AV, or A2Vid load,
  generation, decode, write, and end-to-end phase timings to stderr. Legacy
  merged distilled roots do not expose phase timings.
- `--timings-output`: write those timings as JSON.
- `--quiet`, `-q`: suppress diagnostics.

## Prompting Patterns

- Describe subject, motion, camera movement, environment, lighting, and style.
- For image-to-video, prompt the motion you want, not only what is already visible.
- For directed image-to-video, pass `--image` and `--end-image` so the first
  and last latent frames are both anchored.
- Keep early drafts short: `--num-frames 65` at `24` fps is a fast test.
- Use `--quality final` for delivery renders even when the deliverable is
  video-only. Add `--output-mode audio-video --fps 24` when dialogue, score,
  or SFX is part of the output.
- Suppressing audio is an output contract, not a meaningful speed lane. The
  large speed/quality tradeoff comes from `--quality draft` versus `final`.
- Split-layout distilled generation retains the model's joint audio/video
  denoising tokens because audio-to-video cross attention contributes to the
  video result, but skips loading the audio VAE/vocoder and never writes an
  audio track.
- Use standard aspect ratios before custom sizes.
- With `--audio`, the source latent stays frozen through the guided full/dev
  stage and distilled-LoRA refinement; the selected source segment is muxed as
  the soundtrack. Short audio and incompatible models fail without fallback.
- Use `--preflight --json` before long renders to confirm model availability,
  keyframe paths, output overwrite risk, resolved dimensions, and resolved
  frame count/duration. Preflight also rejects `--timings` for legacy merged
  distilled roots and Wan2.2 before generation starts.

## Examples

```bash
mere.run video generate \
  "the singer performs beneath sweeping blue spotlights" \
  --audio ./song.wav \
  --audio-start-time 42 \
  --duration 6 \
  --image ./artist.png \
  --image-strength 0.9 \
  --output ./shot.mp4
```

```bash
mere.run video generate \
  "slow cinematic dolly shot through a rainy neon alley, reflections on pavement, moody blue and magenta light" \
  --width 768 --height 512 \
  --num-frames 65 \
  --output ./alley.mp4 \
  --preflight \
  --json
```

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
  "a red fox runs across a snowy clearing, detailed winter fur, natural motion" \
  --quality final \
  --width 768 --height 512 \
  --duration 4 \
  --seed 23 \
  --output ./fox-final.mp4
```

```bash
mere.run video generate \
  "two actors talking beside a window while a restrained orchestral score and distant city sirens play underneath" \
  --quality final \
  --output-mode audio-video \
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

## Resident Distilled Session

`mere.run video session` keeps the standalone `video-ltx23-av-mlx` transformer,
text encoder, VAEs, vocoder, and upscaler loaded while it processes serial JSONL
requests. Each stdin line produces exactly one stdout line; diagnostics remain
on stderr. Required request keys are `prompt` and `output`.

```bash
printf '%s\n' \
  '{"id":"fox-1","prompt":"a red fox runs across a snowy clearing","output":"./fox-1.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  '{"id":"fox-2","prompt":"a red fox runs across a snowy clearing","output":"./fox-2.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  | mere.run video session --model video-ltx23-av-mlx
```

Successful result lines include phase timings and `resident_model_reused`; the
second result reports zero model-load time. This session emits synchronized AV
from the standalone distilled checkpoint. It intentionally rejects
`video-ltx23-full-mlx`: the full quality/A2Vid lane fuses the distilled LoRA
into the dev transformer in place and must reload before another generation.

## Iteration Tips

- Lock seed after motion is promising.
- If motion is weak, make verbs and camera movement more explicit.
- For audio-video output, prefer `--duration` so the CLI picks the nearest legal `8n+1`
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
