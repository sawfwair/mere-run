# Video Export Latents

## Purpose

Run native distilled LTX denoising and export final stage latents as safetensors. Use this for debugging, research, or downstream latent workflows rather than normal MP4 generation.

## Required Models

Managed id: `video-ltx-av`, or a local distilled LTX model root with the expected tokenizer, text encoder, distilled weights, and spatial upscaler.

## Install And Check

```bash
mere.run model pull video-ltx-av --accept-model-license
mere.run video export-latents --help
```

## Parameters

- positional prompt: latent generation prompt.
- `--model`, `-m`: managed id or local model root.
- `--model-root`: explicit local distilled root.
- `--output`, `-o`: safetensors path.
- `--width`, `--height`: output size, must be divisible by 64.
- `--num-frames`: frame count, must satisfy `8n+1`.
- `--seed`: deterministic seed.
- `--quiet`, `-q`: suppress diagnostics.

## Usage Patterns

- Use the same prompt and geometry you would use for `video generate`.
- Keep dimensions small while debugging latent shapes.
- Export latents with fixed seed when comparing runtime changes.

## Examples

```bash
mere.run video export-latents \
  "a cinematic drone flyover at sunrise" \
  --model video-ltx-av \
  --width 768 --height 512 \
  --num-frames 65 \
  --seed 42 \
  --output ./flyover.safetensors
```

## Iteration Tips

- Confirm model layout with `mere.run model info video-ltx-av --components`.
- Keep exported artifacts out of git.
- Pair latent exports with the exact command and commit hash when debugging.

## Troubleshooting

- Missing tokenizer or upscaler: inspect the local model root layout.
- Shape mismatch: use dimensions divisible by 64 and frame counts of `8n+1`.
- Need a playable result: use `mere.run video generate` instead.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VideoCommand.swift
- https://docs.ltx.video/open-source-model/usage-guides/text-to-video
- https://huggingface.co/mlx-community/LTX-2-distilled-bf16
