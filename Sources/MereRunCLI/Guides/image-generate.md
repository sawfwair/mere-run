# Image Generate

## Purpose

Create a PNG from a text prompt, optionally conditioned on an input image and LoRA. Use this when the user wants a local image result, not just model inspection.

## Required Models

Use one installed image model:

- `image-zimage-nano`, `image-zimage-base`, `image-zimage-max`
- `image-klein-max`, `image-klein-base`, `image-klein-nano`

## Install And Check

```bash
mere.run model capabilities
mere.run model pull image-zimage-nano
mere.run image generate --help
mere.run guide image generate --model image-zimage-nano
```

## Parameters

- `--prompt`, `-p`: positive prompt.
- `--negative-prompt`, `-n`: negative prompt, only used when `--cfg` is greater than `1.0`.
- `--cfg`, `--cfg-scale`: guidance scale. Start at `1.0` for Z-Image and use higher values only when the model/family benefits.
- `--sigma-shift`: FlowMatch schedule shift. Image-to-light/latent paths often use `8`.
- `--output`, `-o`: output PNG path.
- `--width`, `-W`, `--height`, `-H`: output dimensions in pixels.
- `--steps`, `-s`: denoise steps. Start with the model default, increase for quality checks.
- `--seed`: deterministic repeatability.
- `--model`, `-m`: model id or local model root.
- `--input`, `-i`: source image for image-to-image.
- `--strength`, `--str`: image-to-image change strength from `0.0` to `1.0`.
- `--max-sequence-length`: prompt token budget.
- `--lora`, `-l`: LoRA safetensors path.
- `--lora-scale`: LoRA strength.
- `--quiet`, `-q`: print only the output path.

## Prompting Patterns

- Lead with subject, action, environment, material, lighting, camera or medium, and style constraints.
- For product or inspection images, prefer concrete nouns over mood words: size, surface, label text, angle, background.
- For Z-Image, keep prompts compact and descriptive. Iterate seed, dimensions, and input strength before piling on adjectives.
- For FLUX.2 Klein, use explicit visual composition and relationships: foreground, background, lens/framing, color palette, and what must be absent.
- Use `--input` plus `--strength 0.25` to preserve layout; use `0.65` or higher for stronger reinterpretation.

## Examples

```bash
mere.run image generate \
  --model image-zimage-nano \
  --prompt "macro product photo of a matte black ceramic mug, warm window light, clean oak table, 85mm lens" \
  --width 1024 --height 1024 \
  --seed 41 \
  --output ./mug.png
```

```bash
mere.run image generate \
  --model image-klein-max \
  --input ./sketch.png \
  --strength 0.45 \
  --prompt "architectural watercolor rendering of the same cabin, pine forest, late afternoon light" \
  --output ./cabin-watercolor.png
```

## Iteration Tips

- Lock `--seed` once composition is close, then change one parameter at a time.
- If the image is chaotic, reduce prompt length and remove conflicting style references.
- If the subject is correct but weakly styled, keep the same seed and add one style or lighting phrase.
- If a LoRA overwhelms the image, lower `--lora-scale` before changing the prompt.

## Troubleshooting

- Missing model: run `mere.run model pull <model-id>` or pass a local model root with `--model`.
- Negative prompt has no effect: confirm `--cfg` is greater than `1.0`.
- Input image ignored: lower dimensions may be easier to condition; also reduce or raise `--strength` depending on whether the output is too close or too different.
- Out of memory: choose a smaller model, reduce width/height, or use a nano tier.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ImageGenerateCommand.swift
- https://huggingface.co/docs/diffusers/api/pipelines/z_image
- https://docs.bfl.ai/guides/prompting_guide_flux2_klein
