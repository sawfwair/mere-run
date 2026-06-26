# Image Generate

## Purpose

Create a PNG from a text prompt, optionally conditioned on an input image and LoRA. Use this when the user wants a local image result, not just model inspection.

## Required Models

Use one installed image model:

- `image-zimage-nano`, `image-zimage-base`, `image-zimage-max`
- `image-klein-max`, `image-klein-base`, `image-klein-nano`
- `image-bonsai-binary`, `image-bonsai-ternary`
- `image-ideogram4-sdnq-uint4`
- `image-hidream-o1`, `image-hidream-o1-dev`
- `image-krea2-turbo`

## Install And Check

```bash
mere.run model capabilities
mere.run model pull image-zimage-nano
mere.run model pull image-bonsai-binary
mere.run model pull image-bonsai-ternary
mere.run model pull image-krea2-turbo
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
- `--input`, `-i`: source image for image-to-image. For FLUX.2 Klein, this is
  treated as a single reference image.
- `--ref-image`: repeatable reference image for FLUX.2 Klein or HiDream O1
  editing/personalization.
- `--strength`, `--str`: image-to-image/reference change strength from `0.0` to
  `1.0`. FLUX.2 Klein `--ref-image` defaults to a clean reference unless
  strength is set.
- `--max-sequence-length`: prompt token budget.
- `--structured-prompt`, `--json-prompt`: expand the prompt into a structured JSON caption with a local text chat model before generation.
- `--structured-prompt-model`: text chat model id for the adapter. Defaults to `text-chat-gemma4-12b-4bit`.
- `--structured-prompt-model-root`: optional local model root for the adapter.
- `--structured-prompt-max-tokens`: max new tokens for the adapter.
- `--structured-prompt-output`: write the generated JSON caption for review or reuse.
- `--lora`, `-l`: LoRA safetensors path.
- `--lora-scale`: LoRA strength.
- `--quiet`, `-q`: print only the output path.

## Prompting Patterns

- Lead with subject, action, environment, material, lighting, camera or medium, and style constraints.
- For product or inspection images, prefer concrete nouns over mood words: size, surface, label text, angle, background.
- For Z-Image, keep prompts compact and descriptive. Iterate seed, dimensions, and input strength before piling on adjectives.
- For FLUX.2 Klein, use explicit visual composition and relationships: foreground, background, lens/framing, color palette, and what must be absent. Use
  `--ref-image` for card anatomy, product identity, or edit guidance. `--input`
  is accepted as a single-reference shorthand for Klein.
- For Ideogram 4 SDNQ, use `--structured-prompt` when a short prompt needs stronger object, spatial, lighting, camera, or text-render control. The adapter converts the prompt into a long JSON caption and raises the prompt token budget to 2048.
- For Bonsai binary or ternary, start with four steps at 512 or 1024 square; the manifest applies its native FlowMatch sigma shift.
- For Krea 2 Turbo, start with the managed default: 1024 square, 8 steps, CFG 0.0, and no reference or input image. Raw-trained Krea 2 LoRAs can be loaded with `--lora`.
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
  --model image-bonsai-binary \
  --prompt "editorial photo of a bonsai tree inside a glass greenhouse, rain on the windows, soft reflected light" \
  --width 1024 --height 1024 \
  --seed 17 \
  --output ./bonsai.png
```

```bash
mere.run image generate \
  --model image-klein-max \
  --input ./sketch.png \
  --strength 0.45 \
  --prompt "architectural watercolor rendering of the same cabin, pine forest, late afternoon light" \
  --output ./cabin-watercolor.png
```

```bash
mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a cinematic product photo of a translucent portable speaker, crisp reflections" \
  --width 1024 --height 1024 \
  --steps 8 \
  --output ./speaker.png
```

```bash
mere.run image generate \
  --model image-ideogram4-sdnq-uint4 \
  --prompt "a knight and a white horse in a meadow, remove the helmet, make it sunny" \
  --structured-prompt \
  --structured-prompt-output ./knight-prompt.json \
  --output ./knight.png
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
- Unsupported mode on Krea 2 Turbo: use text-to-image with optional `--lora`; reference images and input images are not wired for that family yet.
- Out of memory: choose a smaller model, reduce width/height, or use a nano tier. Krea 2 Turbo is a large BF16 component install and is best treated as a 96 GB+ Apple Silicon path.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ImageGenerateCommand.swift
- https://huggingface.co/docs/diffusers/api/pipelines/z_image
- https://docs.bfl.ai/guides/prompting_guide_flux2_klein
- https://huggingface.co/krea/Krea-2-Turbo
- https://huggingface.co/krea/Krea-2-Raw
- https://github.com/krea-ai/krea-2
