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
mere.run model pull image-zimage-nano --accept-model-license
mere.run model pull image-bonsai-binary
mere.run model pull image-bonsai-ternary
mere.run model pull image-krea2-turbo --accept-model-license
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
- `--mask`: black/white edit mask used with `--input`. White pixels may be
  regenerated; black pixels are restored exactly from the source after generation.
- `--outpaint`: editable padding as `top,right,bottom,left`, placed inside the
  requested `--width`/`--height`. Requires `--input`.
- `--mask-feather`: blend radius at mask and outpaint boundaries. Defaults to 8 pixels.
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
- `--preflight`: inspect model, input, LoRA, structured-prompt, and output paths without running generation.
- `--json`: with `--preflight`, emit a structured report for scripts and apps.
- `--quiet`, `-q`: print only the output path.

## Preflight

Use preflight when the caller needs a cheap go/no-go contract before model load
or image generation. JSON is written to stdout and diagnostics stay structured:

```bash
mere.run image generate \
  --model image-zimage-nano \
  --prompt "macro product photo of a matte black ceramic mug" \
  --output ./mug.png \
  --preflight \
  --json
```

The report includes `status`, `diagnostics`, resolved model/output summaries,
input and LoRA file checks, effective sampling settings, and declarative actions
such as `start-generation`, `pull-model`, and `open-output-directory`. Hard
blockers exit nonzero after the JSON is printed. Save `result.run_plan` when you
want to replay the exact normalized generation request later:

```bash
mere.run image run-plan ./mug.plan.json --preflight --json
mere.run image run-plan ./mug.plan.json --materialize ./runs/mug --json
mere.run image run-plan ./mug.plan.json
```

Materialization writes `plan.json`, `actions.json`, `run.json`, an initial
events file, and artifact folders before generation starts. The copied plan
relocates the output image and optional structured-prompt sidecar into the run
directory, then execution appends started/finished/failed events to the same
stream.

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
- For a surgical edit, paint white only over the target region in a black mask.
  The backend receives the complete image context, then mere.run restores every
  protected pixel and feathers only the edit boundary.

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
# Replace only the white area of mask.png.
mere.run image generate \
  --model image-klein-max \
  --input ./storefront.png \
  --mask ./sign-mask.png \
  --mask-feather 12 \
  --prompt "replace the painted sign with a navy enamel sign reading NIGHT MARKET" \
  --width 1024 --height 768 \
  --output ./storefront-edited.png
```

```bash
# Add 128 px on each side while preserving the original center.
mere.run image generate \
  --model image-klein-max \
  --input ./portrait.png \
  --outpaint 0,128,0,128 \
  --prompt "continue the studio backdrop and soft rim lighting naturally" \
  --width 1280 --height 1024 \
  --output ./portrait-wide.png
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

## Klein LoRA Reference Inference

Use `--ref-image` when applying a Klein LoRA to a source composition. This is
the preferred path for repeatable Klein reference conditioning; `--input` is
accepted as a single-reference shorthand, but `--ref-image` keeps the recipe
clear and works with multiple references.

Good starting settings for a style LoRA are `--strength 0.55`,
`--lora-scale 1.5`, 1024x768 output, 16 steps, and a locked seed once the pose
is close. Put the trigger token first, then describe the subject, action,
relationship, and style. For limb-heavy reference poses, state the expected
body layout explicitly.

```bash
mere.run image generate \
  --model image-klein-9b \
  --ref-image ./reference-pose.png \
  --strength 0.55 \
  --prompt "TRIGGER_TOKEN two dancers in a rainy city street, full body pose, natural human anatomy, no extra limbs, clean cinematic film still, crisp faces, reflective pavement" \
  --lora ./style-adapter.safetensors \
  --lora-scale 1.5 \
  --width 1024 --height 768 \
  --steps 16 \
  --seed 525252 \
  --output ./style-reference.png
```

If edges look crunchy in a showcase render, render the same seed at a larger
matching aspect ratio, such as 1280x960 with 24 steps, then downsample to the
delivery size with a high-quality image resizer.

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
