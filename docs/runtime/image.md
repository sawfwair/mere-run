# Image Runtime

This page covers the image-generation part of `mere.run`: what commands exist, what
model families are supported, and how the code is organized.

## Public surface

- `mere.run image generate`
- `mere.run image validate`

## Model families

The public image families are:

- `image-klein-*`: Klein image family
- `image-zimage-*`: ZImage image family
- `image-hidream-o1*`: HiDream O1 unified pixel-transformer family

Common managed IDs:

- `image-klein-nano`
- `image-klein-base`
- `image-klein-max`
- `image-zimage-nano`
- `image-zimage-base`
- `image-zimage-max`
- `image-hidream-o1-dev`
- `image-hidream-o1`

## Typical workflows

### Generate an image

```bash
swift run mere.run image generate \
  --model image-zimage-nano \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png
```

### Image-to-image

```bash
swift run mere.run image generate \
  --prompt "turn this into a pencil sketch" \
  --input ./photo.png \
  --strength 0.6 \
  --output ./sketch.png
```

### HiDream O1 references

HiDream O1 is registered with text-only, one-reference instruction editing, and
multi-reference subject-personalization capabilities. Reference images are
repeatable; with a single reference, `--keep-original-aspect` preserves the
reference aspect ratio when building the HiDream sample.

The native runtime validates model roots, decodes the typed upstream
configuration, tokenizes the upstream chat-template prompt, builds scheduler
inputs, constructs text/reference sample metadata, and runs HiDream generation
through the downloaded Qwen3-VL decoder, vision tower, timestep embedder, patch
embedder, generation-aware attention mask, and final pixel head. Dev uses the
fixed flash FlowMatch schedule with CFG 0.0 by default; Full uses CFG 5.0 by
default and the shifted Flow UniPC scheduler. Reference-image modes run native
Qwen3-VL vision preprocessing and replace chat-template image placeholders
before appending target/reference pixel patches for denoising.

```bash
swift run mere.run image generate \
  --model image-hidream-o1-dev \
  --prompt "a clean studio product photo of the subject" \
  --ref-image ./subject-front.png \
  --ref-image ./subject-side.png \
  --output ./subject.png
```

Use `--steps` or `--cfg` when you want to override the model-specific defaults.
Use `--keep-original-aspect` with a single `--ref-image` for edit cases where
the output should follow the source image aspect ratio.

Installed HiDream smoke tests are intentionally opt-in because each checkpoint
is large and GPU time is meaningful:

```bash
MERERUN_RUN_E2E=installed MERERUN_E2E_HIDREAM=1 ./scripts/check.sh
MERERUN_RUN_E2E=installed MERERUN_E2E_HIDREAM_FULL=1 ./scripts/check.sh
```

### Deterministic validation

```bash
swift run mere.run image validate --family zimage --test all
swift run mere.run image validate --family klein --test pipeline
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/ImageGenerateCommand.swift`
- `Sources/MereRunCLI/Commands/ImageValidateCommand.swift`

### Klein family

- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator.swift`
- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+ModelLoading.swift`
- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Generation.swift`
- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Chat.swift`

### ZImage family

- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator.swift`
- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+ModelLoading.swift`
- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+Inference.swift`
- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+LoRA.swift`

### HiDream O1 family

- `Sources/MereRunCore/HiDreamO1/HiDreamO1Generator.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Resources.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Configs.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Model.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1SampleBuilder.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1ImagePreprocessor.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1TokenizerAndTemplate.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Scheduler.swift`

### Image editing support

- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator.swift`
- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+ModelLoading.swift`
- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+Encoding.swift`

## How image generation flows

At a high level:

1. the CLI parses prompt, model choice, size, steps, optional image input, and optional reference images
2. model resolution maps a canonical model ID or explicit path to a local root
3. the runtime loads the matching components for the chosen family
4. prompt encoding and optional conditioning data are prepared
5. the denoise loop or family-specific generation path runs
6. latents are decoded and written as an image artifact

The image families do not share identical implementation internals, but they are
presented through the same public `mere.run image generate` command.

## Validation philosophy

`mere.run image validate` exists so contributors can run deterministic checks on:

- VAE behavior
- text-encoder behavior
- transformer behavior
- full pipeline behavior

It is intentionally more engineering-oriented than normal end-user workflows.
If you change image internals, this command is the first place to verify that a
family still behaves consistently.

## Related docs

- [CLI Reference](../cli.md)
- [Model Management](./model-management.md)
- [Architecture Reading Map](../architecture.md)
