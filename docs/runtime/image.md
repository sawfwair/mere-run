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

Common managed IDs:

- `image-klein-nano`
- `image-klein-base`
- `image-klein-max`
- `image-zimage-nano`
- `image-zimage-base`
- `image-zimage-max`

## Typical workflows

### Generate an image

```bash
swift run mere.run image generate \
  --model image-zimage-max \
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

### Image editing support

- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator.swift`
- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+ModelLoading.swift`
- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+Encoding.swift`

## How image generation flows

At a high level:

1. the CLI parses prompt, model choice, size, steps, and optional image input
2. model resolution maps a canonical model ID or explicit path to a local root
3. the runtime loads the matching components for the chosen family
4. prompt encoding and optional conditioning data are prepared
5. the denoise loop or family-specific generation path runs
6. latents are decoded and written as an image artifact

The two families do not share identical implementation internals, but they are
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
