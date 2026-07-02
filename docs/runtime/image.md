# Image Runtime

This page covers the image-generation part of `mere.run`: what commands exist, what
model families are supported, and how the code is organized.

## Public surface

- `mere.run image generate`
- `mere.run image train-lora`
- `mere.run image validate`

## Model families

The public image families are:

- `image-klein-*`: Klein image family
- `image-bonsai-binary`: PrismML Bonsai binary FLUX.2 Klein deployment
- `image-bonsai-ternary`: PrismML Bonsai ternary FLUX.2 Klein deployment
- `image-zimage-*`: ZImage image family
- `image-hidream-o1*`: HiDream O1 unified pixel-transformer family
- `image-krea2-raw`: Krea 2 Raw base checkpoint for LoRA training
- `image-krea2-turbo`: Krea 2 Turbo text-to-image and LoRA inference family
- `image-ideogram4-sdnq-uint4`: Ideogram 4 SDNQ uint4 text-to-image family

Common managed IDs:

- `image-klein-nano`
- `image-klein-base`
- `image-klein-base-9b`
- `image-klein-max`
- `image-bonsai-binary`
- `image-bonsai-ternary`
- `image-zimage-nano`
- `image-zimage-base`
- `image-zimage-max`
- `image-hidream-o1-dev`
- `image-hidream-o1`
- `image-krea2-raw`
- `image-krea2-turbo`
- `image-ideogram4-sdnq-uint4`

## Typical workflows

### Generate an image

```bash
swift run mere.run image generate \
  --model image-zimage-nano \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png
```

### Klein generation and LoRA training

Klein models run through the native Swift FLUX.2 Klein runtime. Base Klein
models can also train LoRA adapters with `image train-lora`. For serious Klein
LoRA training, use the undistilled BF16 `image-klein-base-9b` model id or a
local equivalent model root, then use the saved adapter with Klein
`image generate --lora`. The loader accepts mflux-format Klein transformer
shards and maps their time-guidance weights into the Swift transformer module
layout.
Use `--recipe klein-fast-style` for the canonical fast local style recipe. It
trains on `image-klein-base-9b` at rank 16 for 1000 steps, LR `0.00005`, max
side `512`, disk-backed latent caching, compiled-step disablement, 250-step
checkpoints, and the fast Klein target surface. Use
`--lora-target-mode transformer-linear-walk` when you want an ai-toolkit-style
comparison that trains every transformer Linear/QuantizedLinear layer instead
of the default suffix allowlist.

```bash
swift run mere.run model pull image-klein-base-9b
swift run mere.run model pull image-klein-9b
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --visualize \
  --quiet
swift run mere.run image generate \
  --model image-klein-9b \
  --prompt "TRIGGER_TOKEN a ceramic mug in the trained style" \
  --lora ./style-klein.safetensors \
  --lora-scale 2.0 \
  --output ./style-klein.png
```

For reference-guided Klein LoRA inference, run the adapter on the distilled
Klein model and pass the source composition with `--ref-image`. A practical
starting point for style transfer is `--strength 0.55`, `--lora-scale 1.5`,
1024x768, 16 steps, and a locked seed once the composition is close. Put the
trigger token first, then describe the subject/action relationship, visible
anatomy, and style:

```bash
swift run mere.run image generate \
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

For cleaner public-facing exports, keep the seed and prompt fixed, render at a
larger matching aspect ratio such as 1280x960 with 24 steps, then downsample to
1024x768 with a high-quality image resizer.

Add `--visualize` during training to start a loopback dashboard with the live
loss curve, progress events, samples, checkpoints, and run artifacts. Reopen a
completed or copied run directory later with:

```bash
swift run mere.run image visualize-run ./runs/my-style
```

### Bonsai binary and ternary

`image-bonsai-binary` and `image-bonsai-ternary` map to PrismML's Apple Silicon
Bonsai Image snapshots. They run through the native Swift FLUX.2 Klein runtime
using the upstream packed transformer layout (`transformer-packed-mflux`) and
the 4-bit MLX text encoder layout (`text_encoder-mlx-4bit`). Binary is the
smallest 1-bit g128 deployment; ternary is the larger 2-bit quality-oriented
variant. The binary path uses a native Swift/Metal packed 1-bit affine matmul
kernel, with a dequantized MLX fallback for non-GPU or unsupported shapes while
upstream `mlx-swift` does not expose a `bits=1` quantized matmul kernel.

```bash
swift run mere.run model pull image-bonsai-binary
swift run mere.run image generate \
  --model image-bonsai-binary \
  --prompt "a tiny bonsai tree in a sunlit greenhouse, editorial product photo" \
  --width 1024 --height 1024 \
  --output ./bonsai.png
```

### Image-to-image and Klein references

```bash
swift run mere.run image generate \
  --prompt "turn this into a pencil sketch" \
  --input ./photo.png \
  --strength 0.6 \
  --output ./sketch.png
```

For FLUX.2 Klein, `--input` is treated as a single reference image and is routed
through the same reference-image pipeline as `--ref-image`. Use `--ref-image`
directly when you want repeatable Klein references, or when you want the clean
default reference conditioning.

```bash
swift run mere.run image generate \
  --model image-klein-base \
  --prompt "a vertical gross-out trading card with the same sticker anatomy" \
  --ref-image ./card-reference.png \
  --output ./card.png
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

### Krea 2 Raw and Turbo

`image-krea2-raw` maps to `krea/Krea-2-Raw` and installs Krea's base
checkpoint for LoRA training. `image-krea2-turbo` maps to `krea/Krea-2-Turbo`
and uses the native Swift MLX runtime for Krea's distilled 8-step text-to-image
model. Both managed pulls use the split Diffusers component layout and
deliberately skip the root `raw.safetensors` / `turbo.safetensors` duplicate
transformer files.

Train adapters on Raw, then preview or run them on Turbo:

```bash
swift run mere.run model pull image-krea2-raw
swift run mere.run model pull image-krea2-turbo
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-krea2.safetensors \
  --recipe krea-cinematic-style \
  --quiet
swift run mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a cinematic product photo in the trained style" \
  --lora ./style-krea2.safetensors \
  --width 1024 --height 1024 \
  --steps 8 \
  --output ./speaker.png
```

Krea's published LoRA examples use Diffusers-format `lora_A` / `lora_B`
adapters trained on Raw, rendered on Turbo at 8 steps, guidance 0.0, and LoRA
weight 1.0. The native Krea target set matches the published adapter surface:
264 Linear modules on the full model, including image input, text
projection/fusion, time embedding/projection, transformer attention/feed-forward
gates, and final output projection.
Use `--recipe krea-fast-style` for a quick local Krea proof pass: Raw base, 100
steps, LR `0.0005`, 10-step warmup/cosine decay, 768 square, rank `32`, alpha
`32`, and the full native Krea target surface. Treat this as a smoke recipe and
inspect images before trusting it as a final style adapter. Use
`--recipe krea-cinematic-style` for the safer movie-style lane: 200 steps, LR
`0.0001`, 20-step warmup/cosine decay, 768x416, rank `32`, alpha `32`, and
compiled-step disablement. Override `--width`/`--height` when your source set is
not widescreen.

The wired Krea generation mode is text-to-image with optional LoRA adapters;
image-to-image and reference inputs are not wired for this family yet.

### Ideogram 4 SDNQ

`image-ideogram4-sdnq-uint4` maps to `WaveCut/ideogram-4-sdnq-uint4`. The
managed model path can pull and validate the SDNQ diffusers layout, including
the separate `unconditional_transformer` branch used for guidance. The native
quantization bridge decodes SDNQ asymmetric uint4 linear, embedding, and Conv2d
weights, the runtime builds Qwen3-VL concatenated text features, packs Ideogram
4 text/image samples, runs positive/unconditional CFG denoising, and decodes
through the Flux2-style VAE. Plain text-to-image generation is wired through
`image generate`; image-to-image, reference inputs, and LoRA are not supported
for this family yet.

Ideogram also works with the `image generate --structured-prompt` adapter. The
adapter uses a local text chat model to expand a short prompt into a long,
FIBO-style structured JSON caption covering objects, background, lighting,
aesthetics, camera characteristics, style, context, and text renders. When the
adapter is enabled, the CLI raises the image prompt token budget to 2048 unless
the user already requested a larger value. Use `--structured-prompt-output` to
save the generated JSON for review or later refinement.

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

### Krea 2 family

- `Sources/MereRunCore/Krea2/Krea2Generator.swift`
- `Sources/MereRunCore/Krea2/Krea2LoRAInjector.swift`
- `Sources/MereRunCore/Krea2/Krea2LoRATrainer.swift`
- `Sources/MereRunCore/Krea2/Krea2RawResources.swift`
- `Sources/MereRunCore/Krea2/Krea2Resources.swift`
- `Sources/MereRunCore/Krea2/Krea2Configs.swift`
- `Sources/MereRunCore/Krea2/Krea2Model.swift`
- `Sources/MereRunCore/Krea2/Krea2ModelLoader.swift`
- `Sources/MereRunCore/Krea2/Krea2SampleBuilder.swift`

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
