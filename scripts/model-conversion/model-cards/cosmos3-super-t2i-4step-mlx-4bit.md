---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Cosmos3-Super-Text2Image-4Step
pipeline_tag: text-to-image
library_name: mlx
tags:
  - mlx
  - apple-silicon
  - cosmos3
  - text-to-image
  - quantized
---

# Cosmos3-Super Text2Image 4-Step MLX 4-bit

This repository is a native MLX conversion of
[`nvidia/Cosmos3-Super-Text2Image-4Step`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image-4Step),
pinned at revision `aa0d5a57b7b045d68daa60fbacd84ec723c7cb7b`.

The 64-billion-parameter mixture-of-transformers backbone uses 4-bit affine
group quantization with a group size of 64. The VAE and normalization,
projection, and time-embedding tensors retain their source precision. The
unused sound projections and language-model head are omitted because this
artifact supports text-to-image generation only.

## Use with mere.run

Use the native Cosmos3 command and pass the downloaded repository as a local
model root:

```bash
mere.run video cosmos3 \
  "A glass observatory above a bioluminescent ocean at blue hour" \
  --mode text-to-image \
  --model /path/to/Cosmos3-Super-Text2Image-4Step-MLX-4bit \
  --width 768 \
  --height 768 \
  --seed 42 \
  --output cosmos3-super.png
```

The distilled checkpoint has a fixed four-step stochastic schedule and does
not use classifier-free guidance. `mere.run` selects four steps and a guidance
scale of 1 automatically for this artifact. Flow-shift and scheduler overrides
are not supported, and negative prompts are ignored.

This native path does not bundle or run NVIDIA's separate Cosmos guardrail.
Add prompt and output safety checks appropriate to your application before
showing generated images to users.

## Conversion and provenance

The conversion is deterministic at the tensor level and was produced by
`scripts/model-conversion/convert_cosmos3_super_t2i_mlx.py` in the `mere.run`
repository. See `CONVERSION.json` for the converter digest, package versions,
hardware record, tensor counts, and quantization settings. See
`SOURCE_MANIFEST.json` for the pinned source inventory and `SHA256SUMS` for the
published artifact checksums.

The following source tensors are intentionally omitted:

- `lm_head.weight`
- `audio_modality_embed`
- `audio_proj_in.*`
- `audio_proj_out.*`

The upstream model card is retained as `UPSTREAM_MODEL_CARD.md`.

## Validation status

Tensor inventory, configuration decoding, native module construction, and the
published distilled scheduler are covered by automated tests. Conversion does
not by itself establish visual-quality parity with NVIDIA's BF16 checkpoint.
Record device, prompt, seed, output digest, peak memory, and elapsed time when
qualifying a generated image.

### Native qualification

The published artifact completed an end-to-end 768x768 native generation on an
Apple M4 Max MacBook Pro with 128 GB of unified memory and macOS 26.5.2. The
command used a debug build of `mere.run`, four steps, seed 42, and this prompt:

> A glass observatory above a bioluminescent ocean at blue hour

| Measurement | Result |
| --- | --- |
| Wall time | 60.85 seconds |
| Maximum resident memory | 38,657,916,928 bytes |
| Peak memory footprint | 53,394,599,464 bytes |
| Swaps | 0 |
| Output SHA-256 | `006a73d23850c1ab6ae9c05e5521109a8a2678f69e42fe362cd3a837d77c7495` |

This is native runtime smoke evidence, not a BF16 parity or broad visual-quality
evaluation.

![Native MLX qualification output](https://huggingface.co/Sawfwair/Cosmos3-Super-Text2Image-4Step-MLX-4bit/resolve/main/examples/qualification-768-seed42.png)

## License

This model is distributed under the
[NVIDIA Open Model Development and Weight License 1.1](https://openmdw.ai/license/1-1/).
The artifact includes an archived copy in `OPENMDW-1.1-LICENSE.html`. Review the
license and the upstream use restrictions before downloading or redistributing
the weights.
