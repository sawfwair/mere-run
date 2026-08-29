---
base_model: Lightricks/LTX-2.5
license: other
license_name: ltx-2-community-license-agreement
license_link: https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md
pipeline_tag: text-to-video
tags:
  - ltx-video
  - ltx-2.5
  - mlx
  - mere.run
---

# LTX 2.5 Distilled BF16 for mere.run

This is the self-contained, inference-only LTX 2.5 Distilled package used by
`mere.run` on Apple Silicon. It is derived from the immutable
`Lightricks/LTX-2.5` revision
`dd53cc2cd45bbeaa3563dfb575cba3f49cf44761`.

The video transformer remains BF16. Its tensor payload bytes are copied into
mere.run's native module-key namespace and physical order so installation does
not retain an upstream checkpoint beside a generated optimization cache. The
Gemma 4 language tower uses MLX affine Q4/group-64 for eligible language
weights; the LTX projection and tokenizer assets remain BF16/raw.

## Contents

- native BF16 distilled transformer and shared connector;
- MLX Q4 Gemma 4 text tower with BF16 LTX projection;
- convolutional video VAE, audio VAE/vocoder, spatial upsampler, and duration
  head;
- governing LTX and Gemma license and notice files.

There are no ComfyUI INT8, NVFP4, dev-transformer, DiffVAE, temporal-upscaler,
or training artifacts in this distribution.

## Use

```bash
mere.run model pull video-ltx25-distilled-bf16 --accept-model-license
mere.run video generate \
  "a clockwork observatory turns beneath a starry sky" \
  --model video-ltx25-distilled-bf16 \
  --output-mode audio-video \
  --output observatory.mp4
```

No local conversion, re-keying, or `mere.run model optimize` step is required.

## Terms

LTX 2.5 is governed by the LTX-2 Community License Agreement and Acceptable Use
Policy included in this repository. The bundled Gemma 4 weights are governed by
the included Gemma terms. Review all terms before use or redistribution.
