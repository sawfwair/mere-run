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

# LTX 2.5 Full BF16 for mere.run

This is the self-contained, inference-only Full/Dev LTX 2.5 package used by
`mere.run` on Apple Silicon. It is derived from the immutable
`Lightricks/LTX-2.5` revision
`dd53cc2cd45bbeaa3563dfb575cba3f49cf44761`.

The dev and distilled video transformers and Gemma 4 text encoder remain BF16.
Transformer tensor payload bytes are copied into mere.run's native module-key
namespace and physical order. Installation therefore does not retain two
official transformers beside generated native optimization copies.

## Contents

- native BF16 dev and distilled transformers plus one shared connector;
- official BF16 Gemma 4 text encoder and LTX projection;
- convolutional and diffusion video VAEs plus the audio VAE/vocoder;
- spatial and temporal upsamplers, distilled LoRA, and duration head;
- governing LTX and Gemma license and notice files.

There are no ComfyUI INT8, NVFP4, duplicate source-transformer, IC-LoRA adapter,
or training artifacts in this distribution. Optional DFR, HDR, and Dub-It
adapters remain separate installs.

## Use

```bash
mere.run model pull video-ltx25-full-bf16 --accept-model-license
mere.run video generate \
  "an intricate clockwork observatory turns beneath the stars" \
  --model video-ltx25-full-bf16 \
  --output-mode audio-video \
  --output observatory.mp4
```

No local conversion, re-keying, or `mere.run model optimize` step is required.

## Terms

LTX 2.5 is governed by the LTX-2 Community License Agreement and Acceptable Use
Policy included in this repository. The bundled Gemma 4 weights are governed by
the included Gemma terms. Review all terms before use or redistribution.
