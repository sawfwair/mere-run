# MiniMax-H3

This directory contains the native Swift/MLX implementation of MiniMax-H3. The
runtime consumes a flat, self-contained MLX artifact rather than either
roughly 144 GB upstream partition directly.

Supported open-checkpoint paths:

- text-to-video-and-audio with the FL2VA checkpoint;
- first-frame and first/last-frame video-and-audio conditioning with FL2VA;
- ordered image, video (including its soundtrack), and audio reference
  conditioning with Ref2VA. The released limits are 12 total references,
  including at most 9 images, 3 videos, and 3 audio clips; audio cannot be the
  only reference type.

H3 emits 24 fps RGB video and native 32 kHz stereo audio. Frame counts use the
released `17*n+5` temporal geometry. The transformer is CFG-distilled; the
runtime therefore performs a single model evaluation per denoising step.
The default schedule adapts to packed-row cost at 9, 16, or 31 points. A
source-bound cache stores the exact released 31-point AdaLN curve and resamples
it for explicit point-count overrides. The converted transformer stores global
Q/K/V slabs; the unmodified video VAE retains the released per-head interleave.

Compact Q4 is the automatic MacBook lane. Desktop Macs with sufficient unified
memory may expand those weights once to resident BF16 for compute-bound denoise;
the CLI can force either mode. H3 inference also raises MLX wired residency
through the shared ticket coordinator so weights and activation workspaces do
not silently fall out of the GPU residency set.

The model weights are governed by the MiniMax-H3 Community License, not the
mere.run source license. In particular, the published license excludes use,
distribution, and display in the United States, European Union, United
Kingdom, and Republic of Korea. Managed artifacts are explicit-pull only and
require the user to acknowledge the model license. Conversion must likewise
run in an allowed territory. See `scripts/model-conversion/README.md`.

The explicit-pull FL2VA package is pinned to
`ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit@32bfc37f1dc8bd331394573859a627bc0aa9822b`.
Ref2VA is a local conversion lane because this repository does not publish a
redistributable converted snapshot. Convert the pinned
`Comfy-Org/MiniMax-H3` ConvRot source with
`scripts/model-conversion/convert_minimax_h3_convrot.py`, then stage the
result as `transformer.safetensors` beside the FL package's exact conditioner,
VAEs, tokenizer, configuration, license, notice, and modification files.
Change only `partition` in `config.json` to `ref2va`; retain the conversion
receipt next to the root as provenance.

The implementation follows the published MiniMax/Hugging Face architecture.
No ComfyUI source is included or used at runtime.
