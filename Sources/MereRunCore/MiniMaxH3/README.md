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
The default schedule adapts to packed-row cost at 9, 16, or 21 points. The
maximum acceleration mode caps automatic schedules at 12 points (11 model
evaluations). A
source-bound cache stores the exact released 31-point AdaLN curve and resamples
it for explicit point-count overrides. The converted transformer stores global
Q/K/V slabs; the unmodified video VAE retains the released per-head interleave.

Compact Q4 remains the automatic lane on lower-memory MacBooks. Memory-qualified
MacBooks with at least 96 GiB of unified memory, and desktop Macs with sufficient
headroom, expand those weights once to resident BF16 for compute-bound denoise;
the CLI can force either mode. H3 inference also raises MLX wired residency
through the shared ticket coordinator so weights and activation workspaces do
not silently fall out of the GPU residency set.

The checksum-pinned `minimax-h3-turbo-4step` and
`minimax-h3-lightx2v-4step` LoRAs run only with the BF16 FL2VA transformer.
The EMA-850 adapter's 259 mixed-rank pairs are applied as activation-space
deltas, its fused QKV rows are deinterleaved to match the runtime's global
slabs, and its AdaLN deltas are included while the exact four-evaluation
schedule cache is built. The LightX2V adapter's 312 native PEFT pairs retain
their separate Q/K/V projections and published alpha/rank scale while their
deltas are fused into the BF16 transformer once before denoising. This avoids
both an expanded converted checkpoint and per-block LoRA matmuls. Neither
adapter can be combined with Ref2VA or tail-residual reuse.

`--h3-acceleration quality` executes every transformer block and preserves the
native same-seed trajectory. The explicit `balanced` and `maximum` modes trade
exact trajectory identity for speed: a full step captures the contribution of
the trailing 50% or 75% of blocks, eligible adjacent steps recompute the prefix
and reuse that residual, and a full refresh follows at most two cache hits.
Both modalities must have a sigma delta below 0.12. At least two complete
evaluations precede reuse, and the final schedule region always executes all
50 blocks.

The model weights are governed by the MiniMax-H3 Community License, not the
mere.run source license. In particular, the published license excludes use,
distribution, and display in the United States, European Union, United
Kingdom, and Republic of Korea. Managed artifacts are explicit-pull only and
require the user to acknowledge the model license. Conversion must likewise
run in an allowed territory. See `scripts/model-conversion/README.md`.

The explicit-pull FL2VA package is published as
`Sawfwair/MiniMax-H3-FL2VA-MLX-4bit` and pinned to its immutable verified Hub
commit `e1244ad93d60c737c7e0f065a1c9372f3de7caf8`. It is produced only from
`MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027` by
`scripts/model-conversion/convert_minimax_h3_official_mlx.py`; no converted or
quantized third-party checkpoint is a weight input. The package includes the
source manifest and conversion hashes alongside the runtime files. Its 52
fused transformer QKV matrices are deinterleaved from the released per-head
rows into the global Q/K/V slabs expected by the native runtime before Q4
packing.
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
