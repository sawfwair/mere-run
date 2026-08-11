# MiniMax-H3

This directory contains the native Swift/MLX implementation of MiniMax-H3. The
runtime consumes a flat, self-contained MLX artifact rather than either
roughly 144 GB upstream partition directly.

Supported open-checkpoint paths:

- text-to-video-and-audio with the FL2VA checkpoint;
- first-frame and first/last-frame video-and-audio conditioning with FL2VA;
- up to 12 arbitrary frame-indexed FL2VA image conditions;
- ordered image, video (including its soundtrack), and audio reference
  conditioning with Ref2VA. The released limits are 12 total references,
  including at most 9 images, 3 videos, and 3 audio clips; audio cannot be the
  only reference type;
- resident FL2VA and Ref2VA sliding windows that condition on overlapping
  motion, the boundary frame, and matching generated stereo audio.

H3 emits 24 fps RGB video and native 32 kHz stereo audio. Frame counts use the
released `17*n+5` temporal geometry. The transformer is CFG-distilled; the
runtime therefore performs a single model evaluation per denoising step.
The default schedule adapts to packed-row cost at 9, 16, or 21 points. The
maximum acceleration mode caps automatic schedules at 12 points (11 model
evaluations). A
source-bound cache stores the exact released 31-point AdaLN curve and resamples
it for explicit point-count overrides. The converted transformer stores global
Q/K/V slabs; the unmodified video VAE retains the released per-head interleave.

Ref2VA image references preserve source aspect ratio and are downscaled only
when their area exceeds the internal render canvas. Standalone audio references
retain their complete 2-15 second span, subject to the released 15-second total
limit. Condition augmentation and target video/audio initialization use the
released independent seeded streams and native latent layouts.

Compact Q4 remains the automatic lane on lower-memory MacBooks. Memory-qualified
MacBooks with at least 96 GiB of unified memory, and desktop Macs with sufficient
headroom, expand those weights once to resident BF16 for compute-bound denoise;
the CLI can force either mode. H3 inference also raises MLX wired residency
through the shared ticket coordinator so weights and activation workspaces do
not silently fall out of the GPU residency set.

The checksum-pinned `minimax-h3-turbo-4step`,
`minimax-h3-lightx2v-4step`, `minimax-h3-lightx2v-8step-v1`, and
`minimax-h3-lightx2v-4step-v1-768p` LoRAs run only with the BF16 FL2VA
transformer.
The EMA-850 adapter's 259 mixed-rank pairs are applied as activation-space
deltas, its fused QKV rows are deinterleaved to match the runtime's global
slabs, and its AdaLN deltas are included while the exact four-evaluation
schedule cache is built. The LightX2V adapter's 312 native PEFT pairs retain
their separate Q/K/V projections and published alpha/rank scale while their
deltas are fused into the BF16 transformer once before denoising. This avoids
both an expanded converted checkpoint and per-block LoRA matmuls. Neither
v1.0 recipe is treated as the legacy four-step release: the 8-step adapter
defaults to nine schedule points with shifts 12/3 and also accepts its
published five-point fallback, while the 768p adapter uses five points, shifts
6/3, and alpha 128. The recommended 768p canvas is 1344x768. H3 Turbo
adapters cannot be combined with Ref2VA or denoise-step cache reuse. They can
use the attention-only `balanced` and `maximum` paths; every scheduled
evaluation still executes all 50 blocks.

`--h3-acceleration quality` executes every transformer block and preserves the
native same-seed trajectory. At packed sequences of at least 12,000 tokens,
the explicit `balanced` and `maximum` modes add an independently implemented
Apple Metal dynamic-sparse attention path inspired by
[Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/). Text, conditioning video,
and generated-audio prefix queries stay on MLX's dense fused attention path.
Every target-video query keeps prefix keys and neighboring 64-token video
blocks exact. Query-dependent high-score blocks are also exact; skipped blocks
still contribute through key centroids and summed values in the same online
softmax accumulator. The first two transformer layers, leading denoise region,
and final evaluation remain dense. A once-per-shape all-dense Metal comparison
must pass before the sparse path can run.

Without a Turbo adapter, `balanced` and `maximum` also trade exact trajectory
identity for speed through a modality-aware adaptive cache.
Every step executes the first transformer block. The runtime then compares its
target-only residual with the last full refresh using separate global and
worst-time-slice drift measurements for video and audio. A cache hit reuses the
target residual from blocks 2 through 50; a rejected or non-finite measurement
executes all 50 blocks and refreshes both residuals.

`balanced` admits global drift up to 0.08 and worst-time-slice drift up to 0.12,
allows at most two adjacent cache hits, and reserves the final two evaluations
for full refreshes. The measured `maximum` envelope is 0.30 global and 0.40
worst-time-slice drift, with at most four adjacent hits and a mandatory final
full evaluation. Both modes require two complete evaluations before the first
reuse. Cache telemetry reports all four drift measurements and exact executed
block counts. `MERERUN_H3_CACHE_STRATEGY=scheduled-tail` retains the former
fixed-depth policy only as a reproducible benchmark baseline.

Long-form generation uses a typed global window plan. Window frame counts keep
the target `17*n+5` geometry and overlap counts use `17*n+1`, so all history
except the last boundary frame encodes into complete 17-frame video chunks.
The boundary becomes the next first-frame condition. Its exact overlap
waveform is encoded once, divided into audio history and two boundary latents,
and packed beside video history before the target A/V rows. Only the generated
target after that boundary is appended. Global frame injection is localized
per window without changing its requested output index. The generator retains
the Qwen conditioner, transformer and AdaLN schedule, both VAEs, and Ref2VA
reference encodings across the complete rollout.

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
The explicit-pull Ref2VA package is published as
`Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit` and pinned to Hub commit
`61dc387ef1a7166425cdacd63c2340598dcc364f`. Its transformer comes from the exact
`Comfy-Org/MiniMax-H3@fd70b39279d1ae6eb214c903f53e1bec3af19a77` ConvRot
source through `scripts/model-conversion/convert_minimax_h3_convrot.py`. The
converter reads each tensor's embedded ConvRot group size (256 for the 200
transformer matrices and 64 for the 50 AdaLN matrices) independently from the
MLX affine output group size of 64. The published transformer is affine
INT8/group-64, exactly 36,024,412,656 bytes with SHA-256
`234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2`.
The conditioner is also affine INT8/group-64. Lower Ref2VA precision did not
meet the visual quality bar, so 8-bit is the supported floor. The package is
self-contained and retains its source-bound 31-point AdaLN cache, source
manifest, conversion receipt, hashes, license, notice, and modification
disclosure. The cache is tied to the immutable transformer SHA-256 and lets a
normal managed pull omit the schedule-only AdaLN/time-embedding branch without
running `model optimize` locally.

The implementation follows the published MiniMax/Hugging Face architecture.
No ComfyUI source is included or used at runtime.
