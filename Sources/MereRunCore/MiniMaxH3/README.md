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
source-bound cache pack stores exact production schedules at 5, 9, 12, 16, 21,
and 31 points for shifts 12/3 plus the LightX2V 5- and 9-point 6/3 schedules. Custom
schedules resample the densest table and emit a visible not-bit-exact
diagnostic. The converted transformer stores global Q/K/V slabs; the
unmodified video VAE retains the released per-head interleave.

Ref2VA image references preserve source aspect ratio and are downscaled only
when their area exceeds the internal render canvas. Standalone audio references
retain their complete 2-15 second span, subject to the released 15-second total
limit. Condition augmentation and target video/audio initialization use the
released independent seeded streams and native latent layouts.

Compact BF16 is the maximum-fidelity lane and affine Q8/group-64 is the smaller
high-quality lane. Legacy Q4 remains installable for compatibility but is no
longer recommended. Memory-qualified hosts may expand Q8 once to resident BF16
for compute-bound denoise; the CLI can force quantized or resident execution.
H3 inference also raises MLX wired residency through the shared ticket
coordinator so weights and activation workspaces do not silently fall out of
the GPU residency set.

The checksum-pinned `minimax-h3-turbo-4step`,
`minimax-h3-lightx2v-4step`, `minimax-h3-lightx2v-8step-v1`, and
`minimax-h3-lightx2v-4step-v1-768p`, and
`minimax-h3-lightx2v-8step-v1-768p` LoRAs run with compact BF16 or affine Q8
FL2VA and reject the legacy Q4 package. `minimax-h3-lightx2v-ref2v-4step-v0.1`
instead targets Ref2VA. Ref2VA behavior is unchanged: its managed package is
expanded to resident BF16 before the adapter is installed, so this recipe
requires `resident-bf16` or a memory-qualified automatic selection; forced
quantized execution is rejected.

The managed `minimax-h3-fasth3-vsa-datafree-4step` release targets only compact
BF16 FL2VA. It installs FastVideo's 5.3 GB student adapter, applies the exact
four-evaluation DMD schedule, and runs its VSA-H3 tile routing and learned
compression gates with MLX Metal. The runtime retains every prefix key tile and
the highest-scoring 10% of video key tiles for each video query tile. Prefix
query tiles remain dense.

FastH3 changes the schedule-only AdaLN weights that the compact BF16 model
omits. The managed `video-minimax-h3-fasth3-vsa-datafree-mlx` package includes a
source-bound cache beside the adapter, so end users need one model pull and no
preparation step. Package builders use
`scripts/model-conversion/prepare_minimax_h3_fasth3_vsa.py --adapter PATH`; the
script range-reads about 26 GiB from the pinned student transformer but doesn't
store the 70 GB transformer or the complete 148 GB checkpoint. FastH3 accepts
text-only FL2VA generation with `--h3-acceleration quality`; it rejects frame
conditioning, continuation, references, non-unit adapter strength, and other
H3 approximation modes.

All standard projection pairs execute as activation-space low-rank wrappers
around the dense or stock MLX quantized base linear. QKV keeps independent
query, key, and value pairs while preserving the runtime's global-QKV slab
order. The EMA-850 adapter's 51 AdaLN pairs augment the selected in-memory
source-bound cache from `silu(cached time embedding)`; the 26 GB base
projections are never restored or fused. Adapter use disables the incompatible
affine-Q8 exact kernels but retains blockwise compilation, adaptive reuse, and
dynamic-sparse scheduling. The v1.0 recipes don't use the legacy four-step
defaults. The 8-step adapter
defaults to nine schedule points with shifts 12/3 and also accepts its
published five-point fallback. The four-step 768p adapter uses five points,
shifts 6/3, and alpha 128. The eight-step 768p adapter uses nine points, shifts
6/3, and alpha 8, and it doesn't accept a five-point fallback. The recommended
768p canvas is 1344x768. The pinned compact roots select an exact nine-point
shifts-6/3 AdaLN table for this recipe. Full BF16 source roots compute the same
table from their AdaLN weights. The Ref2VA v0.1
recipe uses five schedule points, shifts 12/3, and alpha 8. FL2VA adapters
cannot be combined with Ref2VA references, and the Ref2VA adapter requires
them. Adapters retain blockwise compilation, dynamic-sparse attention, and the
existing guarded adaptive-reuse policy in `balanced` and `maximum`; the policy's
warm-up, streak, and final-refresh requirements still apply to the shortened
adapter schedules.

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

`balanced` and `maximum` can also trade exact trajectory identity for speed
through a modality-aware adaptive cache.
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

The legacy explicit-pull FL2VA package is published as
`Sawfwair/MiniMax-H3-FL2VA-MLX-4bit` and pinned to its immutable verified Hub
commit `e1244ad93d60c737c7e0f065a1c9372f3de7caf8`. It is produced only from
`MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027` by
`scripts/model-conversion/convert_minimax_h3_official_mlx.py`; no converted or
quantized third-party checkpoint is a weight input. The package includes the
source manifest and conversion hashes alongside the runtime files. Its 52
fused transformer QKV matrices are deinterleaved from the released per-head
rows into the global Q/K/V slabs expected by the native runtime before Q4
packing. Q4 remains a compatibility path but is no longer recommended for
quality-sensitive generation.

The compact maximum-fidelity BF16 and smaller affine Q8/group-64 packages are
published separately as
`Sawfwair/MiniMax-H3-FL2VA-MLX-BF16@4ce4b1d870f7b1b0c75672fd4f2867c1f5df7b5f`
and
`Sawfwair/MiniMax-H3-FL2VA-MLX-8bit@86500cb6ebec22c006597e41840b26ef1099fdd7`.
Both use only the same pinned official
source, keep the Q8 conditioner, FP16 video VAE, and FP32 audio VAE, and omit
the schedule-only AdaLN, timestep-MLP, and reconstructed RoPE weights. Their
source-bound cache pack contains exact 5, 9, 12, 16, 21, and 31-point tables at
shifts 12/3 plus the LightX2V 5- and 9-point 6/3 tables. Custom schedules interpolate
from the densest table with a visible not-bit-exact diagnostic.
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
