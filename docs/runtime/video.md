# Video runtime

Use the video runtime to generate a clip from a prompt. With a unified
audiovisual (AV) checkpoint, the model writes the soundtrack in the same pass.
Anchor the clip to a start frame, steer it toward an end frame, condition its
motion on a song, or recast a performer in an existing shot. Keep the checkpoint
resident to render multiple takes without reloading it.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run video generate` | Generate MP4 video with native Swift/MLX video models. |
| `mere.run video cosmos3` | Run native NVIDIA Cosmos3-Edge generation and action modes. |
| `mere.run video animate` | Animate or replace a masked subject with native Swift/MLX SCAIL-2. |
| `mere.run video prepare-masks` | Prepare reviewable, palette-safe SCAIL-2 masks with native SAM 3.1. |
| `mere.run video session` | Keep an LTX 2.3 or LTX 2.5 runtime resident for JSONL generation requests. |
| `mere.run video export-latents` | Run native Swift/MLX distilled LTX denoising and export final latents. |

## macOS Studio

The optional macOS app compiles against the same typed capability contract
emitted by:

```bash
mere.run catalog video.generate --json
```

The primary Video composer exposes independent quality and output choices,
text-to-video, a start image, an end keyframe, source-audio A2V with segment
offset and modality guidance, duration or frame-count targeting, negative
prompts, Wan controls, preflight, and timing receipts. Attaching source audio
selects final quality plus synchronized audio-video output automatically.

Video's **SCAIL** button opens a first-class Subject Studio. It authors one to
six reference subjects with text, box, and point selectors; previews palette-safe
SAM 3.1 masks; tracks the complete driving clip; accepts immutable keyframe
corrections or painted binary masks; and shows reference, driving, contact-sheet,
and result playback together. Animation/replacement semantics, trim range,
dimensions, profile, adapter, seed, segmentation/overlap, tail, audio, output,
and preflight controls map directly to the public CLI contract. Every mask and
animation run remains live and restartable in Library.

**Advanced Video** provides typed controls for Cosmos3 generation and action modes, raw
mask-plan execution, LTX latent export, and resident LTX sessions. Raw arguments
are an escape hatch, not the capability contract.

## Model family

- `video-minimax-h3-fl2va-mlx`: legacy compatibility MiniMax-H3 FL2VA package
  with a direct-from-official Q4 transformer core, Q8 conditioner, and bundled
  source-bound AdaLN cache. It generates RGB video at 24 frames per second and
  synchronized 32 kHz stereo audio from text, a first frame, or directed
  first and last frames. Frame
  counts follow `17*n+5`; width and height are multiples of 32. The package remains
  installable but is no longer the recommended quality path.
- `video-minimax-h3-fl2va-bf16-mlx`: maximum-fidelity compact FL2VA package.
  Its official-source BF16 denoising core omits the schedule-only AdaLN,
  timestep-MLP, and reconstructed RoPE tensors and uses a source-bound pack of
  exact production schedules. The Q8 conditioner, FP16 video VAE, and FP32
  audio VAE are stored in the same single-root immutable artifact.
- `video-minimax-h3-fl2va-8bit-mlx`: smaller high-quality FL2VA fallback. It
  uses MLX affine INT8/group-64 for eligible core linears while retaining the
  same conditioner, VAEs, cache pack, and provenance as compact BF16. Q8 is a
  storage and memory option, not an advertised speedup without measurements.
- `video-minimax-h3-fasth3-vsa-datafree-mlx`: dedicated four-evaluation FastH3
  package with the compact BF16 base, student adapter, AdaLN sidecar, and Metal
  VSA-H3 sparse attention in one explicit license-accepting pull.
- `video-minimax-h3-ref2va-mlx`: explicit-pull 8-bit Ref2VA package. Repeated
  `--reference image:path|video:path|audio:path` options retain request order.
  Video soundtracks are conditioned with their video; a standalone audio
  reference must be paired with an image or video. The transformer and Qwen
  conditioner use MLX affine INT8/group-64; 8-bit is the published Ref2VA
  quality floor. Its source-bound AdaLN cache is bundled, so no post-pull
  `model optimize` step is required.
- `video-ltx23-av-mlx`: standalone distilled LTX 2.3 MLX checkpoint for fast
  drafts. This is the default `--quality draft` checkpoint.
- `video-ltx23-full-mlx`: LTX 2.3 dev checkpoint, official distilled LoRA,
  vocoder, VAEs, and x2 upscaler. This is the shared quality bundle for both
  `--quality final`, generated synchronized audio, and source-audio A2Vid.
- `video-ltx23-a2vid-mlx`: compatibility ID for existing A2Vid installs. New
  installs should use `video-ltx23-full-mlx`.
- `video-ltx25-distilled-bf16`: public, self-contained LTX 2.5 distribution for
  native synchronized video and stereo audio. The transformer, VAEs, upsampler,
  and duration head remain BF16; the bundled Gemma 4 language tower is MLX Q4.
  Pull it once with `--accept-model-license`; no Hugging Face gate, companion
  download, or local conversion is required. Gemma and the image VAE encoder
  are released after their conditioning tensors are evaluated, before
  denoising begins.
- `video-ltx25-full-bf16`: complete official LTX 2.5 BF16 root with dev and
  distilled transformers, distilled LoRA, convolutional and diffusion VAEs,
  spatial/temporal upsamplers, and DurationHead. It powers the native full,
  HQ, DFR, A2Vid, Retake, HDR/EXR, Dub-It, and text-to-audio workflows.
- `video-ltx-av`: legacy merged LTX root, superseded by LTX 2.3. Only still required
  by `video export-latents`; not recommended for `video generate`.
- `video-wan22-ti2v-5b-mlx`: native Wan2.2 TI2V image-to-video lane. `video
  generate --model video-wan22-ti2v-5b-mlx` requires `--image` (it animates a
  start frame and rejects `--end-image`), snaps width/height to a 32px grid and
  frame counts to `4n+1`, and takes `--steps` (40), `--guidance-scale` (5), and
  `--shift` (5).
- `video-scail2-14b-mlx`: separately packaged MIT-licensed SCAIL-2 14B MLX
  bundle for reference- and mask-conditioned subject animation/replacement with
  long-video clean history.
- `video-cosmos3-edge-mlx`: official pinned NVIDIA Cosmos3-Edge snapshot for
  image/video generation and editing, multimodal reasoning, policy,
  forward/inverse dynamics, and camera-controlled world transitions. Its model
  weights are governed by OpenMDW-1.1.

## Typical workflows

### MiniMax-H3 synchronized video and audio

```bash
mere.run model pull video-minimax-h3-fl2va-mlx --accept-model-license
mere.run video generate "a glass marble rolls across wood with a delicate rattle" \
  --model video-minimax-h3-fl2va-mlx \
  --image ./marble.png \
  --num-frames 124 \
  --output ./marble-h3.mp4

# Maximum-fidelity BF16 transformer; same native Studio and CLI workflow
mere.run model pull video-minimax-h3-fl2va-bf16-mlx --accept-model-license
mere.run video generate "a cinematic rain-soaked bus stop at night" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --width 1280 --height 768 --duration 10 --steps 21 \
  --output ./bus-stop-bf16.mp4

# Smaller high-quality Q8 core; the same FL2VA adapters are supported
mere.run model pull video-minimax-h3-fl2va-8bit-mlx --accept-model-license

# Published adapter recipes for compact BF16 or Q8 FL2VA
mere.run adapter pull minimax-h3-turbo-4step
mere.run adapter pull minimax-h3-lightx2v-4step
mere.run adapter pull minimax-h3-lightx2v-8step-v1
mere.run adapter pull minimax-h3-lightx2v-4step-v1-768p
mere.run adapter pull minimax-h3-lightx2v-ref2v-4step-v0.1

# The dedicated FastH3 package includes its base, adapter, and AdaLN cache.
mere.run model pull video-minimax-h3-fasth3-vsa-datafree-mlx \
  --accept-model-license
mere.run video generate "a lighthouse in a winter storm" \
  --model video-minimax-h3-fasth3-vsa-datafree-mlx \
  --output ./lighthouse-fasth3.mp4

mere.run video generate "a superhero waits beneath an umbrella at a bus stop" \
  --model video-minimax-h3-fl2va-8bit-mlx \
  --width 960 --height 544 \
  --h3-adapter minimax-h3-lightx2v-8step-v1 \
  --output ./bus-stop-turbo-8step.mp4

mere.run video generate "a superhero waits beneath an umbrella at a bus stop" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --width 1344 --height 768 \
  --h3-adapter minimax-h3-lightx2v-4step-v1-768p \
  --output ./bus-stop-turbo-4step-768p.mp4

mere.run model pull video-minimax-h3-ref2va-mlx --accept-model-license
mere.run video generate "preserve the person and use the reference motion" \
  --model video-minimax-h3-ref2va-mlx \
  --reference image:./person.png \
  --reference video:./motion.mp4 \
  --h3-adapter minimax-h3-lightx2v-ref2v-4step-v0.1 \
  --h3-weight-mode resident-bf16 \
  --output ./ref2va.mp4

# Long FL2VA or Ref2VA shots keep the H3 runtime resident and condition each
# new window on overlapping motion plus its matching generated soundtrack.
mere.run video generate "one continuous tracking shot through a night market" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --duration 15 \
  --h3-window-frames 124 \
  --h3-window-overlap 35 \
  --h3-acceleration maximum \
  --output ./market-long.mp4

# Exact zero-based FL2VA frame injection can direct intermediate beats.
mere.run video generate "the actor crosses three connected sets" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --num-frames 175 \
  --h3-frame 72:./second-set.png \
  --h3-frame 144:./third-set.png \
  --output ./three-sets.mp4
```

MiniMax-H3 is CFG-distilled, so `--steps` is a schedule-point count without a
second unconditional model evaluation. The Community License excludes use,
distribution, and display in the United States, European Union, United
Kingdom, and Republic of Korea. Read and accept the model terms before pulling,
converting, using, or redistributing any artifact. Passing
`--accept-model-license` and continuing with the download confirms that you
accept those terms and agree to comply with them.

The `minimax-h3-turbo-4step` EMA-850 adapter and all five LightX2V releases
are checksum-pinned separately.
EMA-850 remains an activation-space adapter because its 51 AdaLN deltas
participate in schedule-cache construction. For compact models those deltas are
computed from `silu(cached time embedding)` and added to an in-memory augmented
table keyed by source, schedule, adapter hash, and strength. LightX2V standard
and QKV projections also run as activation-space low-rank wrappers around dense
or stock MLX quantized linears; base weights are never expanded or fused. The
legacy releases default to five schedule
points, which are four model evaluations. LightX2V v1.0 8-step defaults to nine
schedule points (eight evaluations), accepts the upstream four-evaluation
fallback, and uses video/audio shifts 12/3 with alpha 8. The LightX2V v1.0
four-step 768p release uses five schedule points, shifts 6/3, and alpha 128.
The eight-step 768p release uses nine schedule points, shifts 6/3, and alpha 8;
it doesn't accept the earlier release's four-evaluation fallback. Both 768p
releases target a 1344x768 canvas. EMA-850 and the four LightX2V FL2VA adapters
support compact BF16 and Q8, reject legacy Q4, and cannot be combined with
Ref2VA references. The separate
`minimax-h3-lightx2v-ref2v-4step-v0.1` release targets Ref2VA, uses five schedule
points with shifts 12/3 and alpha 8, and requires ordered references. mere.run
expands the managed INT8 Ref2VA transformer to resident BF16 before fusing the
adapter, so pass `--h3-weight-mode resident-bf16` unless automatic admission is
known to qualify. Forced quantized execution is rejected. No H3 adapter can use
denoise-step cache reuse. Omit `--steps` to select the pinned recipe.
`--h3-acceleration quality`
keeps the fully dense path; `balanced` and `maximum` may use attention-only dynamic
sparsity while still executing all 50 blocks on every model evaluation.
Strength `1.0` applies an adapter's released weights exactly; the strength
control remains available for prompt-specific tuning.

The pinned compact BF16 and Q8 model roots contain the exact nine-point
shifts-6/3 AdaLN table used by the eight-step 768p adapter. The refreshed table
closes against the pinned official source on MLX Metal. A tensor comparison
against the former interpolation path covered 116,444,160 values: 88.40% were
already exact, while the remaining values reached 0.125 maximum absolute error
and 0.001427 RMSE. This proves why the exact table is preferable; it doesn't by
itself constitute a same-seed visual or audio quality qualification.

`video-minimax-h3-fasth3-vsa-datafree-mlx` is a self-contained FastVideo student
package. One explicit model pull installs the compact BF16 base, FastH3 adapter,
and source-bound AdaLN cache. Generation doesn't require another model or
adapter download. The recipe requires adapter strength `1.0`. Its
four denoising evaluations use base sigma points `0.999`, `0.749`, `0.5`, and
`0.25`, followed by the clean endpoint. MLX Metal runs FastH3's 64-token VSA
tiles, per-head top-k routes at 90% video-key sparsity, and learned pooled-value
compression gates. Prefix queries remain dense, and video queries retain every
prefix key tile. The recipe accepts text-only generation. It rejects frame
conditioning, continuation, references, and additional H3 approximation modes.

The package build uses
`scripts/model-conversion/prepare_minimax_h3_fasth3_vsa.py` to reconstruct the
schedule-only AdaLN values that the compact BF16 base omits. The script verifies
the adapter, range-reads about 26 GiB from the pinned student transformer, and
writes only the approximately 120 MB source-bound cache. End users who pull the
dedicated managed model don't run this build step.

`--h3-frame FRAME:PATH` adds an FL2VA keyframe at an exact zero-based output
frame. Values may be repeated up to the released 12-frame condition limit and
remain on the global timeline when sliding windows are enabled.
`--h3-window-frames` enables resident long-form generation for FL2VA or Ref2VA.
The window count must use H3's `17*n+5` target geometry; overlap supplied by
`--h3-window-overlap` must use `17*n+1` and leave at least 22 frames for each
new target. Each continuation encodes all overlap frames except the boundary
as motion history, uses the final overlap frame as the next first-frame
condition, and carries the corresponding 32 kHz stereo waveform as history
and boundary audio latents. Only new frames and samples are appended, so the
final MP4 has the exact aligned global duration. Conditioner, transformer,
AdaLN table, reference encodings, and both VAEs remain resident across windows.

The compact BF16, Q8, legacy Q4, and Ref2VA managed packages include
source-bound inference-only AdaLN caches; no post-pull optimization is
required. Compact BF16 and Q8 packages select exact tables for 5, 9, 12, 16, 21,
or 31 points at shifts 12/3 and the LightX2V 768p 5- and 9-point schedules at
shifts 6/3. A custom schedule interpolates from the densest table and emits a visible
non-bit-exact diagnostic. Generation skips the 13B-parameter
AdaLN/time-embedding branch. By default H3 uses 9 points through
13,500 packed rows, 16 through 26,000, and 21 above that. Maximum acceleration
caps the automatic schedule at 12 points; `--steps` remains an
explicit schedule-point override. MacBooks below 96 GiB keep quantized weights
resident by default; memory-qualified desktops and 96+ GiB MacBooks select the
faster resident BF16 path when the requested geometry leaves the required
runtime reserve. `--h3-weight-mode` can force either path. `model optimize`
remains available for compatible locally converted roots that still contain
the full AdaLN branch.

The denoise policy is independently selectable. `--h3-acceleration quality`
uses dense attention, executes all 50 blocks, and is the exact default. At
12,000 or more packed rows, `balanced` and `maximum` dynamically route distant
target-video attention blocks on Apple GPUs. Prefix queries, all prefix keys,
neighboring video blocks, the first two transformer layers, the leading
schedule region, and the final evaluation remain dense. Skipped blocks retain
a centroid-and-summed-value correction in the online-softmax accumulator, and
a once-per-shape dense-route gate must pass before sparse execution is admitted.

Without an H3 adapter, those two modes additionally use the modality-aware
adaptive first-block cache. Every evaluation executes block 1 and measures
global plus worst-time-slice drift independently for video and audio. A cache
hit reuses only the target residual from blocks 2 through 50. `balanced`
refreshes after at most two hits and reserves the final two evaluations;
`maximum` refreshes after at most four hits and reserves the final evaluation.
Both require two complete evaluations before reuse. Dynamic attention and cache
reuse are approximate and may change motion or composition for the same prompt
and seed; use `quality` when exact-seed fidelity matters.

`--h3-acceleration velocity-reuse-2` is an isolated experimental bake-off arm.
It keeps the quality schedule, protects the first and final full evaluations,
and linearly extrapolates the complete synchronized video/audio velocity from
the two most recent full evaluations on intervening odd steps. Video and audio use
their independent shifted schedules, and the extrapolation ratio is clamped to
`[-2, 2]`. It disables the other approximation policies so its timing and
quality deltas can be attributed directly; it is not an automatic or default
mode.

Ref2VA reference images preserve source aspect ratio and are downscaled only
when their area exceeds the internal render canvas, with both dimensions
rounded to 32-pixel multiples. Standalone reference audio keeps its complete
2-15 second duration rather than being truncated to target-video duration;
ordered reference audio remains capped at 15 seconds in total. H3 condition
augmentation and target video/audio latents use the released independent
seeded streams and native latent layouts.

The isolated `layers-45` and `layers-40` arms rank mean absolute attention and
MLP gates from the exact AdaLN table, protect blocks 0, 1, and 49, and skip the
lowest remaining blocks. They reduce executed block work but keep all
weights loaded, so no residency reduction is claimed.

`--h3-acceleration token-reduction` preserves the complete packed prefix and
target audio while pairing adjacent horizontal target-video tokens after block
3. It uses averaged spatial RoPE coordinates on the reduced grid. During the
first ten denoise evaluations the reduced path continues through block 39;
later evaluations restore before block 30. Each full-resolution video token is
reconstructed from its saved value plus the corresponding reduced token's
update from the pooled baseline, preserving within-pair detail. The arm is
isolated from the other approximation policies and is not a default.

`--h3-render-width` and `--h3-render-height` select an explicit internal target
canvas for FL2VA or Ref2VA. Set both; they must be same-aspect 32px multiples no
larger than the requested output. DiT and VAE decode operate on that smaller
grid. The decoded RGB frames are then returned to `--width` and `--height` with
the pinned h3.c high-quality vImage scaling contract. A 512x512 output can use
384 x 384 for the 75% arm or 320 x 320 for the 62.5% arm. Reduced rendering is
rejected with sliding windows because continuation conditioning has
not yet been resampled and qualified.

### Cosmos3 generation, actions, and reasoning

```bash
mere.run model pull video-cosmos3-edge-mlx

mere.run video cosmos3 "the camera enters the open doorway" \
  --mode image-to-video \
  --image ./doorway.png \
  --width 512 --height 320 --num-frames 17 \
  --output ./enter-doorway.mp4

mere.run video cosmos3 "Identify safe routes through the room." \
  --mode reasoner \
  --image ./doorway.png
```

Use `mere.run guide video-cosmos3` for the full mode matrix and action-domain
contract. The same runtime powers `world serve --backend cosmos3`, which keeps
the transformer, VAE, and terminal latent resident across transitions.

### SCAIL-2 subject animation and replacement

Install the immutable Sawfwair MLX snapshot and the separately licensed SAM
3.1 model:

```bash
swift run mere.run model pull video-scail2-14b-mlx
swift run mere.run model pull vision-segment-sam31 --accept-model-license
```

For interactive recast work, also pull the checksum-pinned Apache-2.0
LightX2V Wan 2.1 I2V adapter. Adapter weights remain in the local adapter store
and are not bundled with mere.run:

```bash
swift run mere.run adapter pull scail2-lightx2v-4step
```

Create a schema-version 1 mask plan, preview one target frame, and then prepare
the immutable full mask revision:

```bash
swift run mere.run video prepare-masks \
  --plan ./mask-plan.json \
  --output-dir ./mask-preview \
  --preview-frame 12 \
  --json

swift run mere.run video prepare-masks \
  --plan ./mask-plan.json \
  --output-dir ./approved-mask-candidate \
  --json
```

The full result contains an exact-geometry/FPS driving proxy, per-subject
prepared reference images and masks, a ProRes 4444 categorical mask video,
overlay preview, contact sheet, tracking and quality reports, and a canonical
SHA-256 manifest. Set the plan's `mode` to `replacement` to aspect-fit every
reference into the target canvas and matte all non-subject pixels to black;
`animation` preserves the fitted reference scene. The plan
supports one to six stable subjects, text/box/point selectors, and dense painted
PNG corrections. White is background; legal subject colors are blue, red,
green, magenta, cyan, and yellow.

The native transformer preserves upstream global self-attention while splitting
the query axis into independently evaluated Metal command buffers. Every query
slice still attends to the complete key/value sequence; the split only keeps a
full 81-frame window below the macOS GPU watchdog and does not introduce local
or sliding-window attention.

Prepared review artifacts always use white as the canonical background. During
inference, `video animate` matches the official SCAIL-2 mask-role semantics:
animation keeps the main reference background visible and hides the driving
background, while replacement hides the reference background and keeps the
driving scene visible. Additional subject-reference backgrounds are hidden.
Subject colors and reference-to-driving correspondence are unchanged.

After reviewing the mask artifacts, preflight and render natively:

```bash
swift run mere.run video animate \
  "a dancer in a red silk dress" \
  --reference ./reference-performer-prepared.png \
  --reference-mask ./reference-performer-mask.png \
  --driving-video ./pose.mp4 \
  --driving-mask ./pose-mask.mp4 \
  --preflight --json

swift run mere.run video animate \
  "a dancer in a red silk dress" \
  --reference ./reference-performer-prepared.png \
  --reference-mask ./reference-performer-mask.png \
  --driving-video ./pose.mp4 \
  --driving-mask ./pose-mask.mp4 \
  --mode animation \
  --tail-policy pad-trim \
  --audio-source driving \
  --output ./animated.mp4
```

The default `--profile fast` managed four-step path selects the pinned
`scail2-lightx2v-4step` adapter, disables classifier-free guidance, uses shift
5, Euler updates, and the adapter's exact `1000, 750, 500, 250` training-step
schedule at 832x480. `--profile quality` remains available as an explicit
opt-in to the configurable 40-step UniPC/CFG recipe:

Preflight resolves that managed default without loading it, then verifies the
installed file's exact byte count and SHA-256. A missing or corrupt adapter is
reported in `missing_input_files`; generation keeps strict resolution and
cannot continue with an unverified file.

```bash
swift run mere.run video animate \
  "a silver puppet follows the dancer" \
  --reference ./reference-performer-prepared.png \
  --reference-mask ./reference-performer-mask.png \
  --driving-video ./pose.mp4 \
  --driving-mask ./pose-mask.mp4 \
  --mode replacement \
  --tail-policy pad-trim \
  --audio-source driving \
  --output ./recast-fast.mp4
```

SCAIL-2's unified motion interface has a task-specific 20-channel input
projection, while the generic Wan 2.1 I2V adapter carries a 36-channel
projection difference. The native fuser requires every shared transformer
target to match exactly, applies the compatible 487 LoRA pairs and parameter
differences, and explicitly leaves only that incompatible input-projection
weight untouched. It does not use a general skip-mismatch mode.

Decoded mask pixels are snapped to the nearest legal color inside a strict
tolerance, and ambiguous or out-of-tolerance pixels are rejected. The default
long-video contract uses 81-frame segments with five decoded frames of clean
overlap. Compatibility defaults remain `--tail-policy drop` and
`--audio-source none`; `pad-trim` keeps already legal `1 mod 4` temporal
lengths unchanged, pads only an incomplete final segment to the next legal
latent length, trims the result to the exact requested frame count, and
`driving` muxes source audio to that exact duration. It never expands a legal
65-frame clip to an unused 81-frame window.

Use `--mode replacement` to place the reference subject into the driving
scene. Pair repeatable `--additional-reference` and
`--additional-reference-mask` values for multi-subject conditioning. The
driving video and driving-mask video must have identical decoded frame counts,
and reference/mask ordering is preserved.

### Fast visual draft

The default `--quality draft` lane is the speed path. It generates video-only
MP4s by default and is the right first pass for prompt, camera, subject, and
composition checks.
For the split-layout LTX 2.3 model, it retains the joint AV denoising
tokens that influence video through audio-to-video cross attention, but skips
loading and decoding the audio VAE/vocoder and writes no audio stream.

```bash
swift run mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --num-frames 65 \
  --output ./clip.mp4
```

### Directed image-to-video

Use `--image` to anchor the first latent frame. Add `--end-image` when the clip
should move toward a specific final keyframe.

```bash
swift run mere.run video generate \
  "a car drives from a bright morning street into a warm sunset road, smooth forward motion" \
  --image ./car-start.png \
  --image-strength 0.9 \
  --end-image ./car-end.png \
  --end-image-strength 0.85 \
  --num-frames 65 \
  --output ./car-start-to-end.mp4
```

`--end-image` requires `--image`; the start conditioning wins if very short
clips make the start and end latent conditioning windows overlap.

### Final-quality render

Quality and output are independent choices. `--quality final` uses the full
dev + distilled-LoRA two-stage pipeline and still writes a video-only MP4 by
default:

```bash
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "a red fox runs across a snowy clearing, detailed winter fur, natural motion" \
  --quality final \
  --duration 4 \
  --output ./fox-final.mp4
```

This is the final-quality checkpoint with audio components omitted from the
deliverable. It is not a separate speed optimization; the meaningful speed
change comes from selecting the draft checkpoint.

### Synchronized AV final render

For LTX 2.3 audio and video, pull the managed model ID and let it install its Gemma
3 companion:

```bash
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "dialogue with clean background music" \
  --quality final \
  --output-mode audio-video \
  --duration 15 \
  --fps 24 \
  --output ./ltx23.mp4
```

Use `--duration` instead of pairing `--num-frames` and `--fps` manually for
representative unified AV tests. LTX 2.3 expects 24 frames per second; for
example, 15 seconds resolves to 361 frames at 24 frames per second because LTX frame counts must
satisfy `8n+1`.

`--quality final --output-mode audio-video` runs the official two-stage quality
path: guided dev denoising for both audio and video at half resolution, followed
by x2 latent upscaling and four-step refinement after fusing the distilled LoRA.
The older `video-ltx-av` merged root is retained only for `video export-latents`.

Add `--timings` to print phase timings for native LTX 2.3 draft, final, or A2Vid
generation. `--timings-output <path>` writes the same typed report as JSON,
including model-component loading, text encoding, each denoising stage, LoRA
fusion where applicable, upsampling, video/audio decode, MP4 writing, unload,
and total wall time.

### Resident LTX generation

`video session` amortizes checkpoint loading across serial synchronized-AV
generations on LTX 2.3 or LTX 2.5, using either the standalone distilled model
or the full dev model. It
reads typed snake-case JSONL requests from stdin and writes one typed JSON
result or error to stdout for each input line.

```bash
printf '%s\n' \
  '{"id":"draft-1","prompt":"a fox runs across snow","output":"./draft-1.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  '{"id":"draft-2","prompt":"a fox runs across snow","output":"./draft-2.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  | swift run mere.run video session --model video-ltx23-av-mlx
```

Use `--model video-ltx23-full-mlx` for the two-stage quality lane. That session
keeps the dev transformer and official distilled LoRA resident, leaves the dev
weights unchanged for Stage 1, and activates the adapter only during Stage 2.
This avoids permanent BF16 weight fusion and permits repeat requests without
checkpoint reload or accumulated weight drift.

LTX 2.5 sessions additionally accept typed `transformer_execution`,
`guidance_projection_cache`, `sampler`, `tea_cache`, `tea_cache_threshold`, and
`tea_cache_calibration_output` request fields. The session-level
`--prompt-cache-capacity` controls exact materialized Gemma connector reuse.
TeaCache is limited to the full two-stage Euler and Res2s paths, remains
disabled by default, and reports computed/reused 48-block stacks in timings.
Its corrected BF16 calibration uses the maximum drift across each synchronized
four-branch guidance group, with conservative default thresholds of `0.235`
for Euler and `0.39` for Res2S. TeaCache is an approximate generation mode:
it is deterministic for a fixed request, but cached and uncached videos are
not pixel-identical. Keep it disabled when exact output reproduction matters.

The first result includes checkpoint-load time. Later results set
`resident_model_reused` to `true` and report zero load time. Full-model requests
still reload the text encoder after the first request so the large transformer,
video decoder, and text encoder do not all remain live during denoising.

### Native source-audio-to-video

Supplying `--audio` selects A2Vid automatically; no LTX variant flag is needed.
The requested segment directly conditions video motion, while that same decoded
source segment becomes the output soundtrack.

```bash
swift run mere.run model pull video-ltx23-full-mlx --accept-model-license
swift run mere.run video generate \
  "the singer performs beneath sweeping blue spotlights" \
  --audio ./song.wav \
  --audio-start-time 42 \
  --duration 6 \
  --image ./artist.png \
  --image-strength 0.9 \
  --output ./shot.mp4
```

The resolved clip duration is the legal `8n+1` frame count divided by `--fps`.
The input must contain at least that much audio after `--audio-start-time`; the
runtime fails rather than padding or silently falling back to text-to-video plus
soundtrack layback. Mono input is duplicated to stereo without auto-gain.

Stage one runs 30 guided full/dev steps by default with frozen audio latents.
Stage two upsamples the video, fuses the official distilled LoRA one tensor pair
at a time, and runs the four upstream distilled sigmas. `--a2v-guidance-scale`,
`--video-cfg-guidance-scale`, `--audio-cfg-guidance-scale`,
`--v2a-guidance-scale`, `--a2v-steps`, and `--negative-prompt` expose the
advanced controls. The `--a2v-guidance-scale` value is also the video stream's
audio-to-video modality guidance in full unified AV.

### Export latents

```bash
swift run mere.run video export-latents \
  "storm clouds over the ocean" \
  --output ./latents.safetensors
```

`video export-latents` still targets the distilled video-only latent path.

Long or high-resolution LTX decodes automatically split the VAE work into
overlapping temporal/spatial tiles. Tile pixels and trapezoidal weights remain
in MLX for device-side accumulation, normalization, and UInt8 conversion; the
runtime no longer reads every float tile back to Swift. Override the automatic
decode budget with `MERERUN_VIDEO_LTX_VAE_DECODE_BUDGET_GB` when validating a
specific memory envelope.

Final MP4 encoding is incremental. The writer transfers one BGRA frame at a
time to FFmpeg stdin or AVAssetWriter and overlaps the next device-to-host frame
transfer with encoder work, avoiding a monolithic host frame buffer and a raw
video spool. Unified-AV audio is likewise transferred and written in aligned
chunks. The decoded MLX frame tensor still exists on device, so this removes
duplicated host-side staging rather than making the generation path
zero-memory. MP4 formats and backend selection are unchanged.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VideoCommand.swift`
- `Sources/MereRunCLI/Commands/VideoAnimateCommand.swift`
- `Sources/MereRunCLI/Commands/VideoSessionCommand.swift`

### Runtime

- `Sources/MereRunCore/LTX/LTXDistilledLatentGenerator.swift`
- `Sources/MereRunCore/LTX/LTXInferenceTimings.swift`
- `Sources/MereRunCore/LTX/LTXGemmaTextEncoder.swift`
- `Sources/MereRunCore/LTX/LTXVideoMP4Writer.swift`
- `Sources/MereRunCore/SCAIL2/SCAIL2Generator.swift`
- `Sources/MereRunCore/SCAIL2/SCAIL2Transformer.swift`

## Source reading notes

The LTX runtime has more low-level model, media, and checkpoint-layout code
than most other runtime families in this repository. Start from the public generation
flow before reading the lower-level model definitions.

Use this reading order:

1. Public generation types and `LTXDistilledLatentGenerator`
2. Request normalization and generation flow
3. Denoise and latent-conditioning helpers
4. Decoding and media assembly code
5. Lower-level model definitions

If you are new to the repository, use the
[Architecture reading map](../architecture.md)
before diving directly into the LTX implementation.
