# Video Runtime

Generate a clip from a prompt. With a unified A/V checkpoint, the model writes
the soundtrack in the same pass rather than layering it on afterwards. Anchor
the clip to a start frame, steer it toward an end frame, or condition the motion
on a song you already have. Recast a performer in an existing shot. Keep the
checkpoint resident and render take after take without paying for the reload.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run video generate` | Generate MP4 video with native Swift/MLX video models. |
| `mere.run video cosmos3` | Run native NVIDIA Cosmos3-Edge generation and action modes. |
| `mere.run video animate` | Animate or replace a masked subject with native Swift/MLX SCAIL-2. |
| `mere.run video prepare-masks` | Prepare reviewable, palette-safe SCAIL-2 masks with native SAM 3.1. |
| `mere.run video session` | Keep an LTX 2.3 runtime resident for JSONL generation requests. |
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

Advanced Video remains the typed home for Cosmos3 generation/action modes, raw
mask-plan execution, LTX latent export, and resident LTX sessions. Raw arguments
are an escape hatch, not the capability contract.

## Model family

- `video-minimax-h3-fl2va-mlx`: explicit-pull MiniMax-H3 FL2VA package with a
  direct-from-official Q4 transformer core, Q8 conditioner, and bundled
  source-bound AdaLN cache. It generates 24 fps RGB and synchronized 32 kHz
  stereo audio from text, a first frame, or directed first/last frames. Frame
  counts follow `17*n+5`; width and height are multiples of 32.
- `video-minimax-h3-fl2va-bf16-mlx`: maximum-fidelity FL2VA package. It streams
  PipeNetwork's pinned 13-shard BF16 MLX transformer directly, discards the
  schedule-only AdaLN tensors after building an exact cache for the requested
  schedule, and deinterleaves the released per-head QKV layout in memory. The tokenizer,
  Q8 conditioner, FP16 video VAE, and FP32 audio VAE come from the same pinned
  native support package. The managed install is about 100 GB on disk; the
  active transformer is about 40 GB after the cached AdaLN tensors are omitted.
- `video-minimax-h3-ref2va-mlx`: identifier for a locally converted Ref2VA
  root. Repeated `--reference image:path|video:path|audio:path` options retain
  request order. Video soundtracks are conditioned with their video; a
  standalone audio reference must be paired with an image or video. There is
  no managed Ref2VA download because mere.run does not publish a converted
  copy of the territory-restricted weights.
- `video-ltx23-av-mlx`: standalone distilled LTX 2.3 MLX checkpoint for fast
  drafts. This is the default `--quality draft` checkpoint.
- `video-ltx23-full-mlx`: LTX 2.3 dev checkpoint, official distilled LoRA,
  vocoder, VAEs, and x2 upscaler. This is the shared quality bundle for both
  `--quality final`, generated synchronized audio, and source-audio A2Vid.
- `video-ltx23-a2vid-mlx`: compatibility ID for existing A2Vid installs. New
  installs should use `video-ltx23-full-mlx`.
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
  --width 1280 --height 768 --duration 10 --steps 20 \
  --output ./bus-stop-bf16.mp4

# Two four-evaluation adapter lanes for the BF16 FL2VA model
mere.run adapter pull minimax-h3-turbo-4step
mere.run adapter pull minimax-h3-lightx2v-4step
mere.run video generate "a superhero waits beneath an umbrella at a bus stop" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --h3-adapter minimax-h3-lightx2v-4step \
  --output ./bus-stop-turbo.mp4

mere.run video generate "preserve the person and use the reference motion" \
  --model-root ./MiniMax-H3-Ref2VA-MLX \
  --reference image:./person.png \
  --reference video:./motion.mp4 \
  --output ./ref2va.mp4
```

MiniMax-H3 is CFG-distilled, so `--steps` is a schedule-point count without a
second unconditional model evaluation. The Community License excludes use,
distribution, and display in the United States, European Union, United
Kingdom, and Republic of Korea. Read and accept the model terms before pulling,
converting, using, or redistributing any artifact. Passing
`--accept-model-license` and continuing with the download confirms that you
accept those terms and agree to comply with them.

The `minimax-h3-turbo-4step` EMA-850 adapter and
`minimax-h3-lightx2v-4step` LightX2V adapter are checksum-pinned separately.
EMA-850 remains an activation-space adapter because its AdaLN deltas
participate in schedule-cache construction. LightX2V has no AdaLN targets, so
the runtime applies its published alpha/rank scale and fuses its deltas into
the BF16 transformer once before denoising; the PEFT tensors are then released
and add no per-block LoRA matmuls. Both default to five schedule points, which
are four model evaluations. They were trained for FL2VA, require the BF16 base
model, and cannot be combined with Ref2VA references or H3 tail-residual reuse.
Omit `--steps` (or set it to `5`) and keep `--h3-acceleration quality`.
Strength `1.0` applies either adapter's released weights exactly; the strength
control remains available for prompt-specific tuning.

The managed package already includes the inference-only AdaLN cache computed
from the official BF16/F32 projections; no post-pull optimization step is
required. Generation skips loading and executing the transformer's
13B-parameter AdaLN/time-embedding branch and resamples its exact released
31-point curve for the selected schedule. By default H3 uses 9 points through
13,500 packed rows, 16 through 26,000, and 21 above that. Maximum acceleration
caps the automatic schedule at 12 points; `--steps` remains an
explicit schedule-point override. Compact cache-backed INT4 transformers
therefore remain correct at arbitrary point counts. MacBooks below 96 GiB keep
Q4 resident by default; memory-qualified desktops and 96+ GiB MacBooks select the
faster resident BF16 path when the requested geometry leaves the required
runtime reserve. `--h3-weight-mode` can force either path. `model optimize`
remains available for compatible locally converted roots that still contain
the full AdaLN branch.

The denoise policy is independently selectable. `--h3-acceleration quality`
executes all 50 blocks and is the exact default. `balanced` recomputes the
leading 50% of blocks on eligible small-sigma-delta steps and refreshes after
at most two cache hits. `maximum` is the explicitly approximate speed lane: it
recomputes the leading nine of 50 blocks and refreshes after at most four cache
hits. Both modes run at least two complete evaluations before reuse and retain
full evaluations at the schedule boundaries. They can materially reduce
denoise time, but they are
approximate and may change motion or composition for the same prompt and seed.

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
PNG corrections. White is background; legal subject colours are blue, red,
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

Decoded mask pixels are snapped to the nearest legal colour inside a strict
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
For the current split-layout LTX 2.3 model, it retains the joint AV denoising
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

For LTX 2.3 audio/video, pull the managed model id and let it install its Gemma
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

Use `--duration` rather than hand-pairing `--num-frames` and `--fps` for
representative unified AV tests. LTX 2.3 expects 24 fps timing; for example,
15 seconds resolves to 361 frames at 24 fps because LTX frame counts must
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

### Resident LTX 2.3 generation

`video session` amortizes checkpoint loading across serial synchronized-AV
generations on either the standalone distilled model or the full dev model. It
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

## Source Reading Notes

The LTX runtime has more low-level model, media, and checkpoint-layout code
than most other runtime families in this repo. Start from the public generation
flow before reading the lower-level model definitions.

The best reading order is:

1. public generation types and `LTXDistilledLatentGenerator`
2. request normalization and generation flow
3. denoise and latent-conditioning helpers
4. decoding and media assembly code
5. lower-level model definitions

If you are new to the repo, use [Architecture Reading Map](../architecture.md)
before diving directly into the LTX implementation.
