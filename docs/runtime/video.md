# Video runtime

Use the video runtime to generate MP4 clips on your Mac with native Swift and
MLX models. You can generate from a prompt, anchor a start or end frame, write a
synchronized soundtrack in the same pass, condition motion on a song, retake a
timed region of an existing clip, or recast a performer in a shot. You can also
keep a checkpoint resident to render several takes without reloading it.

Every command writes its output path to stdout and its progress and diagnostics
to stderr. To serve a navigable, camera-controlled world instead of finished
clips, see [Persistent world runtime](./world.md).

## Choose a model

The following table maps common goals to the managed model ID and the command
that runs it. Every model is an explicit `mere.run model pull`; the runtime
never downloads weights while it generates.

| Goal | Model ID | Command |
| --- | --- | --- |
| Fast drafts for prompt, camera, and composition checks (default) | `video-ltx23-av-mlx` | `video generate` |
| LTX 2.3 final-quality video, generated audio, or source-audio video | `video-ltx23-full-mlx` | `video generate --quality final` |
| LTX 2.5 synchronized video and stereo audio in a self-contained, ungated bundle | `video-ltx25-distilled-bf16` | `video generate`, `video retake`, `video dub-it` |
| LTX 2.5 full, HQ, DFR, keyframe, IC-LoRA, HDR, and source-audio pipelines | `video-ltx25-full-bf16` | `video generate`, `video retake`, `video dub-it` |
| Animate a start frame with Wan 2.2 | `video-wan22-ti2v-5b-mlx` | `video generate --image` |
| Highest-fidelity MiniMax-H3 video with 32 kHz stereo audio | `video-minimax-h3-fl2va-bf16-mlx` | `video generate` |
| Smaller MiniMax-H3 core with the same adapters | `video-minimax-h3-fl2va-8bit-mlx` | `video generate` |
| Four-evaluation MiniMax-H3 text-to-video | `video-minimax-h3-fasth3-vsa-datafree-mlx` | `video generate` |
| MiniMax-H3 from ordered image, video, and audio references | `video-minimax-h3-ref2va-mlx` | `video generate --reference` |
| Cosmos3 image and video generation, actions, and reasoning | `video-cosmos3-edge-mlx` | `video cosmos3` |
| Animate or replace a masked subject | `video-scail2-14b-mlx` | `video animate` |
| Navigate a persistent generated world | `video-dreamx-world-5b-ar-mlx` | `world serve` |

Three IDs remain only for compatibility:

- `video-ltx23-a2vid-mlx` preserves existing LTX 2.3 source-audio installs.
  Use `video-ltx23-full-mlx` for new pulls.
- `video-ltx-av` is the legacy merged LTX root, superseded by LTX 2.3. Only
  `video export-latents` still requires it.
- `video-minimax-h3-fl2va-mlx` is the legacy Q4 FL2VA package. It remains
  installable, but it didn't meet the release visual-quality bar and refuses
  `--h3-adapter`.

For pinned revisions, download sizes, and license terms, see the video entries
in [Model sources](../model-sources.md#music-sfx-and-video-layouts).

## Commands

| Command | What it does |
| --- | --- |
| `mere.run video generate` | Generate MP4 video with native Swift/MLX video models. |
| `mere.run video retake` | Regenerate a timed video/audio region with native LTX 2.5. |
| `mere.run video dub-it` | Generate synchronized video and audio identity from one LTX 2.5 IC-LoRA reference. |
| `mere.run video session` | Keep an LTX 2.3 or LTX 2.5 runtime resident for JSONL generation requests. |
| `mere.run video cosmos3` | Run native NVIDIA Cosmos3 generation and action modes. |
| `mere.run video animate` | Animate or replace a masked subject with native Swift/MLX SCAIL-2. |
| `mere.run video prepare-masks` | Prepare reviewable, palette-safe SCAIL-2 masks with native SAM 3.1. |
| `mere.run video export-latents` | Run native Swift/MLX distilled LTX denoising and export final latents. |

## Check options against the checkpoint

`video generate`, `video retake`, and `video session` check the command line
against the selected checkpoint's family before they resolve or load it. An
option the family can't run stops the command; an option it accepts but never
reads prints a `Warning:` on stderr and the run continues. The Model Scope
section of `mere.run guide video generate` lists the main `video generate`
cases. For the other two commands:

- `video retake` runs `video-ltx25-distilled-bf16` (the default) or
  `video-ltx25-full-bf16`. The distilled checkpoint warns about
  `--negative-prompt`, `--steps`, and the video and audio guidance options, and
  runs the convolutional decoder whatever `--video-decoder` says. Every other
  video checkpoint stops immediately.
- `video session` runs `video-ltx23-av-mlx` (the default),
  `video-ltx23-full-mlx`, or either LTX 2.5 checkpoint. `--ltx-teacache` stops
  the session on the distilled checkpoints before loading, rather than failing
  each request; LTX 2.3 Full warns that TeaCache has no effect. Wan 2.2 and
  MiniMax-H3 stop immediately.

`video-cosmos3-edge-mlx`, `video-scail2-14b-mlx`, and
`video-dreamx-world-5b-ar-mlx` stop `video generate` immediately; use
`video cosmos3`, `video animate`, or `world serve`.

To print the family a command line selects and what the check reports, without
running it, use `catalog resolve`:

```bash
mere.run catalog resolve -- video generate "a harbor" \
  --model video-wan22-ti2v-5b-mlx --image ./frame.png
```

## Preview a generation

To inspect the selected model, inputs, and output settings before rendering,
add `--preflight --json`:

```bash
mere.run video generate "A wooden boat floats on calm water" \
  --model video-ltx23-av-mlx --num-frames 25 \
  --output boat.mp4 --preflight --json
```

Preflight and generation share option validation and model-dependent defaults
for LTX, Wan, and MiniMax-H3. Preflight reads installed model files and adapter
headers without loading tensors, downloading assets, or creating output
directories. Generation resolves the model and checks the inputs again before
loading it.

For HDR IC-LoRA, preflight preserves the requested canvas and reads the
adapter's reference-scale metadata. For H3, it applies the native adapter step
recipe and conditioning rules. Missing files remain blockers; a predicted
configuration doesn't mean the model is installed or ready to run.

Automatic LTX duration prediction leaves the resolved frame count unset until
the duration model runs. For source-audio A2Vid, the previewed frame count is
the configured limit; the available audio segment can shorten the result.
Invalid nonfinite numeric arguments appear as `"Infinity"`, `"-Infinity"`, or
`"NaN"` strings in a blocked JSON response so the diagnostic remains readable.

## Generate a fast draft

The default `--quality draft` lane is the speed path. It writes video-only MP4s
by default and is the right first pass for prompt, camera, subject, and
composition checks. For the split-layout LTX 2.3 model, it retains the joint AV
denoising tokens that influence video through audio-to-video cross attention,
but skips loading and decoding the audio VAE and vocoder and writes no audio
stream.

```bash
mere.run video generate \
  "a cinematic drone flythrough over snowy mountains" \
  --num-frames 65 \
  --output ./clip.mp4
```

LTX snaps width and height down to multiples of 64 and adjusts frame counts to
`8n+1`.

## Anchor start and end frames

Use `--image` to anchor the first latent frame. Add `--end-image` when the clip
should move toward a specific final keyframe.

```bash
mere.run video generate \
  "a car drives from a bright morning street into a warm sunset road, smooth forward motion" \
  --image ./car-start.png \
  --image-strength 0.9 \
  --end-image ./car-end.png \
  --end-image-strength 0.85 \
  --num-frames 65 \
  --output ./car-start-to-end.mp4
```

`--end-image` requires `--image`. If a short clip makes the start and end
latent conditioning windows overlap, the start conditioning wins. LTX 2.5 also
accepts repeatable `--image-conditioning PIXEL_FRAME:PATH[:STRENGTH[:CRF]]`
guides at any frame; see [Use LTX 2.5 pipelines](#use-ltx-2-5-pipelines).

## Render final quality

Quality and output are independent choices. `--quality final` uses the full
dev plus distilled-LoRA two-stage pipeline and still writes a video-only MP4 by
default:

```bash
mere.run model pull video-ltx23-full-mlx --accept-model-license
mere.run video generate \
  "a red fox runs across a snowy clearing, detailed winter fur, natural motion" \
  --quality final \
  --duration 4 \
  --output ./fox-final.mp4
```

Omitting audio from the deliverable isn't a separate speed optimization. The
meaningful speed change comes from choosing the draft checkpoint.

## Generate synchronized audio and video

For LTX 2.3 audio and video, pull the managed model ID. The pull also installs
its Gemma 3 text-encoder companion.

```bash
mere.run model pull video-ltx23-full-mlx --accept-model-license
mere.run video generate \
  "dialogue with clean background music" \
  --quality final \
  --output-mode audio-video \
  --duration 15 \
  --fps 24 \
  --output ./ltx23.mp4
```

Use `--duration` instead of pairing `--num-frames` and `--fps` manually for
representative unified AV tests. LTX 2.3 expects 24 frames per second; for
example, 15 seconds resolves to 361 frames at 24 frames per second because LTX
frame counts must satisfy `8n+1`.

`--quality final --output-mode audio-video` runs the official two-stage quality
path: guided dev denoising for both audio and video at half resolution, followed
by x2 latent upscaling and four-step refinement after fusing the distilled LoRA.

Add `--timings` to print phase timings for native LTX 2.3 draft, final, or A2Vid
generation. `--timings-output PATH` writes the same typed report as JSON,
including model-component loading, text encoding, each denoising stage, LoRA
fusion where applicable, upsampling, video and audio decode, MP4 writing,
unload, and total wall time.

## Generate video from source audio

Supplying `--audio` selects A2Vid automatically; no LTX variant flag is needed.
The requested segment directly conditions video motion, and that same decoded
source segment becomes the output soundtrack.

```bash
mere.run model pull video-ltx23-full-mlx --accept-model-license
mere.run video generate \
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
runtime fails rather than padding or silently falling back to text-to-video
plus a laid-back soundtrack. Mono input is duplicated to stereo without
auto-gain. `--audio-max-duration` caps the decoded source window
independently.

Stage one runs 30 guided full/dev steps by default with frozen audio latents.
Stage two upsamples the video, fuses the official distilled LoRA one tensor pair
at a time, and runs the four upstream distilled sigmas. `--a2v-guidance-scale`,
`--video-cfg-guidance-scale`, `--audio-cfg-guidance-scale`,
`--v2a-guidance-scale`, `--a2v-steps`, and `--negative-prompt` expose the
advanced controls. The `--a2v-guidance-scale` value is also the video stream's
audio-to-video modality guidance in full unified AV.

`video-ltx23-av-mlx` and `video-ltx25-distilled-bf16` reject `--audio`; use
`video-ltx23-full-mlx` or `video-ltx25-full-bf16`.

## Use LTX 2.5 pipelines

The two LTX 2.5 checkpoints implement the official LTX 2.5 inference family
directly in Swift and MLX:

- `video-ltx25-distilled-bf16` is a public, self-contained distribution of
  about 53.9 GB. The transformer, VAEs, upsampler, and duration head stay BF16;
  the bundled Gemma 4 language tower is MLX Q4. Pull it once with
  `--accept-model-license`; it needs no Hugging Face gate, companion download,
  or local conversion.
- `video-ltx25-full-bf16` is the complete gated BF16 bundle of about 119.7 GB.
  It adds the dev transformer, distilled LoRA, diffusion VAE, temporal
  upsampler, and DurationHead for the full, HQ, DFR, A2Vid, Retake, HDR/EXR,
  Dub-It, and text-to-audio workflows.

Gemma and the image VAE encoder are released after their conditioning tensors
are evaluated, before denoising begins.

```bash
mere.run model pull video-ltx25-full-bf16 --accept-model-license
mere.run video generate \
  "a harbor at blue hour, gulls and rigging audible in a soft sea breeze" \
  --model video-ltx25-full-bf16 \
  --output-mode audio-video \
  --output ./harbor.mp4
```

When you omit `--width` and `--height`, LTX 2.5 keeps the selected recipe's
native canvas. When you omit `--num-frames`, DurationHead predicts a duration
in the official 1 to 20 second range; `--auto-duration MIN MAX` narrows it.
`--fps` accepts fractional rates such as `23.976`.

The following options select the other official pipelines:

| Pipeline | Options |
| --- | --- |
| HQ two-stage with Res2s | `--ltx-preset hq` |
| Dev one-stage | `--ltx-pipeline dev-one-stage` |
| Keyframe interpolation | `--ltx-pipeline keyframe-interpolation` with repeated `--image-conditioning` |
| Generated keyframes | `--num-generated-keyframes` or repeated `--generated-keyframe` |
| IC-LoRA reference video | Repeated `--lora` and `--video-conditioning` |
| Diffusion Fidelity Rendering (DFR) | `--dfr`, optional `--detailing-lora` and `--temporal-upsample-rounds` |
| HDR and EXR output | `--hdr`, `--hdr-transfer`, and the other HDR options |

DFR's spatial detailer is a separately gated adapter:

```bash
mere.run adapter pull ltx25-pixel-spatial-upscaler-x2 --accept-license
mere.run video generate \
  "an intricate clockwork observatory turning beneath the stars" \
  --model video-ltx25-full-bf16 \
  --dfr \
  --detailing-lora ltx25-pixel-spatial-upscaler-x2 \
  --temporal-upsample-rounds 1 \
  --output ./observatory.mp4
```

Full-model acceleration is opt-in. `--ltx-transformer-execution compiled`
reuses one compiled MLX graph across the transformer blocks and can change
floating-point results slightly. `--ltx-guidance-projection-cache` reuses
positive-prompt projections across guidance passes. `--ltx-teacache` enables
calibrated block-residual reuse for the full two-stage Euler and HQ Res2s
paths; it's deterministic for a fixed request, but cached and uncached videos
aren't pixel-identical.

For the pinned upstream release, the complete pipeline matrix, and the native
execution boundaries, see [LTX 2.5 upstream parity](../ltx25-upstream-parity.md).

## Retake a region

`video retake` regenerates the half-open interval from `--start-time` to
`--end-time` and keeps the rest of the source timeline. The source can be an
MP4 or MOV file, or a directory of scene-linear EXR frames with
`--frame-rate`.

```bash
mere.run video retake "the guitarist turns toward the camera and smiles" \
  --source ./performance.mp4 \
  --start-time 2.0 \
  --end-time 4.5 \
  --preserve-audio \
  --output ./performance-retake.mp4
```

`--preserve-audio` keeps the source audio and retakes only the video;
`--preserve-video` does the reverse. The default model is
`video-ltx25-distilled-bf16` with seed 10. With `video-ltx25-full-bf16`,
`--steps` (default 30), `--negative-prompt`, and the video and audio guidance
options take effect.

## Transfer identity with Dub-It

`video dub-it` runs the official two-stage distilled Dub-It recipe. It takes a
reference container with both video and audio tracks and exactly one Dub-It
IC-LoRA safetensors file. The frame count and frame rate come from the
reference video, including fractional rates.

```bash
mere.run video dub-it "The speaker performs on a rain-lit street" \
  --reference-video ./speaker.mp4 \
  --ic-lora ./dub-it.safetensors \
  --output ./speaker-street.mp4
```

The output defaults to 1536 x 1024; `--width` and `--height` must be divisible
by 64. The reference video is appended as IC-LoRA conditioning at both stages,
and its audio is appended as clean negative-time reference tokens.

## Keep LTX resident

`video session` amortizes checkpoint loading across serial synchronized-AV
generations on LTX 2.3 or LTX 2.5, using either the standalone distilled model
or the full dev model. It reads typed snake-case JSONL requests from stdin and
writes one typed JSON result or error to stdout for each input line. Required
request keys are `prompt` and `output`.

```bash
printf '%s\n' \
  '{"id":"draft-1","prompt":"a fox runs across snow","output":"./draft-1.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  '{"id":"draft-2","prompt":"a fox runs across snow","output":"./draft-2.mp4","width":512,"height":320,"num_frames":33,"fps":24,"seed":7}' \
  | mere.run video session --model video-ltx23-av-mlx
```

Use `--model video-ltx23-full-mlx` for the two-stage quality lane. That session
keeps the dev transformer and official distilled LoRA resident, leaves the dev
weights unchanged for Stage 1, and activates the adapter only during Stage 2.
This avoids permanent BF16 weight fusion and permits repeat requests without
checkpoint reload or accumulated weight drift.

Full-model requests also accept `guidance_projection_cache`, `tea_cache`,
`tea_cache_threshold`, `tea_cache_calibration_output`, `sampler`, and the
other keys that `mere.run video session --help` lists. The session-level
`--prompt-cache-capacity` (default 8) controls exact reuse of materialized
Gemma connector outputs; `0` turns it off. TeaCache is limited to the full
two-stage Euler and Res2s paths and stays off by default. Its corrected BF16
calibration uses the maximum drift across each synchronized four-branch
guidance group, with conservative default thresholds of `0.235` for Euler and
`0.39` for Res2s. Timings report computed and reused 48-block stacks. Keep
TeaCache off when exact output reproduction matters.

The first result includes checkpoint-load time. Later results set
`resident_model_reused` to `true` and report zero load time. Full-model
requests don't keep the text encoder resident during denoising, so the large
transformer, video decoder, and text encoder aren't all live at once; a later
request reloads the encoder only when its prompt isn't cached.

## Animate a start frame with Wan 2.2

`video-wan22-ti2v-5b-mlx` is a native Wan 2.2 TI2V image-to-video lane. It
requires `--image`, rejects `--end-image`, snaps width and height to a 32-pixel
grid and frame counts to `4n+1`, and takes `--steps` (default 40),
`--guidance-scale` (default 5), and `--shift` (default 5).

```bash
mere.run model pull video-wan22-ti2v-5b-mlx
mere.run video generate "the camera walks forward" \
  --model video-wan22-ti2v-5b-mlx \
  --image ./frame.png \
  --num-frames 41 --width 1280 --height 704 \
  --output ./walk.mp4
```

## Generate with MiniMax-H3

MiniMax-H3 FL2VA generates 24 fps RGB video and synchronized 32 kHz stereo
audio from text, a first frame, or directed first and last frames. Ref2VA
conditions on ordered references instead. Frame counts follow `17n+5`, and
width and height are multiples of 32. The model is CFG-distilled, so `--steps`
is a schedule-point count without a second unconditional model evaluation.

::: warning License territory
The MiniMax-H3 Community License excludes use, distribution, and display in the
United States, European Union, United Kingdom, and Republic of Korea. Read and
accept the model terms before pulling, converting, using, or redistributing any
artifact. Passing `--accept-model-license` and continuing with the download
confirms that you accept those terms and agree to comply with them.
:::

The managed FL2VA packages differ in precision:

- `video-minimax-h3-fl2va-bf16-mlx` is the maximum-fidelity compact package.
  Its official-source BF16 denoising core omits the schedule-only AdaLN,
  timestep-MLP, and reconstructed RoPE tensors and uses a source-bound pack of
  exact production schedules. The Q8 conditioner, FP16 video VAE, and FP32
  audio VAE are stored in the same single-root immutable artifact.
- `video-minimax-h3-fl2va-8bit-mlx` is the smaller high-quality fallback. It
  uses MLX affine INT8/group-64 for eligible core linears and retains the
  same conditioner, VAEs, cache pack, and provenance as compact BF16. Q8 is a
  storage and memory option, not an advertised speedup without measurements.

```bash
mere.run model pull video-minimax-h3-fl2va-bf16-mlx --accept-model-license
mere.run video generate "a cinematic rain-soaked bus stop at night" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --width 1280 --height 768 --duration 10 --steps 21 \
  --output ./bus-stop-bf16.mp4

mere.run model pull video-minimax-h3-fl2va-8bit-mlx --accept-model-license
mere.run video generate "a glass marble rolls across wood with a delicate rattle" \
  --model video-minimax-h3-fl2va-8bit-mlx \
  --image ./marble.png \
  --num-frames 124 \
  --output ./marble-h3.mp4
```

### Use Turbo and LightX2V adapters

The `minimax-h3-turbo-4step` EMA-850 adapter and all five LightX2V releases are
checksum-pinned separately:

```bash
mere.run adapter pull minimax-h3-turbo-4step
mere.run adapter pull minimax-h3-lightx2v-4step
mere.run adapter pull minimax-h3-lightx2v-8step-v1
mere.run adapter pull minimax-h3-lightx2v-4step-v1-768p
mere.run adapter pull minimax-h3-lightx2v-8step-v1-768p
mere.run adapter pull minimax-h3-lightx2v-ref2v-4step-v0.1
```

Pass one with `--h3-adapter` and omit `--steps` to select its pinned recipe:

```bash
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
```

The adapter recipes differ as follows:

| Adapter | Schedule points (evaluations) | Video/audio shifts | Alpha |
| --- | --- | --- | --- |
| `minimax-h3-turbo-4step` and `minimax-h3-lightx2v-4step` | 5 (4) | From the pinned recipe | From the pinned recipe |
| `minimax-h3-lightx2v-8step-v1` | 9 (8); accepts the upstream four-evaluation fallback | 12/3 | 8 |
| `minimax-h3-lightx2v-4step-v1-768p` | 5 (4) | 6/3 | 128 |
| `minimax-h3-lightx2v-8step-v1-768p` | 9 (8); no four-evaluation fallback | 6/3 | 8 |
| `minimax-h3-lightx2v-ref2v-4step-v0.1` | 5 (4) | 12/3 | 8 |

Both 768p releases target a 1344 x 768 canvas. EMA-850 and the four LightX2V
FL2VA adapters support compact BF16 and Q8, reject legacy Q4, and can't be
combined with Ref2VA references. The Ref2VA adapter requires ordered
references; see [Condition on references with Ref2VA](#condition-on-references-with-ref2va).
Strength `1.0` applies an adapter's released weights exactly;
`--h3-adapter-strength` remains available for prompt-specific tuning. No H3
adapter can use denoise-step cache reuse.

EMA-850 remains an activation-space adapter because its 51 AdaLN deltas
participate in schedule-cache construction. For compact models those deltas are
computed from `silu(cached time embedding)` and added to an in-memory augmented
table keyed by source, schedule, adapter hash, and strength. LightX2V standard
and QKV projections also run as activation-space low-rank wrappers around dense
or stock MLX quantized linears; base weights are never expanded or fused.

The pinned compact BF16 and Q8 model roots contain the exact nine-point
shifts-6/3 AdaLN table used by the eight-step 768p adapter. The refreshed table
closes against the pinned official source on MLX Metal. A tensor comparison
against the former interpolation path covered 116,444,160 values: 88.40% were
already exact, and the remaining values reached 0.125 maximum absolute error
and 0.001427 RMSE. This result shows why the exact table is preferable; it
doesn't by itself constitute a same-seed visual or audio quality qualification.

### Use the FastH3 package

`video-minimax-h3-fasth3-vsa-datafree-mlx` is a self-contained FastVideo
student package. One explicit pull installs the premerged Q8 student, Q8 text
encoder, both VAEs, tokenizer, Q8 compression gates, and source-bound AdaLN
table; generation needs no other model or adapter download.

```bash
mere.run model pull video-minimax-h3-fasth3-vsa-datafree-mlx \
  --accept-model-license
mere.run video generate "a lighthouse in a winter storm" \
  --model video-minimax-h3-fasth3-vsa-datafree-mlx \
  --output ./lighthouse-fasth3.mp4
```

The recipe accepts text-only generation and requires adapter strength `1.0`.
It rejects frame conditioning, continuation, references, and additional H3
approximation modes. Its four denoising evaluations use base sigma points
`0.999`, `0.749`, `0.5`, and `0.25`, followed by the clean endpoint. Only the
exact FastH3 ID runs its embedded adapter; another spelling of its upstream
repository runs as FL2VA. FastH3 VSA is unsupported on Linux because its sparse
attention requires Metal kernels.

MLX Metal runs FastH3's 64-token VSA tiles, per-head top-k routes at 90%
video-key sparsity, and learned pooled-value compression gates. The Metal
kernel consumes compact selected-video route tables, so video queries don't
scan rejected tiles. Prefix queries remain dense, and video queries retain
every prefix key tile.

On affine Q8 FastH3 roots, Metal selects the `fasth3-metal` exact-kernel mode.
This mode fuses attention AdaLN, post-attention gate and AdaLN, Q/K
normalization and rotary embedding, FC1 and SwiGLU, and FC2. Float32 activation
streams use Float32 SIMD-group matrix operands and accumulators. The mode runs
inside FastH3's release-mode compiled block regions.

The fused FC1 writes only the compact SwiGLU result. It doesn't create the
9.53 GiB Float32 `[1, 89188, 28672]` projection for a 1,344 by 768 pixel,
294-frame request. The remaining compact result is 4.76 GiB. At shapes where
that result exceeds 4 GiB, the compiled runner evaluates FC1 and SwiGLU before
FC2 to preserve 64-bit indexing. Each kernel validates its complete shape,
dtype, and affine Q8/group-64 contract before dispatch. Set
`MERERUN_H3_EXACT_KERNELS=disabled` only when comparing against the portable
MLX path. To select the combined mode explicitly, set
`MERERUN_H3_EXACT_KERNELS=fasth3-metal`.

The package build uses
`scripts/model-conversion/merge_minimax_h3_fasth3_q8.py` after creating the
source-bound table with `prepare_minimax_h3_fasth3_vsa.py`. The merge script
verifies all 362 LoRA pairs, 82 direct differences, and 50 compression gates.
It merges 208 inference linear targets with FP32 accumulation, rounds once to
BF16, and writes MLX affine Q8/group-64 weights. If you pull the managed model,
you don't run either build step.

### Generate long shots and timed keyframes

For long FL2VA or Ref2VA shots, `--h3-window-frames` keeps the H3 runtime
resident and conditions each new window on overlapping motion plus its matching
generated soundtrack. `--h3-frame FRAME:PATH` injects an FL2VA keyframe at an
exact zero-based output frame.

```bash
mere.run video generate "one continuous tracking shot through a night market" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --duration 15 \
  --h3-window-frames 124 \
  --h3-window-overlap 35 \
  --h3-acceleration maximum \
  --output ./market-long.mp4

mere.run video generate "the actor crosses three connected sets" \
  --model video-minimax-h3-fl2va-bf16-mlx \
  --num-frames 175 \
  --h3-frame 72:./second-set.png \
  --h3-frame 144:./third-set.png \
  --output ./three-sets.mp4
```

You can repeat `--h3-frame` up to the released 12-frame condition limit; the
frames stay on the global timeline when sliding windows are on. The window
count must use H3's `17n+5` target geometry. The overlap, set by
`--h3-window-overlap` (default 18), must use `17n+1` and leave at least 22
frames for each new target.

Each continuation encodes all overlap frames except the boundary as motion
history, uses the final overlap frame as the next first-frame condition, and
carries the corresponding 32 kHz stereo waveform as history and boundary audio
latents. Only new frames and samples are appended, so the final MP4 has the
exact aligned global duration. The conditioner, transformer, AdaLN table,
reference encodings, and both VAEs remain resident across windows.

### Choose weight residency and acceleration

The compact BF16, Q8, legacy Q4, and Ref2VA managed packages include
source-bound inference-only AdaLN caches, so no post-pull `model optimize` step
is required. Compact BF16 and Q8 packages select exact tables for 5, 9, 12, 16,
21, or 31 points at shifts 12/3 and the LightX2V 768p 5- and 9-point schedules
at shifts 6/3. A custom schedule interpolates from the densest table and emits
a visible non-bit-exact diagnostic. Generation skips the 13B-parameter
AdaLN/time-embedding branch. `model optimize` remains available for compatible
locally converted roots that still contain the full AdaLN branch.

By default H3 uses 9 schedule points through 13,500 packed rows, 16 through
26,000, and 21 above that. Maximum acceleration caps the automatic schedule at
12 points; `--steps` remains an explicit override.

MacBooks with less than 96 GiB keep quantized weights resident by default.
Memory-qualified desktops and MacBooks with 96 GiB or more select the faster
resident BF16 path when the requested geometry leaves the required runtime
reserve. `--h3-weight-mode quantized` or `resident-bf16` forces either path.

`--h3-acceleration` selects the denoise policy:

- `quality` (default) uses dense attention, executes all 50 blocks, and is
  exact.
- `balanced` and `maximum` dynamically route distant target-video attention
  blocks on Apple GPUs at 12,000 or more packed rows, and still execute all
  50 blocks on every model evaluation. Prefix queries, all prefix keys,
  neighboring video blocks, the first two transformer layers, the leading
  schedule region, and the final evaluation remain dense. Skipped blocks retain
  a centroid-and-summed-value correction in the online-softmax accumulator, and
  a once-per-shape dense-route gate must pass before sparse execution is
  admitted.
- `layers-45`, `layers-40`, `velocity-reuse-2`, and `token-reduction` are
  experimental research arms, isolated from the other approximation policies
  and never selected automatically.

Without an H3 adapter, `balanced` and `maximum` also use the modality-aware
adaptive first-block cache. Every evaluation executes block 1 and measures
global plus worst-time-slice drift independently for video and audio. A cache
hit reuses only the target residual from blocks 2 through 50. `balanced`
refreshes after at most two hits and reserves the final two evaluations;
`maximum` refreshes after at most four hits and reserves the final evaluation.
Both require two complete evaluations before reuse. Dynamic attention and cache
reuse are approximate and can change motion or composition for the same prompt
and seed; use `quality` when exact-seed fidelity matters.

The research arms work as follows:

- `velocity-reuse-2` keeps the quality schedule, protects the first and final
  full evaluations, and linearly extrapolates the complete synchronized
  video/audio velocity from the two most recent full evaluations on
  intervening odd steps. Video and audio use their independent shifted
  schedules, and the extrapolation ratio is clamped to `[-2, 2]`.
- `layers-45` and `layers-40` rank mean absolute attention and MLP gates from
  the exact AdaLN table, protect blocks 0, 1, and 49, and skip the lowest
  remaining blocks. They reduce executed block work but keep all weights
  loaded, so no residency reduction is claimed.
- `token-reduction` preserves the complete packed prefix and target audio while
  pairing adjacent horizontal target-video tokens after block 3. It uses
  averaged spatial RoPE coordinates on the reduced grid. During the first ten
  denoise evaluations the reduced path continues through block 39; later
  evaluations restore before block 30. Each full-resolution video token is
  reconstructed from its saved value plus the corresponding reduced token's
  update from the pooled baseline, preserving within-pair detail.

### Condition on references with Ref2VA

`video-minimax-h3-ref2va-mlx` is an explicit-pull 8-bit package. Repeated
`--reference image:PATH`, `video:PATH`, or `audio:PATH` options keep their
request order. Video soundtracks are conditioned with their video; a standalone
audio reference must be paired with an image or video. The transformer and Qwen
conditioner use MLX affine INT8/group-64, and 8-bit is the published Ref2VA
quality floor. Its source-bound AdaLN cache is bundled, so no post-pull
`model optimize` step is required.

```bash
mere.run model pull video-minimax-h3-ref2va-mlx --accept-model-license
mere.run video generate "preserve the person and use the reference motion" \
  --model video-minimax-h3-ref2va-mlx \
  --reference image:./person.png \
  --reference video:./motion.mp4 \
  --h3-adapter minimax-h3-lightx2v-ref2v-4step-v0.1 \
  --h3-weight-mode resident-bf16 \
  --output ./ref2va.mp4
```

mere.run expands the managed INT8 Ref2VA transformer to resident BF16 before
fusing the `minimax-h3-lightx2v-ref2v-4step-v0.1` adapter, so pass
`--h3-weight-mode resident-bf16` unless automatic admission is known to
qualify. Forced quantized execution with that adapter is rejected.

Ref2VA reference images preserve source aspect ratio and are downscaled only
when their area exceeds the internal render canvas, with both dimensions
rounded to 32-pixel multiples. Standalone reference audio keeps its complete
2 to 15 second duration rather than being truncated to target-video duration;
ordered reference audio remains capped at 15 seconds in total. H3 condition
augmentation and target video and audio latents use the released independent
seeded streams and native latent layouts.

### Render on a smaller internal canvas

`--h3-render-width` and `--h3-render-height` select an explicit internal target
canvas for FL2VA or Ref2VA. Set both; they must be same-aspect 32-pixel
multiples no larger than the requested output. DiT and VAE decode operate on
that smaller grid. The decoded RGB frames are then returned to `--width` and
`--height` with the pinned h3.c high-quality vImage scaling contract. A
512 x 512 output can use 384 x 384 for the 75% arm or 320 x 320 for the 62.5%
arm. Reduced rendering is rejected with sliding windows because continuation
conditioning hasn't been resampled and qualified.

### MiniMax-H3 receipts

The following dated receipts record the measured H3 runs and their limits:

- [MiniMax-H3 BF16 on M4 Max](../benchmarks/minimax-h3-bf16-m4-max.md)
- [MiniMax-H3 compact BF16 and Q8 on M4 Max](../benchmarks/minimax-h3-compact-bf16-q8-m4-max.md)
- [FastH3 VSA DataFree Metal qualification](../benchmarks/minimax-h3-fasth3-vsa-datafree-m4-max.md)
- [MiniMax-H3 Ref2VA MLX 8-bit validation](../benchmarks/minimax-h3-ref2va-mlx-8bit.md)
- [MiniMax-H3 h3.c transfer program](../benchmarks/minimax-h3-h3c-transfer.md)

## Run Cosmos3 modes

`video-cosmos3-edge-mlx` is NVIDIA's pinned Cosmos3-Edge snapshot. One
checkpoint covers image and video generation and editing, multimodal reasoning,
policy, forward and inverse dynamics, and camera-controlled world transitions.
Its weights are governed by OpenMDW-1.1.

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

`--mode` accepts `text-to-image`, `image-to-image`, `text-to-video` (the
default), `image-to-video`, `video-to-video`, `policy`, `forward-dynamics`,
`inverse-dynamics`, and `reasoner`. Width and height must be divisible by 16,
and video frame counts must be `4n+1`. Run `mere.run guide video cosmos3` for
the full mode matrix and action-domain contract. The same runtime powers
`world serve --backend cosmos3`, which keeps the transformer, VAE, and terminal
latent resident across transitions.

The native runtime also accepts a local MLX Q4 Cosmos3-Super Text2Image 4-Step
root for `--mode text-to-image`. It reads the fixed four-step stochastic
schedule from the model root and disables classifier-free guidance.

## Animate or replace a subject with SCAIL-2

`video-scail2-14b-mlx` is a separately packaged, MIT-licensed SCAIL-2 14B MLX
bundle for reference- and mask-conditioned subject animation or replacement,
with clean history for long videos. Masks come from SAM 3.1.

### Install the models

Install the immutable SCAIL-2 snapshot and the separately licensed SAM 3.1
model:

```bash
mere.run model pull video-scail2-14b-mlx
mere.run model pull vision-segment-sam31 --accept-model-license
```

The default fast profile also uses the checksum-pinned Apache-2.0 LightX2V Wan
2.1 I2V adapter. Adapter weights stay in the local adapter store and aren't
bundled with mere.run:

```bash
mere.run adapter pull scail2-lightx2v-4step
```

### Prepare masks

Create a schema-version 1 mask plan, preview one target frame, and then prepare
the immutable full mask revision:

```bash
mere.run video prepare-masks \
  --plan ./mask-plan.json \
  --output-dir ./mask-preview \
  --preview-frame 12 \
  --json

mere.run video prepare-masks \
  --plan ./mask-plan.json \
  --output-dir ./approved-mask-candidate \
  --json
```

The full result contains an exact-geometry and FPS driving proxy, per-subject
prepared reference images and masks, a ProRes 4444 categorical mask video, an
overlay preview, a contact sheet, tracking and quality reports, and a canonical
SHA-256 manifest. The plan supports one to six stable subjects, text, box, and
point selectors, and dense painted PNG corrections. White is background; legal
subject colors are blue, red, green, magenta, cyan, and yellow.

Set the plan's `mode` to `replacement` to aspect-fit every reference into the
target canvas and matte all non-subject pixels to black; `animation` preserves
the fitted reference scene.

### Render the animation

After you review the mask artifacts, preflight and render natively:

```bash
mere.run video animate \
  "a dancer in a red silk dress" \
  --reference ./reference-performer-prepared.png \
  --reference-mask ./reference-performer-mask.png \
  --driving-video ./pose.mp4 \
  --driving-mask ./pose-mask.mp4 \
  --preflight --json

mere.run video animate \
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

Use `--mode replacement` to place the reference subject into the driving scene:

```bash
mere.run video animate \
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

Pair repeatable `--additional-reference` and `--additional-reference-mask`
values for multi-subject conditioning. The driving video and driving-mask video
must have identical decoded frame counts, and reference and mask ordering is
preserved.

The default `--profile fast` path selects the pinned `scail2-lightx2v-4step`
adapter, turns off classifier-free guidance, and uses shift 5, Euler updates,
and the adapter's exact `1000, 750, 500, 250` training-step schedule at
832 x 480. `--profile quality` is an explicit opt-in to the configurable
40-step UniPC and CFG recipe. Preflight resolves the managed adapter without
loading it, then verifies the installed file's exact byte count and SHA-256. A
missing or corrupt adapter is reported in `missing_input_files`; generation
can't continue with an unverified file.

The default long-video contract uses 81-frame segments with five decoded frames
of clean overlap. The compatibility defaults are `--tail-policy drop` and
`--audio-source none`. `pad-trim` keeps already legal `1 mod 4` temporal
lengths unchanged, pads only an incomplete final segment to the next legal
latent length, and trims the result to the exact requested frame count.
`--audio-source driving` muxes source audio to that exact duration. A legal
65-frame clip is never expanded to an unused 81-frame window.

### How SCAIL-2 runs

Prepared review artifacts always use white as the canonical background. During
inference, `video animate` matches the official SCAIL-2 mask-role semantics:
animation keeps the main reference background visible and hides the driving
background. Replacement hides the reference background and keeps the
driving scene visible. Additional subject-reference backgrounds are hidden.
Subject colors and reference-to-driving correspondence are unchanged. Decoded
mask pixels are snapped to the nearest legal color inside a strict tolerance,
and ambiguous or out-of-tolerance pixels are rejected.

The native transformer preserves upstream global self-attention while splitting
the query axis into independently evaluated Metal command buffers. Every query
slice still attends to the complete key/value sequence; the split only keeps a
full 81-frame window below the macOS GPU watchdog and doesn't introduce local
or sliding-window attention.

SCAIL-2's unified motion interface has a task-specific 20-channel input
projection, but the generic Wan 2.1 I2V adapter carries a 36-channel
projection difference. The native fuser requires every shared transformer
target to match exactly, applies the compatible 487 LoRA pairs and parameter
differences, and explicitly leaves only that incompatible input-projection
weight untouched. It doesn't use a general skip-mismatch mode.

## Export latents

```bash
mere.run video export-latents \
  "storm clouds over the ocean" \
  --output ./latents.safetensors
```

`video export-latents` targets the distilled video-only latent path and writes
final-stage latents as safetensors. It defaults to `video-ltx-av` at 768 x 512,
65 frames, and seed 42.

## Large decodes and MP4 writing

Long or high-resolution LTX decodes automatically split the VAE work into
overlapping temporal and spatial tiles. Tile pixels and trapezoidal weights
remain in MLX for device-side accumulation, normalization, and UInt8
conversion, instead of reading every float tile back to Swift. To validate a
specific memory envelope, override the automatic decode budget with
`MERERUN_VIDEO_LTX_VAE_DECODE_BUDGET_GB`.

Final MP4 encoding is incremental. The writer transfers one BGRA frame at a
time to FFmpeg stdin or AVAssetWriter and overlaps the next device-to-host frame
transfer with encoder work, avoiding a monolithic host frame buffer and a raw
video spool. Unified-AV audio is likewise transferred and written in aligned
chunks. The decoded MLX frame tensor still exists on device, so this removes
duplicated host-side staging rather than making the generation path
zero-memory. MP4 formats and backend selection are unchanged.

## macOS Studio

The optional macOS app compiles against the same typed capability contract
emitted by:

```bash
mere.run catalog video.generate --json
```

**Video ▸ Generate** exposes independent quality and output choices,
text-to-video, a start image, an end keyframe, source-audio A2V with segment
offset and modality guidance, duration or frame-count targeting, negative
prompts, Wan controls, preflight, and timing receipts. Attaching source audio
selects final quality plus synchronized audio-video output automatically.

**Video ▸ Subjects** is the SCAIL subject flow, laid out as a three-stage board
(Plan → Track → Animate). It authors one to six reference subjects with text,
box, and point selectors; previews palette-safe SAM 3.1 masks; tracks the
complete driving clip; accepts immutable keyframe corrections or painted binary
masks; and shows reference, driving, contact-sheet, and result playback
together. Animation and replacement semantics, trim range, dimensions, profile,
adapter, seed, segmentation and overlap, tail, audio, output, and preflight
controls map directly to the public CLI contract. Every mask and animation run
stays live and restartable in the Library.

Cosmos3 generation and action modes, raw mask-plan execution, LTX latent export,
and resident LTX sessions have no designed surface; they run from the **Command
Console**, which builds their form from the same capability contract.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VideoCommand.swift`
- `Sources/MereRunCLI/Commands/VideoAnimateCommand.swift`
- `Sources/MereRunCLI/Commands/VideoSessionCommand.swift`

### Runtime

- `Sources/MereRunCore/LTX/LTXUnifiedAVGenerator.swift`
- `Sources/MereRunCore/LTX/LTXUnifiedAVGenerator+Generation.swift`
- `Sources/MereRunLTXModel/`
- `Sources/MereRunCore/LTX/LTXInferenceTimings.swift`
- `Sources/MereRunCore/LTX/LTXGemmaTextEncoder.swift`
- `Sources/MereRunCore/LTX/LTXVideoMP4Writer.swift`
- `Sources/MereRunCore/SCAIL2/SCAIL2Generator.swift`
- `Sources/MereRunCore/SCAIL2/SCAIL2Transformer.swift`

## Source reading notes

The LTX runtime has more low-level model, media, and checkpoint-layout code
than most other runtime families in this repository. Start from the public
generation flow before reading the lower-level model definitions, in this
order:

1. Public generation types and `LTXUnifiedAVGenerator` state
2. Request normalization and generation flow
3. Denoise and latent-conditioning helpers
4. Decoding and media assembly code
5. Model definitions in `MereRunLTXModel`

If you're new to the repository, read the
[Architecture reading map](../architecture.md) first. For module ownership and
independent tests, see
[LTX runtime boundaries](../internals/ltx-runtime-boundaries.md).
