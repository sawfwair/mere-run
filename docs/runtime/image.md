# Image runtime

The image runtime generates pictures from text, edits existing pictures with
input images, references, or masks, and trains low-rank adaptation (LoRA)
adapters on your own image-caption sets. It can also replay a saved generation
plan and reconstruct a textured three-dimensional (3D) mesh from object
photos. Every model runs locally in native Swift and MLX on Apple Silicon.

To generate your first image with the default model, run:

```bash
mere.run model pull image-zimage-nano
mere.run image generate --prompt "A ceramic mug in soft morning light" \
  --output ./mug.png
```

The command prints the output path to stdout. Progress and diagnostics go to
stderr.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run image generate` | Generate or edit an image with a local image model. |
| `mere.run image train-lora` | Train a local image LoRA adapter. |
| `mere.run image visualize-run` | Open a loopback viewer for a LoRA training run. |
| `mere.run image dataset discover` | Find image-caption dataset candidates under a root folder. |
| `mere.run image run-plan` | Preflight, materialize, or run a saved image workflow plan. |
| `mere.run image validate` | Run deterministic engineering checks for the Z-Image and Klein families. |
| `mere.run image reconstruct-3d` | Reconstruct a colored object mesh from one image with native TripoSR. |
| `mere.run image reconstruct-3d-trellis2` | Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2. |
| `mere.run image reconstruct-3d-multiview` | Reconstruct a colored mesh from four or six supplied views with native InstantMesh. |

## Choose a model

`image generate` uses `image-zimage-nano` when you omit `--model`. The
following table lists every managed generation model with its defaults, its
download size, and the unified memory the capability catalog requires and
recommends. Pass `--steps` or `--cfg` to override a default.

| Model ID | Use it for | Default steps, CFG | Download | Memory (min / rec) |
| --- | --- | --- | --- | --- |
| `image-zimage-nano` | Fast general and photorealistic text-to-image (4-bit Z-Image Turbo) | 4, 1 | 20.5 GB | 12 / 16 GB |
| `image-zimage-max` | Higher-fidelity Z-Image Turbo | 4, 1 | 32.8 GB | 48 / 64 GB |
| `image-zimage-base` | Z-Image base layout for advanced workflows | 4, 1 | 5.9 GB | 48 / 64 GB |
| `image-klein-nano` | Fast FLUX.2 Klein 4B drafts (4-bit) | 4, 1 | 4.6 GB | 8 / 16 GB |
| `image-klein-max` | Higher-fidelity FLUX.2 Klein 4B | 4, 1 | 16.0 GB | 24 / 32 GB |
| `image-klein-9b` | Gated FLUX.2 Klein 9B generation, references, and Klein LoRAs | 4, 1 | 34.7 GB | 48 / 64 GB |
| `image-klein-base` | FLUX.2 Klein 4B base for training and compatibility | 50, 1 | 16.0 GB | 24 / 32 GB |
| `image-klein-base-9b` | Undistilled FLUX.2 Klein 9B base for LoRA training | 50, 1 | 51.5 GB | 64 / 96 GB |
| `image-bonsai-binary` | Smallest Klein deployment (1-bit) | 4, 1 | 3.4 GB | 16 / 24 GB |
| `image-bonsai-ternary` | Compact Klein deployment (2-bit) | 4, 1 | 3.9 GB | 12 / 16 GB |
| `image-flux3-klein` | Experimental FLUX 3 Klein generation and up to 10 references | 16, 1 | 20.3 GB | 32 / 48 GB |
| `image-flux1-dev` | Gated FLUX.1-dev with FLUX.1 LoRAs | 28, 3.5 | 34.0 GB | 48 / 64 GB |
| `image-flux2-dev` | Gated FLUX.2-dev with Dev LoRAs | 50, 4.0 | 78.1 GB | 96 / 128 GB |
| `image-qwen-21` | Qwen Image 2.1 generation, 10-reference editing, and RGBA output | 40, 1 | 33.1 GB | 48 / 96 GB |
| `image-qwen-21-turbo` | Eight-step Qwen Image 2.1 Turbo | 8, 1 | 32.5 GB | 48 / 96 GB |
| `image-qwen-edit-2511` | One-to-three-image editing, quality recipe | 40, 4.0 | 57.7 GB | 96 / 128 GB |
| `image-qwen-edit-2511-lightning` | One-to-three-image editing, four-step Lightning recipe | 4, 1 | 58.6 GB | 96 / 128 GB |
| `image-hidream-o1-dev` | Distilled HiDream O1 text, editing, and personalization | 28, 0 | 35.2 GB | 48 / 64 GB |
| `image-hidream-o1` | Full HiDream O1 | 50, 5.0 | 35.2 GB | 96 / 128 GB |
| `image-sensenova-u1-5-8b-mot` | Raw-pixel generation and multi-image editing | 50, 4 | 50.2 GB | 64 / 96 GB |
| `image-krea2-turbo` | Krea 2 eight-step text-to-image and Raw-trained LoRAs | 8, 0 | 38.7 GB | 96 / 128 GB |
| `image-krea2-raw` | Krea 2 base checkpoint for LoRA training | 52, 3.5 | 38.7 GB | 96 / 128 GB |
| `image-ideogram4-sdnq-uint4` | Ideogram 4 typography-focused text-to-image (SDNQ uint4) | 20, 7.0 | 17.2 GB | 48 / 64 GB |

The 3D reconstruction models are listed in
[Reconstruct a 3D mesh](#reconstruct-a-3d-mesh). Several models have
restrictive licenses; `model pull` shows the terms, and gated models require
`--accept-model-license` and a configured Hugging Face token. For per-model
install steps, read the offline handbook:

```bash
mere.run guide --model image-flux3-klein
mere.run guide --list-models
```

## Generate an image

`image generate` takes a prompt and writes a PNG. Output is 1024 x 1024 unless
you pass `--width` and `--height`. Pass `--seed` to make a run repeatable.

```bash
mere.run image generate \
  --model image-zimage-nano \
  --prompt "A ceramic mug in soft morning light" \
  --width 1024 --height 768 --seed 42 \
  --output ./mug.png
```

Use `--quiet` to print only the output path, `--progress-json` to stream
progress events to stderr as JSON lines, or `--receipt` to print a final JSON
line that lists every output path and kind.

### Preflight a request

To check the model, inputs, and output before generation starts, add
`--preflight --json`. Preflight reads local metadata and files only. It
doesn't prepare an edit canvas, load weights, or download models.

```bash
mere.run image generate \
  --model image-zimage-nano \
  --prompt "A ceramic mug in soft morning light" \
  --output ./mug.png \
  --preflight --json
```

Preflight and execution share model-dependent defaults, sigma schedules, LoRA
compatibility, and input validation. Execution rechecks input and adapter
availability before entering the runtime.

The report uses the shared structured-run envelope. It includes diagnostics,
declarative actions such as `start-generation`, `pull-model`, and
`open-output-directory`, and `result.run_plan`: a normalized `image.generate`
plan you can save and replay.

### Replay a saved plan

Save `result.run_plan` to a file, then preflight, materialize, or run it with
`image run-plan`:

```bash
mere.run image run-plan ./mug.plan.json --preflight --json
mere.run image run-plan ./mug.plan.json --materialize ./runs/mug --json
mere.run image run-plan ./mug.plan.json
```

A materialized generation directory contains `plan.json`, `actions.json`,
`run.json`, an initial events file, and artifact folders. The materialized
plan moves the output image and any structured-prompt sidecar into that
directory. Running it appends `run_started`, `run_finished`, or `run_failed`
events to the same stream. Hard blockers exit nonzero after the report prints.

### Expand a short prompt

`--structured-prompt` uses a local text chat model to expand `--prompt` into a
long structured JSON caption that covers objects, background, lighting,
aesthetics, camera characteristics, style, context, and rendered text. The
default expansion model is `text-chat-gemma4-12b-4bit`; change it with
`--structured-prompt-model`. With expansion on, the CLI raises the image prompt
token budget to 2,048 unless you already requested more. To keep the caption
for review, pass `--structured-prompt-output`:

```bash
mere.run image generate \
  --model image-ideogram4-sdnq-uint4 \
  --prompt "A poster for a night market" \
  --structured-prompt \
  --structured-prompt-output ./night-market.caption.json \
  --output ./night-market.png
```

## Edit an image

`image generate` has three ways to condition on existing pictures. Which ones
apply depends on the model.

| Option | What it does | Models |
| --- | --- | --- |
| `--input` | Image-to-image source. `--strength` (default 0.75) sets how much changes. | Most families; several treat it as the first reference |
| `--ref-image` | Ordered reference images. Repeat for more. | Klein, FLUX 3 Klein, Qwen Image 2.1, Qwen Edit, HiDream O1, SenseNova U1.5 |
| `--mask`, `--outpaint` | Restrict changes to a region of `--input`, or extend its canvas. | Any model that accepts `--input` |

Krea 2 and Ideogram 4 are text-to-image only.

### Transform an input image

```bash
mere.run image generate \
  --prompt "Turn this into a pencil sketch" \
  --input ./photo.png \
  --strength 0.6 \
  --output ./sketch.png
```

### Edit inside a mask or extend the canvas

`--mask` takes a black-and-white image. White pixels can change; black pixels
are restored from `--input` after generation. `--outpaint` adds editable
padding as `top,right,bottom,left` pixels inside the requested `--width` and
`--height`, and places the source in the remaining region. `--mask-feather`
sets the blend radius at mask and outpaint edges (default 8 pixels). Both
options require `--input`.

```bash
mere.run image generate \
  --input ./room.png --mask ./window-mask.png \
  --prompt "A tall window with morning light" \
  --output ./room-window.png

mere.run image generate \
  --input ./room.png --outpaint 0,256,0,256 \
  --width 1536 --height 1024 \
  --prompt "A wide view of the same room" \
  --output ./room-wide.png
```

The runtime prepares an image-to-image canvas and then restores every unmasked
pixel, so the same mask and outpaint contract works across backends that only
expose whole-image conditioning.

## Record and retry an image run

To keep the settings, inputs, and result of an image operation, pass a new
`--run-dir` path:

```bash
mere.run image generate --model image-zimage-nano \
  --prompt "A brass camera on a wooden desk" \
  --output ./camera.png --run-dir ./runs/camera
mere.run run inspect ./runs/camera --json
mere.run run list --root ./runs --json
mere.run run retry ./runs/camera --json
```

The directory must not exist. If you place the output inside it, name the file
`output.png`. Write structured-prompt sidecars outside the directory.
Recording is optional, and preflight doesn't create a record. Output paths,
receipts, progress streams, and run-plan files keep their formats. A saved
preflight plan retains an explicit `--run-dir`.

`image-run.json` uses schema version 1. It separates your requested settings
from resolved settings, including model defaults, expanded prompts, adapter
paths, and the seed chosen before inference. It stores the model manifest,
source revisions when available, the backend, and SHA-256 fingerprints of
retained inputs and outputs. Missing source provenance stays unknown, and
manifest fingerprints don't certify the contents of every weight file.

The states are `preparing`, `running`, `succeeded`, `failed`, `cancelled`, and
`interrupted`. Inspection marks an unfinished record as interrupted when its
process no longer holds the run lock, so a forced stop or reboot is
distinguishable from an observed cancellation. Argument parsing and static
option errors occur before a run is created.

Retry starts generation in a new sibling directory from the recorded resolved
settings and seed. It prints that directory, or the new record with `--json`,
keeps the parent ID, and preserves the original record and output. Before it
starts, retry checks the installed model manifest, copied image inputs, and
local adapter fingerprints. If any of these changed, or the original run
stopped before resolving its settings, run the original command instead. Retry
doesn't resume denoising or expand a prompt again, and identical pixels across
runtime versions or hardware aren't guaranteed.

Run directories contain prompts and copies of input images, masks, and
outputs. They're created with access restricted to your account and kept until
you remove them. Keep the directory where it is: schema version 1 stores
absolute paths. Adapter and model files stay in their installed locations.

## Model families

Each family runs through the same `image generate` command. The sections that
follow cover what differs: install steps, defaults, accepted inputs, and
known limits.

### FLUX 3 Klein

`image-flux3-klein` runs the distilled FLUX 3 Image Klein model and loads the
original BF16 transformer, Qwen3-VL-4B text encoder, and autoencoder directly.
It's experimental. Review the FLUX Kommunity License, request access to the
gated Hugging Face repository, and then install:

```bash
mere.run model pull image-flux3-klein --accept-model-license
mere.run image generate --model image-flux3-klein \
  --prompt "Put the camera from the first image on the desk in the second" \
  --ref-image ./camera.png --ref-image ./desk.png \
  --width 1024 --height 1024 --output ./composite.png
```

- Defaults are 16 steps and CFG 1. Sides must be divisible by 16, from 64 to
  4,096 pixels.
- Supply up to 10 ordered references; `--input` supplies the first one. The
  requested width and height set the output canvas, including for edits.
- Write bounding-box instructions in the prompt using the publisher's prompt
  format. There's no separate bounding-box option or prompt upsampler.
- Negative prompts, CFG other than 1, custom sigmas, `--strength`, LoRA, the
  base checkpoint, FP8, and NVFP4 are unsupported.

The [FLUX 3 Klein qualification report](../benchmarks/flux3-klein-native-qualification-2026-10-15.md)
covers 512 and 1024 generation and one- and two-reference edits on a 36 GiB
M3 Max, with output images and checksums. 4K output and ten-reference
workloads remain unqualified.

### Qwen Image 2.1 and Turbo

Both models load original BF16 safetensors without conversion and support up
to 10 ordered references and RGBA PNG output. To request transparency,
describe a transparent background in the prompt and save as PNG. Both carry
the Qwen Research License, which restricts use to non-commercial research and
evaluation.

```bash
mere.run model pull image-qwen-21 --accept-model-license
mere.run image generate --model image-qwen-21 \
  --prompt "A ceramic teapot on a wooden table" \
  --seed 42 --output ./teapot.png
```

- `image-qwen-21` defaults to 40 steps and guidance 1. Dimensions must be
  multiples of 32. `--cfg` greater than 1 with `--negative-prompt` enables
  classifier-free guidance. `--strength`, LoRA, and custom `--sigmas` are
  unsupported.
- `image-qwen-21-turbo` installs a separate checkpoint and defaults to eight
  steps and CFG 1. It reads the checkpoint's saved sampling grid and static
  scheduler, so it rejects other step counts and `--sigma-shift`. Running the
  base model with `--steps 8` isn't the same as Turbo.

The [Qwen Image 2.1 qualification report](../benchmarks/qwen-image-21-native-qualification-2026-09-20.md)
covers base-checkpoint generation, editing, transparency, and repeatability on
an M4 Max. Dense BF16 Turbo inference remains unqualified. For license terms
and checkpoint revisions, see
[Qwen Image 2.1 model sources](../model-sources.md#qwen-image-2-1).

#### Run the mixed-precision Turbo checkpoint

v0.63.0 and later also load local Qwen Image 2.1 checkpoints with affine
Q4/Q8 group-64 weights. The published
[Qwen Image 2.1 Turbo mixed-Q4/Q8 artifact](https://huggingface.co/Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit)
uses Q4 transformer blocks, Q8 encoder linear weights, and BF16 for the VAE,
embeddings, modulation, and input and output layers. It's separate from the
managed BF16 installs. Download the pinned revision and pass its folder to
`--model`:

```bash
hf download Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit \
  --revision d8276b786f94a4e8944189cc181e839e31759ab5 --local-dir ./qwen21-turbo-mixed
mere.run image generate --model ./qwen21-turbo-mixed \
  --prompt "A blue ceramic teapot" --width 1024 --height 1024 \
  --steps 8 --output ./teapot.png
```

The Qwen Research License still applies. Quantization reduces checkpoint
storage and weight residency; resolution and reference images still drive
activation memory. The
[mixed-checkpoint qualification report](../benchmarks/qwen-image-21-turbo-mixed-qualification-2026-10-09.md)
covers bounded 512 and 1024 generation, replay, text rendering, and
one-reference editing. It doesn't establish general image quality or a minimum
supported memory size.

### Qwen Image Edit 2511

The 2511 path accepts one to three ordered pictures. `--input` is Picture 1,
and each `--ref-image` appends the next picture. Every picture keeps its own
aspect ratio and conditioning grid; `--width` and `--height` set the edited
output independently.

```bash
mere.run model pull image-qwen-edit-2511
mere.run image generate \
  --model image-qwen-edit-2511 \
  --input ./scene.png \
  --ref-image ./material.png \
  --prompt "Replace the chair in Picture 1 with the material from Picture 2" \
  --output ./scene-edited.png
```

The quality model defaults to 40 steps and true CFG 4.0.
`image-qwen-edit-2511-lightning` adds a checksum-pinned BF16 adapter and
requires exactly four steps and CFG 1.0. Passing four steps to the quality
model doesn't activate Lightning weights.

```bash
mere.run model pull image-qwen-edit-2511-lightning
mere.run image generate \
  --model image-qwen-edit-2511-lightning \
  --input ./scene.png \
  --prompt "Change the chair to cobalt blue" \
  --output ./scene-fast.png
```

Hardware-specific memory, quality, and performance qualification is still
required before you treat either recipe as a released default.

### FLUX.1-dev

FLUX.1-dev runs through a dedicated Swift and MLX runtime. The managed package
pins the official Diffusers transformer, CLIP-L and T5-XXL text encoders,
tokenizers, VAE, and flow-matching scheduler. To install it, accept access on
Hugging Face, configure a Hugging Face token, and acknowledge the FLUX.1 dev
Non-Commercial License v1.1.1:

```bash
mere.run model pull image-flux1-dev --accept-model-license
```

To run adapters, repeat `--lora`. Each value takes an independent inline
scale as `PATH_OR_ID=SCALE`:

```bash
mere.run image generate \
  --model image-flux1-dev \
  --prompt "TRIGGER_TOKEN a glass sculpture in a quiet museum" \
  --lora ./flux1-fast.safetensors=1.0 \
  --lora ./flux1-style.safetensors=0.8 \
  --output ./flux1-fast-style.png
```

FLUX.1 adapters target a 3,072-wide transformer, FLUX.2-dev uses a 6,144-wide
transformer, and Klein uses a separate architecture. Every adapter in a stack
must match the loaded model; mere.run rejects mismatched shapes before
denoising.

The license limits the weights and derivatives to non-commercial,
non-production use and requires filters or manual review for generated
content. The local acceptance record doesn't certify that a workflow complies.

### FLUX.2-dev

FLUX.2-dev runs through the shared Swift and MLX FLUX.2 runtime. The managed
layout keeps the gated BFL transformer unchanged and uses a pinned 4-bit
Mistral Small 3.2 text encoder, so FLUX.2-dev transformer LoRAs stay
compatible. To install it, accept access on Hugging Face, configure a token,
and acknowledge the FLUX Non-Commercial License and BFL Acceptable Use Policy:

```bash
mere.run model pull image-flux2-dev --accept-model-license
```

Pass a local safetensors file to `--lora`, and repeat the option to stack
adapters in order. Values without an inline scale use `--lora-scale`
(default 1.0).

```bash
mere.run image generate \
  --model image-flux2-dev \
  --prompt "TRIGGER_TOKEN a glass sculpture in a quiet museum" \
  --lora ./flux2-dev-style.safetensors \
  --output ./flux2-dev-style.png
```

For an eight-step fast path, pull the checksum-pinned `fal/FLUX.2-dev-Turbo`
adapter after you review its inherited FLUX.2-dev terms, and stack it with
your own adapter:

```bash
mere.run adapter pull flux2-dev-turbo-8step --accept-license
mere.run image generate \
  --model image-flux2-dev \
  --prompt "TRIGGER_TOKEN a glass sculpture in a quiet museum" \
  --lora flux2-dev-turbo-8step=1.0 \
  --lora ./flux2-dev-style.safetensors=0.8 \
  --output ./flux2-dev-turbo-style.png
```

The Turbo adapter switches the defaults to eight steps, guidance 2.5, and the
publisher's pre-shifted sigma schedule. An explicit `--cfg` overrides the
guidance, and `--sigmas` overrides the schedule; if you also pass
`--steps`, it must equal the number of nonterminal sigma values. Don't combine
`--sigmas` with `--sigma-shift`.

The license requires filters or manual review for generated content. The local
acceptance record doesn't certify that a workflow satisfies the license or the
BFL Acceptable Use Policy.

### FLUX.2 Klein

Klein models run through the native Swift FLUX.2 Klein runtime. The distilled
models (`image-klein-nano`, `image-klein-max`, `image-klein-9b`) default to four
steps. The base models (`image-klein-base`, `image-klein-base-9b`) default to
50 steps and are the targets for [LoRA training](#train-a-lora).

For Klein, `--input` is a single reference and uses the same pipeline as
`--ref-image`. Use `--ref-image` for repeatable references and clean default
reference conditioning; `--strength` defaults to 0.0 for Klein references.

```bash
mere.run image generate \
  --model image-klein-base \
  --prompt "A vertical trading card with the same sticker anatomy" \
  --ref-image ./card-reference.png \
  --output ./card.png
```

To apply a Klein LoRA to a reference composition, run it on a distilled Klein
model. A practical starting point for style transfer is `--strength 0.55`,
`--lora-scale 1.5`, 1024 x 768, 16 steps, and a fixed seed after the
composition is close. Put the trigger token first, then describe the subject
and action, visible anatomy, and style:

```bash
mere.run image generate \
  --model image-klein-9b \
  --ref-image ./reference-pose.png \
  --strength 0.55 \
  --prompt "TRIGGER_TOKEN two dancers in a rainy city street, full body pose, natural human anatomy, no extra limbs, clean cinematic film still, crisp faces, reflective pavement" \
  --lora ./style-adapter.safetensors \
  --lora-scale 1.5 \
  --width 1024 --height 768 \
  --steps 16 \
  --seed 525252 \
  --output ./style-reference.png
```

For cleaner exports, keep the seed and prompt fixed, render at a larger
matching aspect ratio such as 1280 x 960 with 24 steps, and downsample to
1024 x 768.

### Bonsai binary and ternary

`image-bonsai-binary` and `image-bonsai-ternary` map to PrismML's Apple
Silicon Bonsai Image snapshots. They run through the Klein runtime with the
upstream packed transformer layout (`transformer-packed-mflux`) and the 4-bit
MLX text encoder (`text_encoder-mlx-4bit`). Binary is the smallest 1-bit g128
deployment and uses native packed 1-bit affine matmul kernels on Metal and
Linux CUDA, with a dequantized MLX fallback for unsupported shapes. Ternary is
the larger 2-bit variant and uses the native 2-bit kernels on both backends.

```bash
mere.run model pull image-bonsai-binary
mere.run image generate \
  --model image-bonsai-binary \
  --prompt "A tiny bonsai tree in a sunlit greenhouse, editorial product photo" \
  --output ./bonsai.png
```

### HiDream O1

HiDream O1 supports text-only generation, one-reference instruction editing,
and multi-reference subject personalization. `--ref-image` is repeatable. With
a single reference, `--keep-original-aspect` keeps the reference's aspect
ratio for the output.

```bash
mere.run image generate \
  --model image-hidream-o1-dev \
  --prompt "A clean studio product photo of the subject" \
  --ref-image ./subject-front.png \
  --ref-image ./subject-side.png \
  --output ./subject.png
```

The runtime tokenizes the upstream chat-template prompt and runs generation
through the downloaded Qwen3-VL decoder, vision tower, and final pixel head.
Dev uses the fixed flash FlowMatch schedule with CFG 0; Full uses CFG 5.0 and
the shifted Flow UniPC scheduler. Reference modes run native Qwen3-VL vision
preprocessing before target and reference pixel patches enter denoising.

Installed HiDream smoke tests are opt-in because each checkpoint is large:

```bash
MERERUN_RUN_E2E=installed MERERUN_E2E_HIDREAM=1 ./scripts/check.sh
MERERUN_RUN_E2E=installed MERERUN_E2E_HIDREAM_FULL=1 ./scripts/check.sh
```

### SenseNova U1.5

`image-sensenova-u1-5-8b-mot` pins the official SenseNova U1.5 MoT checkpoint.
The runtime loads its dual understanding and generation experts, builds
reusable prefix KV caches, and denoises RGB pixels directly, with no separate
text encoder or VAE. Defaults match the published examples: 50 steps, CFG 4,
and timestep shift 3. Output dimensions must be multiples of 32.

```bash
mere.run model pull image-sensenova-u1-5-8b-mot

# Text to image
mere.run image generate \
  --model image-sensenova-u1-5-8b-mot \
  --prompt "A lunar greenhouse at blue hour, cinematic photography" \
  --width 2048 --height 2048 \
  --output ./lunar-greenhouse.png

# Instruction editing; repeat --ref-image for additional references.
mere.run image generate \
  --model image-sensenova-u1-5-8b-mot \
  --prompt "Keep the subject, replace the background with a snowy mountain pass" \
  --input ./subject.png \
  --width 2048 --height 2048 \
  --output ./mountain-edit.png
```

### Krea 2 Raw and Turbo

`image-krea2-raw` installs Krea's base checkpoint (`krea/Krea-2-Raw`) for LoRA
training. `image-krea2-turbo` (`krea/Krea-2-Turbo`) runs the distilled
eight-step text-to-image model. Both pulls use the split Diffusers component
layout and skip the duplicate root `raw.safetensors` and `turbo.safetensors`
transformer files.

Krea 2 is text-to-image with optional LoRA adapters; `--input` and reference
images aren't supported. Krea's published adapters are Diffusers-format
`lora_A`/`lora_B` files trained on Raw and rendered on Turbo at eight steps,
guidance 0, and weight 1.0. To train your own, see
[Train a Krea 2 adapter](#train-a-krea-2-adapter).

`image generate` also takes experimental Krea controls:
`--krea-conditioning-multiplier`, `--krea-conditioning-layer-weights`, and
`--krea-base-quantization-bits` (4 or 8), which quantizes the frozen
transformer and loads generation phases sequentially.

### Ideogram 4 SDNQ

`image-ideogram4-sdnq-uint4` maps to `WaveCut/ideogram-4-sdnq-uint4`. The pull
validates the SDNQ Diffusers layout, including the separate
`unconditional_transformer` branch used for guidance. The runtime decodes SDNQ
asymmetric uint4 linear, embedding, and Conv2d weights, builds Qwen3-VL text
features, runs positive and unconditional CFG denoising, and decodes through a
FLUX.2-style VAE. Only text-to-image is supported; image-to-image, references,
and LoRA aren't. Ideogram pairs well with [`--structured-prompt`](#expand-a-short-prompt).

The packer emits one uniform segment, so Ideogram skips the redundant block
mask exactly. At 1024 x 1024 with 4,096 image tokens, 64 text tokens, and 18
heads, this avoids a 16.50 MiB one-byte mask and a 1.160 GiB float32
masked-score intermediate for each layer. Those figures estimate avoided
transient allocation and memory traffic, not a guaranteed RSS reduction.
Layer buffers are reused, so don't multiply the per-layer number by the 34
layers to predict peak memory.

Custom query-key-value (QKV) normalization, adaptive layer normalization
(AdaLN), and residual Metal kernels are opt-in with
`MERERUN_IDEOGRAM4_FUSED_KERNELS=1`. Their isolated microbenchmarks were
faster, but a fixed 256 x 256, one-step installed-checkpoint release A/B
measured the warm portable graph at 2.532 seconds and the custom path at 4.169
seconds (1.65x slower), with the same roughly 2.94 GiB warm incremental MLX
peak. The two paths were internally deterministic and visually close but not
bit-exact, so the portable graph remains the default.

### Z-Image

`image-zimage-nano` is the default model. `image-zimage-base` can reuse the
tokenizer, text encoder, and VAE from an installed `image-zimage-nano` or
`image-zimage-max`.

Z-Image automatically uses dynamic sparse attention for sequences of at least
12,000 tokens. The first two transformer layers, the leading 20% of denoising,
and the final denoising step stay dense. A once-per-shape numerical gate falls
back to dense attention when the sparse route isn't faithful enough. The
sparse path is approximate, so same-seed high-resolution images can differ
from dense generation; smaller shapes stay on the byte-identical dense path.
In a warm 1792 x 1792 four-step run, it reduced generation time from 151.41 to
133.03 seconds (12.1%), and peak reported memory stayed about 109.8 GB. To keep
all attention dense, set `MERERUN_DYNAMIC_SPARSE_ATTENTION=0`.

## Train a LoRA

`image train-lora` trains an adapter from a folder of images with matching
UTF-8 `.txt` captions, and prints the adapter path to stdout. Two trainers are
available:

| Trainer | Train on | Run the adapter on | Recipes |
| --- | --- | --- | --- |
| FLUX.2 Klein | `image-klein-base-9b` or another Klein base | Distilled Klein models such as `image-klein-9b` | `klein-fast-style` |
| Krea 2 | `image-krea2-raw` (the default `--model`) | `image-krea2-turbo` | `krea-fast-style`, `krea-cinematic-style` |

Without a recipe, training uses 1,000 steps, learning rate `1e-4`, rank 16,
alpha equal to the rank, batch size 1, and 1024 x 1024. `--seed` defaults to
wall-clock time when omitted or zero. A `--recipe` sets curated values, and an
explicit flag always overrides the recipe. `mere.run image train-lora --help`,
`mere.run guide image train-lora`, and the [CLI reference](../cli.md) list
every option and which trainer accepts it.

mere.run checks options against the selected trainer before it resolves the
base model. Krea 2 refuses every Klein-only option, including
`--resume-from`, `--checkpoint-interval`, previews, and the target, rank,
timestep, and loss controls, instead of starting a different run. Both
trainers share the [image training operation](../internals/image-training-execution.md)
for recipe resolution, dataset preparation, and execution.

### Prepare a dataset

If you start from a project data root instead of a dataset folder, find
image-caption leaves first. This keeps root folders, evaluation folders, and
loose generated images out of training:

```bash
mere.run image dataset discover --root ./project-data --json
```

Use one of the returned candidate paths as `--data`. Add
`--training-recipe` or `--training-model` to get a ready-made
`train-lora --preflight` command for each candidate.

### Train a Klein adapter

`--recipe klein-fast-style` trains `image-klein-base-9b` for 1,000 steps at
learning rate `0.00005` and max side 512, with disk-backed latent caching
(`--low-ram`), `--no-compile`, 250-step checkpoints, and the `fal-klein-fast`
target preset at rank 16. To train every transformer `Linear` and
`QuantizedLinear` layer instead of the default suffix allowlist, use
`--lora-target-mode transformer-linear-walk`.

```bash
mere.run model pull image-klein-base-9b --accept-model-license
mere.run model pull image-klein-9b --accept-model-license
mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --visualize \
  --quiet
mere.run image generate \
  --model image-klein-9b \
  --prompt "TRIGGER_TOKEN a ceramic mug in the trained style" \
  --lora ./style-klein.safetensors \
  --lora-scale 2.0 \
  --output ./style-klein.png
```

The loader accepts mflux-format Klein transformer shards. To continue a Klein
run, pass a saved checkpoint to `--resume-from`; the trainer restores it before
the next optimizer step. Klein can also render preview samples during training
with `--sample-interval` (the default preview model is `image-klein-9b`).

### Train a Krea 2 adapter

Train on Raw, then run the adapter on Turbo:

```bash
mere.run model pull image-krea2-raw --accept-model-license
mere.run model pull image-krea2-turbo --accept-model-license
mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-krea2.safetensors \
  --recipe krea-cinematic-style \
  --quiet
mere.run image generate \
  --model image-krea2-turbo \
  --prompt "A cinematic product photo in the trained style" \
  --lora ./style-krea2.safetensors \
  --output ./speaker.png
```

The native Krea target set matches the published adapter surface: 264 `Linear`
modules covering image input, text projection and fusion, time embedding and
projection, transformer attention and feed-forward gates, and the final output
projection. The recipes are:

- `krea-fast-style`: a smoke-test recipe. 100 steps, learning rate `0.0005`,
  10-step warmup with cosine decay, 768 x 768, rank 32, and alpha 32. Inspect
  the images before you trust it as a final style adapter.
- `krea-cinematic-style`: a lower-learning-rate movie-style recipe. 200 steps,
  learning rate `0.0001`, 20-step warmup with cosine decay, 768 x 416, rank 32,
  alpha 32, and `--no-compile`. Override `--width` and `--height` when your
  source set isn't widescreen.

### Preflight and replay training

Run a preflight before you spend training time:

```bash
mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --preflight --json
```

Preflight inspects the dataset, resolved recipe, model availability, output
path, warnings, hard blockers, and next actions without loading the model. With
`--json`, it prints one JSON report to stdout that includes `result.run_plan`.
Save the plan and replay it with `image run-plan`:

```bash
mere.run image run-plan ./style.plan.json --preflight --json
mere.run image run-plan ./style.plan.json --materialize ./runs/style --json
mere.run image run-plan ./style.plan.json
```

A materialized LoRA run directory contains `plan.json`, `actions.json`,
`run.json`, and an initial events file. The materialized plan moves the output
adapter into that directory, so `image visualize-run` has a stable folder to
watch before training starts. Saved schema-v1 plans keep
`--base-quantization-bits`, including when you relocate the output path.

### Watch training

Add `--visualize` to `train-lora` to start a loopback dashboard with the live
loss curve, progress events, samples, checkpoints, and run artifacts (default
port 8787). To reopen a finished or copied run directory later, run:

```bash
mere.run image visualize-run ./runs/style
```

## Reconstruct a 3D mesh

Three subcommands turn object images into colored meshes. Each runs a native
runtime and has a `vision image-to-3d*` alias that uses the same
implementation.

| Command | Input | Model ID | Download | Memory (min / rec) |
| --- | --- | --- | --- | --- |
| `image reconstruct-3d` | One image (TripoSR) | `image-3d-triposr` | 1.7 GB | 16 / 32 GB |
| `image reconstruct-3d-trellis2` | One image, 512-resolution PBR O-Voxel mesh (TRELLIS.2) | `image-3d-trellis2-4b` | 11.0 GB | 64 / 128 GB |
| `image reconstruct-3d-multiview` | Exactly four or six supplied views (InstantMesh) | `image-3d-instantmesh-base` | 1.3 GB | 24 / 48 GB |

```bash
mere.run image reconstruct-3d ./object.png --output ./object-3d
mere.run image reconstruct-3d-trellis2 ./object.png \
  --output ./object-trellis2 \
  --seed 42
mere.run image reconstruct-3d-multiview \
  --view ./front.png --view ./right.png --view ./back.png --view ./left.png \
  --output ./object-instantmesh
```

- The single-image commands crop transparent PNG foregrounds automatically.
  TRELLIS.2 requires transparent alpha unless you pass `--already-framed`,
  which treats an opaque, isolated object as framed over black.
- All three accept `--dry-run` to verify inputs and checkpoints without loading
  weights, and `--json` for structured stdout.
- TripoSR takes `--resolution` from 2 to 512 (default 256), and InstantMesh
  from 2 to 256 (default 128). Both take `--no-vertex-colors` for
  geometry-only export.
- TRELLIS.2 takes `--texture-seed` to re-roll only the texture,
  `--seal-radius` and `--remesh-band` for the watertight remesh, and
  `--no-remesh` to export the raw dual-grid crust.
- InstantMesh doesn't generate views. It optionally takes a `--cameras` JSON
  file with one 16-value camera-to-world and intrinsics camera per view. It
  runs a verified converted safetensors package; the managed ID reports the
  required offline conversion.

The [Vision runtime](./vision.md) page covers the `vision image-to-3d*`
aliases and TRELLIS.2 output artifacts.

## Use macOS Studio

**Image ▸ Generate** covers text-to-image, image-to-image, multi-reference
editing, and a mask editor for masks and outpaint padding. Its settings include
negative prompts, size and sampling, edit strength, HiDream aspect
preservation, structured-prompt expansion, LoRA catalog IDs or local
safetensors, and Krea conditioning and quantization controls. The two to four
most important settings sit as chips under the prompt; the rest are in the
inspector (⌥⌘I), rendered from the capability contract.

**Image ▸ Train** is the trainer: image-caption previews, Krea 2 and FLUX.2
Klein recipes, memory and schedule controls, preflight, launch and resume, live
loss charts, samples, checkpoints, history, and run comparison.

**Image ▸ Datasets** covers deterministic validation, dataset-discovery
candidate cards, and saved-plan preflight and materialization with durable
paths.

Reconstruction lives in the **3D** domain: single-image TripoSR and TRELLIS.2
flows, ordered and reorderable four- and six-view InstantMesh inputs, every
engine-specific control, preflight, immutable output directories, manifest
statistics, and orbitable GLB, OBJ, and PLY previews. Progress and every mesh,
texture, voxel, and manifest artifact stay in the Library.

The **Command view** (⌥⌘C) shows the exact command your settings build, and
the **Command Console** window runs any image command the contract declares,
including ones without a designed surface. The app runs the public CLI for all
work. `mere.run catalog --json` and the shared contract tests keep every form
field aligned with the CLI help.

## Tune memory and throughput

Qwen Image Edit, Z-Image, FLUX.2 Klein, and HiDream O1 combine unconditional
and conditional CFG into one transformer batch on Apple Silicon Macs with at
least 24 GiB of physical memory and enough estimated MLX allocation headroom
for the requested resolution. The estimate subtracts MLX active and cache
allocations from physical memory; it isn't host-wide memory-pressure
telemetry. HiDream masks the padding used to align different prompt lengths.
Z-Image keeps the serial path when the two encoded prompt shapes can't be
paired exactly, and the memory-constrained FLUX.2 iOS pipeline keeps its
serial two-pass path.

To override the policy, set `MERERUN_IMAGE_BATCHED_CFG` to `batched` for
throughput or `serial` for lower memory. The model-specific
`MERERUN_QWEN_IMAGE_BATCHED_CFG`, `MERERUN_ZIMAGE_BATCHED_CFG`,
`MERERUN_FLUX2_BATCHED_CFG`, and `MERERUN_HIDREAM_BATCHED_CFG` variables take
precedence. Unset them, or use `auto`, for the memory-aware default.

::: warning
A forced `batched` policy ignores the headroom estimate. It can raise peak
memory or exhaust unified memory at large resolutions. It still falls back to
serial execution when conditioning shapes are incompatible.
:::

On Linux GPU hosts, FLUX.2 Klein evaluates its first denoising step one
transformer stage at a time and reports stages slower than one second, so a
cold CUDA kernel compile doesn't look like the process stopped responding. `MERERUN_FLUX2_STAGED_FIRST_STEP`
overrides that default, and `MERERUN_FLUX2_TIMING=1` prints every stage.

## Validate image families

`image validate` runs deterministic engineering checks for contributors who
change image internals. It covers the VAE, text encoder, transformer, and full
pipeline for the `zimage` (default) and `klein` families:

```bash
mere.run image validate --family zimage --test all
mere.run image validate --family klein --test pipeline
```

Use `--save-reference` to record reference latents and outputs, and
`--compare` with `--reference-dir` to check a change against them. Artifacts
go to `./validation_output` unless you pass `--output`.

## Source layout

Every family follows the same flow: the CLI parses the request, model
resolution maps the ID or path to a local root, the runtime loads the family's
components and prepares prompt and image conditioning, the denoising loop
runs, and the decoder writes the image. The internals differ by family:

| Area | Source |
| --- | --- |
| CLI commands | `Sources/MereRunCLI/Commands/ImageGenerateCommand.swift`, `ImageTrainLoRACommand.swift`, `ImageValidateCommand.swift` |
| Shared request, plan, and masks | `Sources/MereRunCore/ImageGenerationPlan.swift`, `ImageGenerationOperation.swift`, `ImageMaskEditing.swift`, `ImageRunRecord.swift` |
| Transformer layers and shared VAE | `Sources/MereRunImageModels/` (FLUX.2, FLUX 3, Z-Image, Qwen Image 2.1) |
| FLUX.2 Klein and Bonsai | `Sources/MereRunCore/Flux2Klein/` |
| FLUX 3 Klein | `Sources/MereRunCore/Flux3/` |
| FLUX.1-dev | `Sources/MereRunCore/Flux1/` |
| Z-Image | `Sources/MereRunCore/ZImageTurbo/` |
| Qwen Image 2.1 | `Sources/MereRunCore/QwenImage21/` |
| Qwen Image Edit 2511 | `Sources/MereRunCore/QwenImageEdit/` |
| HiDream O1 | `Sources/MereRunCore/HiDreamO1/` |
| SenseNova U1.5 | `Sources/MereRunCore/SenseNovaU15/` |
| Krea 2 | `Sources/MereRunCore/Krea2/` |
| Ideogram 4 | `Sources/MereRunCore/Ideogram4/` |
| LoRA training options | `Sources/MereRunCore/LoRA/` |
| 3D reconstruction | `Sources/MereRunCore/TripoSR/`, `Trellis2/`, `InstantMesh/` |

For the shared operation, see
[Image generation operation](../internals/image-generation-operation.md).

## Related docs

- [CLI reference](../cli.md)
- [Model management](./model-management.md)
- [Model sources](../model-sources.md)
- [Architecture reading map](../architecture.md)
