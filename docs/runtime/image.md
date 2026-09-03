# Image runtime

Use the image runtime to generate an image from a prompt, train a low-rank
adaptation (LoRA) on your own pictures, replay a generation from a saved plan,
or convert one photo into a textured three-dimensional (3D) mesh. The runtime
supports local model families from compact ZImage checkpoints through FLUX.1
Dev, FLUX.2 Dev and Klein, HiDream O1, SenseNova U1.5, Krea 2, Qwen Image Edit
2511, and Ideogram 4.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run image generate` | Generate images with local image models. |
| `mere.run image train-lora` | Train a local image LoRA adapter. |
| `mere.run image visualize-run` | Open a local LoRA training run viewer to watch loss while it trains. |
| `mere.run image dataset` | Inspect image training datasets, and discover candidates under a root. |
| `mere.run image run-plan` | Run a saved image workflow plan. |
| `mere.run image validate` | Run advanced deterministic validation for local image model families. |
| `mere.run image reconstruct-3d` | Reconstruct a colored object mesh from one image with native TripoSR. |
| `mere.run image reconstruct-3d-trellis2` | Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2. |
| `mere.run image reconstruct-3d-multiview` | Reconstruct a colored mesh from four or six supplied views with native InstantMesh. |

## macOS Studio

The primary Create Image workspace covers text-to-image, image-to-image, and
multi-reference editing. Its options include negative prompts, size and
sampling, edit strength, HiDream aspect preservation, structured-prompt
expansion, LoRA catalog IDs or local safetensors, Krea conditioning and
quantization controls, preflight JSON, and JSON progress events.

Create Image's **Create 3D** button opens the first-class reconstruction
workspace. It provides single-image TripoSR and TRELLIS.2 PBR flows, ordered and
reorderable 4/6-view InstantMesh inputs, every engine-specific runtime control,
preflight, immutable output directories, manifest statistics, and embedded
orbitable GLB/OBJ/PLY previews. Progress and every mesh, texture, voxel, and
manifest artifact remain in Library.

Create Image's **Training** button opens the unified Training Studio with
image-caption previews, Krea/Klein recipes, memory and schedule controls,
preflight, launch/resume, live loss charts, samples, checkpoints, history, and
run comparison. **Utilities** opens deterministic validation, dataset discovery
candidate cards, and saved-plan preflight/materialization with durable paths.

**Advanced > Image** also exposes the public command family as guided forms:

- Generation and editing with repeatable references, structured prompts, and LoRAs
- Krea 2 and FLUX.2 Klein LoRA training, recipes, memory controls, checkpoints,
  preview samples, schedules, benchmarks, and the live dashboard;
- Dataset discovery, saved-workflow preflight, materialization, execution, and
  durable-run visualization
- Deterministic image-stack validation
- TripoSR, TRELLIS.2, and four-view or six-view InstantMesh reconstruction

The app launches the public CLI for all work. `mere.run catalog --json` and the
shared contract tests keep every form flag aligned with ArgumentParser help.

## Model families

The public image families are:

- `image-klein-*`: Klein image family
- `image-flux1-dev`: gated FLUX.1-dev generation and LoRA inference
- `image-flux2-dev`: gated FLUX.2-dev generation and LoRA inference
- `image-bonsai-binary`: PrismML Bonsai binary FLUX.2 Klein deployment
- `image-bonsai-ternary`: PrismML Bonsai ternary FLUX.2 Klein deployment
- `image-zimage-*`: ZImage image family
- `image-hidream-o1*`: HiDream O1 unified pixel-transformer family
- `image-sensenova-u1-5-8b-mot`: SenseNova U1.5 raw-pixel generation and editing family
- `image-krea2-raw`: Krea 2 Raw base checkpoint for LoRA training
- `image-krea2-turbo`: Krea 2 Turbo text-to-image and LoRA inference family
- `image-qwen-edit-2511`: Qwen Image Edit 2511 40-step quality lane
- `image-qwen-edit-2511-lightning`: pinned four-step Qwen Edit Lightning lane
- `image-ideogram4-sdnq-uint4`: Ideogram 4 SDNQ uint4 text-to-image family

Common managed IDs:

- `image-flux1-dev`
- `image-flux2-dev`
- `image-klein-nano`
- `image-klein-base`
- `image-klein-base-9b`
- `image-klein-max`
- `image-bonsai-binary`
- `image-bonsai-ternary`
- `image-zimage-nano`
- `image-zimage-base`
- `image-zimage-max`
- `image-hidream-o1-dev`
- `image-hidream-o1`
- `image-sensenova-u1-5-8b-mot`
- `image-krea2-raw`
- `image-krea2-turbo`
- `image-qwen-edit-2511`
- `image-qwen-edit-2511-lightning`
- `image-ideogram4-sdnq-uint4`

## Typical workflows

### Generate an image

```bash
swift run mere.run image generate \
  --model image-zimage-nano \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png
```

Preflight the same request when a script, agent, or app needs structured
model/input/output checks before generation starts:

```bash
swift run mere.run image generate \
  --model image-zimage-nano \
  --prompt "a ceramic mug in soft morning light" \
  --output ./mug.png \
  --preflight \
  --json
```

The JSON report uses the shared structured-run envelope and includes
diagnostics plus declarative actions such as `start-generation`, `pull-model`,
and `open-output-directory`. It also includes `result.run_plan`, a normalized
`image.generate` plan that can be saved and replayed:

```bash
swift run mere.run image run-plan ./mug.plan.json --preflight --json
swift run mere.run image run-plan ./mug.plan.json --materialize ./runs/mug --json
swift run mere.run image run-plan ./mug.plan.json
```

Materialized generation directories contain `plan.json`, `actions.json`,
`run.json`, an initial events file, and artifact folders. The materialized plan
relocates the output image and optional structured-prompt sidecar into that
directory; executing it appends `run_started`, `run_finished`, or `run_failed`
events to the same stream. Hard blockers exit nonzero after the report is
printed.

### Qwen Image Edit 2511

The native 2511 path accepts one to three ordered images. `--input` is Picture
1; each repeated `--ref-image` appends the next picture. Every picture keeps
its own aspect ratio and conditioning grid. The explicit `--width` and
`--height` control the edited output independently.

```bash
swift run mere.run model pull image-qwen-edit-2511
swift run mere.run image generate \
  --model image-qwen-edit-2511 \
  --input ./scene.png \
  --ref-image ./material.png \
  --prompt "Replace the chair in Picture 1 with the material from Picture 2" \
  --output ./scene-edited.png
```

The quality model defaults to 40 steps and true CFG 4.0. The separate
`image-qwen-edit-2511-lightning` model includes a checksum-pinned BF16 adapter
and is intentionally strict: it requires four steps and CFG 1.0. Selecting four
steps on the quality model does not silently activate Lightning weights.

```bash
swift run mere.run model pull image-qwen-edit-2511-lightning
swift run mere.run image generate \
  --model image-qwen-edit-2511-lightning \
  --input ./scene.png \
  --prompt "Change the chair to cobalt blue" \
  --output ./scene-fast.png
```

The base download is about 57.7 GB. Hardware-specific memory, quality, and
performance qualification remains required before treating either lane as a
released default.

### Generate with FLUX.1-dev LoRAs

FLUX.1-dev runs through a dedicated Swift and MLX runtime. The managed package
pins the official Diffusers transformer, CLIP-L and T5-XXL text encoders,
tokenizers, VAE, and flow-matching scheduler. The download is approximately
34 GB.

To install the model, accept access on Hugging Face, configure a Hugging Face
token, and acknowledge the FLUX.1 dev Non-Commercial License v1.1.1:

```bash
swift run mere.run model pull image-flux1-dev --accept-model-license
```

To run one or more FLUX.1 adapters, repeat `--lora`. Each value accepts an
independent inline scale as `PATH_OR_ID=SCALE`:

```bash
swift run mere.run image generate \
  --model image-flux1-dev \
  --prompt "TRIGGER_TOKEN a glass sculpture in a quiet museum" \
  --lora ./flux1-fast.safetensors=1.0 \
  --lora ./flux1-style.safetensors=0.8 \
  --output ./flux1-fast-style.png
```

The model defaults to 28 denoising steps and embedded guidance scale 3.5.
FLUX.1 adapters target a 3,072-wide transformer. FLUX.2-dev uses a 6,144-wide
transformer, and Klein uses a separate architecture. mere.run rejects adapters
that don't match the selected model before denoising.

The license limits the weights and derivatives to non-commercial,
non-production use. It also requires filters or manual review for generated
content. The local acceptance record doesn't certify that a workflow complies
with the license.

### Generate with FLUX.2-dev LoRAs

FLUX.2-dev runs through the shared Swift and MLX FLUX.2 runtime. The managed
layout keeps the gated BFL transformer unchanged and uses a pinned 4-bit
Mistral Small 3.2 text encoder. Existing FLUX.2-dev transformer LoRAs remain
compatible with this layout.

To install the model, accept access on Hugging Face, configure a Hugging Face
token, and acknowledge the FLUX Non-Commercial License and BFL Acceptable Use
Policy:

```bash
swift run mere.run model pull image-flux2-dev --accept-model-license
```

To run an adapter, pass its local safetensors file to `--lora`. Repeat the
option to stack adapters in order. Each value accepts an inline scale as
`PATH_OR_ID=SCALE`; `--lora-scale` remains the default for values without an
inline scale.

```bash
swift run mere.run image generate \
  --model image-flux2-dev \
  --prompt "TRIGGER_TOKEN a glass sculpture in a quiet museum" \
  --lora ./flux2-dev-style.safetensors \
  --lora-scale 1.0 \
  --output ./flux2-dev-style.png
```

For the fast path, pull the checksum-pinned `fal/FLUX.2-dev-Turbo` adapter after
reviewing and accepting its inherited FLUX.2-dev terms:

```bash
swift run mere.run adapter pull flux2-dev-turbo-8step --accept-license
swift run mere.run image generate \
  --model image-flux2-dev \
  --prompt "TRIGGER_TOKEN a glass sculpture in a quiet museum" \
  --lora flux2-dev-turbo-8step=1.0 \
  --lora ./flux2-dev-style.safetensors=0.8 \
  --output ./flux2-dev-turbo-style.png
```

Selecting the managed Turbo adapter defaults to eight steps, guidance scale
2.5, and the publisher's pre-shifted sigma schedule. An explicit `--cfg`
overrides the guidance default. An explicit `--sigmas` overrides the recipe;
when you also pass `--steps`, its value must equal the number of nonterminal
sigma values. Don't combine `--sigmas` with `--sigma-shift`.

Every adapter in a stack must target the exact loaded architecture. In
particular, FLUX.1-dev adapters use a 3,072-wide transformer and aren't
compatible with the 6,144-wide FLUX.2-dev transformer. mere.run rejects these
shape mismatches before denoising.

The model defaults to 50 steps and embedded guidance scale 4.0. The managed
download is approximately 78 GB. The capability catalog requires 96 GB of
unified memory and recommends 128 GB.

The license requires filters or manual review for generated content. The local
acceptance record does not certify that a workflow satisfies the license or the
BFL Acceptable Use Policy.

### Klein generation and LoRA training

Klein models run through the native Swift FLUX.2 Klein runtime. Base Klein
models can also train LoRA adapters with `image train-lora`. For production Klein
LoRA training, use the undistilled BF16 `image-klein-base-9b` model id or a
local equivalent model root, then use the saved adapter with Klein
`image generate --lora`. The loader accepts mflux-format Klein transformer
shards and maps their time-guidance weights into the Swift transformer module
layout.
Use `--recipe klein-fast-style` for the canonical fast local style recipe. It
trains on `image-klein-base-9b` at rank 16 for 1000 steps, LR `0.00005`, max
side `512`, disk-backed latent caching, compiled-step disablement, 250-step
checkpoints, and the fast Klein target surface. Use
`--lora-target-mode transformer-linear-walk` when you want an ai-toolkit-style
comparison that trains every transformer Linear/QuantizedLinear layer instead
of the default suffix allowlist.

Core `train-lora` hyperparameters and their defaults:

- `--training-steps` / `--steps`: number of optimizer steps; default `1000`
- `--batch-size`: training batch size; default `1`
- `--learning-rate` / `--lr`: learning rate; default `1e-4`
- `--rank`: LoRA rank; default `16`
- `--alpha`: LoRA alpha; defaults to the rank
- `--seed`: random seed; defaults to wall-clock time when omitted or zero
- `--resume-from`: continue a FLUX.2 Klein run from a saved adapter checkpoint

`--recipe` presets set curated values for steps, learning rate, rank, and
alpha; an explicit flag always wins over the recipe. `swift run mere.run image
train-lora --help` and the [CLI reference](../cli.md) list the full flag
surface.

Klein resume restores the selected adapter checkpoint before the next optimizer
step. Krea 2 rejects `--resume-from` explicitly instead of silently starting
another run. Training Studio discovers checkpoint artifacts beside a prior output
and sends the same public flag.

```bash
swift run mere.run model pull image-klein-base-9b --accept-model-license
swift run mere.run model pull image-klein-9b --accept-model-license
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --visualize \
  --quiet
swift run mere.run image generate \
  --model image-klein-9b \
  --prompt "TRIGGER_TOKEN a ceramic mug in the trained style" \
  --lora ./style-klein.safetensors \
  --lora-scale 2.0 \
  --output ./style-klein.png
```

For reference-guided Klein LoRA inference, run the adapter on the distilled
Klein model and pass the source composition with `--ref-image`. A practical
starting point for style transfer is `--strength 0.55`, `--lora-scale 1.5`,
1024 x 768, 16 steps, and a locked seed after the composition is close. Put the
trigger token first, then describe the subject/action relationship, visible
anatomy, and style:

```bash
swift run mere.run image generate \
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

For cleaner public-facing exports, keep the seed and prompt fixed, render at a
larger matching aspect ratio such as 1280 x 960 with 24 steps, and then
downsample to 1024 x 768 with a high-quality image resizer.

If you are starting from a project data root instead of a specific dataset
folder, discover image-caption leaves first:

```bash
swift run mere.run image dataset discover \
  --root ./project-data \
  --json
```

Use one of the returned candidate paths as `--data` for training. This keeps
root folders, eval folders, and loose generated images out of the training path.

Run a preflight before spending training time:

```bash
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-klein.safetensors \
  --recipe klein-fast-style \
  --preflight \
  --json
```

The preflight path inspects the dataset, resolved recipe, model availability,
output path, warnings, hard blockers, and next actions without loading the full
model or starting training. It prints a single JSON report to stdout when
`--json` is set; diagnostics stay out of stdout. The JSON includes
`result.run_plan`, which can be saved and replayed without reconstructing CLI
state:

```bash
swift run mere.run image run-plan ./style.plan.json --preflight --json
swift run mere.run image run-plan ./style.plan.json --materialize ./runs/style --json
swift run mere.run image run-plan ./style.plan.json
```

Materialized LoRA run directories contain `plan.json`, `actions.json`,
`run.json`, and an initial events file. The materialized `plan.json` moves the
output adapter path into that run directory so `image visualize-run` has a
stable folder to watch before training starts. Running that materialized plan
appends started, progress, finished, or failed events to the same stream.

Add `--visualize` during training to start a loopback dashboard with the live
loss curve, progress events, samples, checkpoints, and run artifacts. Reopen a
completed or copied run directory later with:

```bash
swift run mere.run image visualize-run ./runs/my-style
```

### Bonsai binary and ternary

`image-bonsai-binary` and `image-bonsai-ternary` map to PrismML's Apple Silicon
Bonsai Image snapshots. They run through the native Swift FLUX.2 Klein runtime
using the upstream packed transformer layout (`transformer-packed-mflux`) and
the 4-bit MLX text encoder layout (`text_encoder-mlx-4bit`). Binary is the
smallest 1-bit g128 deployment; ternary is the larger 2-bit quality-oriented
variant. The binary path uses native packed 1-bit affine matmul kernels on
Metal and Linux CUDA, with a dequantized MLX fallback for non-GPU or
unsupported shapes. The ternary path continues through the existing native
2-bit kernels on both backends.

```bash
swift run mere.run model pull image-bonsai-binary
swift run mere.run image generate \
  --model image-bonsai-binary \
  --prompt "a tiny bonsai tree in a sunlit greenhouse, editorial product photo" \
  --width 1024 --height 1024 \
  --output ./bonsai.png
```

### Image-to-image and Klein references

```bash
swift run mere.run image generate \
  --prompt "turn this into a pencil sketch" \
  --input ./photo.png \
  --strength 0.6 \
  --output ./sketch.png
```

For FLUX.2 Klein, `--input` is treated as a single reference image and is routed
through the same reference-image pipeline as `--ref-image`. Use `--ref-image`
directly when you want repeatable Klein references, or when you want the clean
default reference conditioning.

```bash
swift run mere.run image generate \
  --model image-klein-base \
  --prompt "a vertical gross-out trading card with the same sticker anatomy" \
  --ref-image ./card-reference.png \
  --output ./card.png
```

### HiDream O1 references

HiDream O1 is registered with text-only, one-reference instruction editing, and
multi-reference subject-personalization capabilities. Reference images are
repeatable; with a single reference, `--keep-original-aspect` preserves the
reference aspect ratio when building the HiDream sample.

The native runtime validates model roots, decodes the typed upstream
configuration, tokenizes the upstream chat-template prompt, builds scheduler
inputs, constructs text/reference sample metadata, and runs HiDream generation
through the downloaded Qwen3-VL decoder, vision tower, timestep embedder, patch
embedder, generation-aware attention mask, and final pixel head. Dev uses the
fixed flash FlowMatch schedule with CFG 0.0 by default; Full uses CFG 5.0 by
default and the shifted Flow UniPC scheduler. Reference-image modes run native
Qwen3-VL vision preprocessing and replace chat-template image placeholders
before appending target/reference pixel patches for denoising.

```bash
swift run mere.run image generate \
  --model image-hidream-o1-dev \
  --prompt "a clean studio product photo of the subject" \
  --ref-image ./subject-front.png \
  --ref-image ./subject-side.png \
  --output ./subject.png
```

Use `--steps` or `--cfg` when you want to override the model-specific defaults.
Use `--keep-original-aspect` with a single `--ref-image` for edit cases where
the output should follow the source image aspect ratio.

Installed HiDream smoke tests are intentionally opt-in because each checkpoint
is large and GPU time is meaningful:

```bash
MERERUN_RUN_E2E=installed MERERUN_E2E_HIDREAM=1 ./scripts/check.sh
MERERUN_RUN_E2E=installed MERERUN_E2E_HIDREAM_FULL=1 ./scripts/check.sh
```

### SenseNova U1.5 generation and editing

`image-sensenova-u1-5-8b-mot` pins the official SenseNova U1.5 MoT checkpoint.
The native Swift/MLX runtime loads its dual understanding/generation experts,
builds reusable prefix KV caches, and denoises RGB pixels directly; it does not
launch Python and does not require a separate text encoder or VAE. The managed
manifest defaults match the published examples: 50 steps, CFG 4, and timestep
shift 3. Output dimensions must be multiples of 32.

```bash
swift run mere.run model pull image-sensenova-u1-5-8b-mot

# Text to image
swift run mere.run image generate \
  --model image-sensenova-u1-5-8b-mot \
  --prompt "A lunar greenhouse at blue hour, cinematic photography" \
  --width 2048 --height 2048 \
  --output ./lunar-greenhouse.png

# Instruction editing; repeat --ref-image for additional references.
swift run mere.run image generate \
  --model image-sensenova-u1-5-8b-mot \
  --prompt "Keep the subject, replace the background with a snowy mountain pass" \
  --input ./subject.png \
  --width 2048 --height 2048 \
  --output ./mountain-edit.png
```

The BF16 checkpoint is about 50 GB on disk. The capability gate requires at
least 64 GB unified memory and recommends 96 GB for generation headroom.

### Krea 2 Raw and Turbo

`image-krea2-raw` maps to `krea/Krea-2-Raw` and installs Krea's base
checkpoint for LoRA training. `image-krea2-turbo` maps to `krea/Krea-2-Turbo`
and uses the native Swift MLX runtime for Krea's distilled 8-step text-to-image
model. Both managed pulls use the split Diffusers component layout and
deliberately skip the root `raw.safetensors` / `turbo.safetensors` duplicate
transformer files.

Train adapters on Raw, then preview or run them on Turbo:

```bash
swift run mere.run model pull image-krea2-raw --accept-model-license
swift run mere.run model pull image-krea2-turbo --accept-model-license
swift run mere.run image train-lora \
  --data ./style-dataset \
  --output ./style-krea2.safetensors \
  --recipe krea-cinematic-style \
  --quiet
swift run mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a cinematic product photo in the trained style" \
  --lora ./style-krea2.safetensors \
  --width 1024 --height 1024 \
  --steps 8 \
  --output ./speaker.png
```

Krea's published LoRA examples use Diffusers-format `lora_A` / `lora_B`
adapters trained on Raw, rendered on Turbo at 8 steps, guidance 0.0, and LoRA
weight 1.0. The native Krea target set matches the published adapter surface:
264 Linear modules on the full model, including image input, text
projection/fusion, time embedding/projection, transformer attention/feed-forward
gates, and final output projection.
Use `--recipe krea-fast-style` for a local Krea smoke test: Raw base, 100
steps, LR `0.0005`, 10-step warmup/cosine decay, 768 square, rank `32`, alpha
`32`, and the full native Krea target surface. Treat this as a smoke recipe and
inspect images before trusting it as a final style adapter. Use
`--recipe krea-cinematic-style` for the lower-learning-rate movie-style lane:
200 steps, learning rate (LR) `0.0001`, 20-step warmup and cosine decay,
768 x 416, rank `32`, alpha `32`, and
compiled-step disablement. Override `--width`/`--height` when your source set is
not widescreen.

The wired Krea generation mode is text-to-image with optional LoRA adapters;
image-to-image and reference inputs are not wired for this family yet.

### Ideogram 4 SDNQ

`image-ideogram4-sdnq-uint4` maps to `WaveCut/ideogram-4-sdnq-uint4`. The
managed model path can pull and validate the SDNQ diffusers layout, including
the separate `unconditional_transformer` branch used for guidance. The native
quantization bridge decodes SDNQ asymmetric uint4 linear, embedding, and Conv2d
weights, the runtime builds Qwen3-VL concatenated text features, packs Ideogram
4 text/image samples, runs positive/unconditional CFG denoising, and decodes
through the Flux2-style VAE. Plain text-to-image generation is wired through
`image generate`; image-to-image, reference inputs, and LoRA are not supported
for this family yet.

Ideogram also works with the `image generate --structured-prompt` adapter. The
adapter uses a local text chat model to expand a short prompt into a long,
FIBO-style structured JSON caption covering objects, background, lighting,
aesthetics, camera characteristics, style, context, and text renders. When the
adapter is enabled, the CLI raises the image prompt token budget to 2048 unless
the user already requested a larger value. Use `--structured-prompt-output` to
save the generated JSON for review or later refinement.

The supported packer emits one uniform segment, so Ideogram skips the redundant
block mask exactly and by default. At a representative 1024 x 1024 shape with
4096 image tokens, 64 text tokens, and 18 heads, this avoids a 16.50 MiB
one-byte mask plus a 1.160 GiB float32 masked-score graph intermediate per
layer. Those are avoided transient allocation and memory-traffic estimates, not
a guaranteed RSS reduction; layer buffers are reused, so the per-layer number
must not be multiplied by the model's 34 layers to predict peak memory.

Custom query-key-value (QKV) normalization, adaptive layer normalization
(AdaLN), and residual Metal kernels remain opt-in with
`MERERUN_IDEOGRAM4_FUSED_KERNELS=1`. Although their isolated microbenchmarks
were faster, a fixed 256 x 256, one-step installed-checkpoint release A/B measured
the warm portable graph at 2.532 seconds and the custom path at 4.169 seconds, 1.65x slower,
with the same roughly 2.94 GiB warm incremental MLX peak. The two paths were
internally deterministic and visually close but not bit-exact, so the portable
graph remains the production default. The exact uniform-mask removal is
independent of this experimental switch.

### 3D reconstruction

Three subcommands turn object images into colored meshes, each backed by a
native runtime:

- `image reconstruct-3d`: single-image reconstruction with native TripoSR
  (managed ID `image-3d-triposr`). This is the canonical image-family spelling
  of `vision image-to-3d`; both run the same implementation.
- `image reconstruct-3d-trellis2`: single-image 512-resolution PBR O-Voxel
  reconstruction with native MLX TRELLIS.2 (managed ID `image-3d-trellis2-4b`).
- `image reconstruct-3d-multiview`: reconstruction from views you supply with
  native InstantMesh (managed ID `image-3d-instantmesh-base`). Repeat `--view`
  exactly four or six times; the command does not generate views.

```bash
swift run mere.run image reconstruct-3d ./object.png --output ./object-3d
swift run mere.run image reconstruct-3d-trellis2 ./object.png \
  --output ./object-trellis2 \
  --seed 42
swift run mere.run image reconstruct-3d-multiview \
  --view ./front.png --view ./right.png --view ./back.png --view ./left.png \
  --output ./object-instantmesh
```

The single-image commands crop transparent PNG foregrounds automatically;
TRELLIS.2 requires transparent alpha unless `--already-framed` opts an opaque,
isolated object into black-background conditioning. All three accept
`--dry-run` to verify inputs and checkpoints and print the execution plan
without loading weights, and `--json` for structured stdout. TripoSR and
InstantMesh take a `--resolution` extraction-grid override (2 through 512 and
2 through 256 respectively) plus `--no-vertex-colors` for geometry-only
export; multiview optionally takes a `--cameras` JSON file with one 16-value
C2W transformation and intrinsics camera per view. The
[Vision runtime](./vision.md) page covers
the equivalent `vision image-to-3d*` commands and TRELLIS.2 output artifacts.

### Deterministic validation

```bash
swift run mere.run image validate --family zimage --test all
swift run mere.run image validate --family klein --test pipeline
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/ImageGenerateCommand.swift`
- `Sources/MereRunCLI/Commands/ImageValidateCommand.swift`

### Klein family

- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator.swift`
- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+ModelLoading.swift`
- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Generation.swift`
- `Sources/MereRunCore/Flux2Klein/Flux2KleinGenerator+Chat.swift`

### ZImage family

- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator.swift`
- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+ModelLoading.swift`
- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+Inference.swift`
- `Sources/MereRunCore/ZImageTurbo/ZImageTurboGenerator+LoRA.swift`

Z-Image automatically uses dynamic sparse attention for sequences of at least
12,000 tokens. The first two transformer layers, the leading 20% of denoising,
and the final denoise step stay dense, and a once-per-shape numerical gate falls
back to dense attention whenever the sparse route is not sufficiently faithful.
Set `MERERUN_DYNAMIC_SPARSE_ATTENTION=0` to keep all attention dense. The sparse
path is approximate, so same-seed high-resolution images can differ from dense
generation; smaller shapes stay on the byte-identical dense path. In a warm
1792x1792 four-step run it reduced generation time from 151.41 to 133.03 seconds
(12.1%) while peak reported memory remained approximately 109.8 GB.

### HiDream O1 family

- `Sources/MereRunCore/HiDreamO1/HiDreamO1Generator.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Resources.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Configs.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Model.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1SampleBuilder.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1ImagePreprocessor.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1TokenizerAndTemplate.swift`
- `Sources/MereRunCore/HiDreamO1/HiDreamO1Scheduler.swift`

### SenseNova U1.5 family

- `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Generator.swift`
- `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Model.swift`
- `Sources/MereRunCore/SenseNovaU15/SenseNovaU15VisionAndHead.swift`
- `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Tokenizer.swift`
- `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Scheduler.swift`
- `Sources/MereRunCore/SenseNovaU15/SenseNovaU15ImageIO.swift`
- `Sources/MereRunCore/SenseNovaU15/SenseNovaU15Resources.swift`

### Krea 2 family

- `Sources/MereRunCore/Krea2/Krea2Generator.swift`
- `Sources/MereRunCore/Krea2/Krea2LoRAInjector.swift`
- `Sources/MereRunCore/Krea2/Krea2LoRATrainer.swift`
- `Sources/MereRunCore/Krea2/Krea2RawResources.swift`
- `Sources/MereRunCore/Krea2/Krea2Resources.swift`
- `Sources/MereRunCore/Krea2/Krea2Configs.swift`
- `Sources/MereRunCore/Krea2/Krea2Model.swift`
- `Sources/MereRunCore/Krea2/Krea2ModelLoader.swift`
- `Sources/MereRunCore/Krea2/Krea2SampleBuilder.swift`

### Image editing support

- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator.swift`
- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+ModelLoading.swift`
- `Sources/MereRunCore/QwenImageEdit/QwenImageEditGenerator+Encoding.swift`

Qwen Image Edit, Z-Image, FLUX.2 Klein, and HiDream O1 automatically combine
unconditional and conditional CFG into one transformer batch on Apple Silicon
Macs with at least 24 GiB of physical memory and sufficient estimated MLX
allocation headroom for the requested resolution. The estimate subtracts MLX
active and cache allocations from physical memory; it is not host-wide
available-memory or operating-system pressure telemetry. HiDream masks the
padding used to align different prompt lengths; Z-Image keeps the serial path
when the two encoded prompt shapes cannot be paired exactly. The
memory-constrained FLUX.2 iOS pipeline also keeps its serial two-pass path.

Set `MERERUN_IMAGE_BATCHED_CFG=batched` to force the throughput policy or
`MERERUN_IMAGE_BATCHED_CFG=serial` to force the lower-memory policy. The
model-specific `MERERUN_QWEN_IMAGE_BATCHED_CFG`,
`MERERUN_ZIMAGE_BATCHED_CFG`, `MERERUN_FLUX2_BATCHED_CFG`, and
`MERERUN_HIDREAM_BATCHED_CFG` variables take precedence. Unset the variables
(or use `auto`) for the memory-aware default. A forced batched policy bypasses
the headroom check but still falls back to serial execution when conditioning
shapes are incompatible. Because forced batching ignores the estimate, it can
increase peak memory or exhaust unified memory at large resolutions.

## How image generation flows

Image generation follows these steps:

1. The CLI parses the prompt, model choice, size, steps, optional image input,
   and optional reference images.
2. Model resolution maps a canonical model ID or explicit path to a local root.
3. The runtime loads the matching components for the chosen family.
4. The runtime prepares prompt encoding and optional conditioning data.
5. The denoise loop or family-specific generation path runs.
6. The runtime decodes the latents and writes an image artifact.

The image families do not share identical implementation internals, but they are
presented through the same public `mere.run image generate` command.

## Validation philosophy

`mere.run image validate` exists so contributors can run deterministic checks on:

- VAE behavior
- text-encoder behavior
- transformer behavior
- full pipeline behavior

It is intentionally more engineering-oriented than normal end-user workflows.
If you change image internals, this command is the first place to verify that a
family still behaves consistently.

## Related docs

- [CLI reference](../cli.md)
- [Model management](./model-management.md)
- [Architecture reading map](../architecture.md)
