# MiniMax-H3-Image native checkpoint qualification

The native Swift/MLX runtime generated coherent 2048 × 2048 and 4096 × 4096
images from the original released checkpoint on an Apple M5 Max with 40 GPU
cores and 128 GiB
of unified memory, running macOS 27.2 on AC power. The first real-checkpoint
test exposed a reversed Euler update, which was corrected before the successful
run. The images below are AI-generated qualification outputs.

## Checkpoint and measurements

The source is
[MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/tree/492fd8c4eca666f1085f57e30738c1e0d86b3439)
at revision `492fd8c4eca666f1085f57e30738c1e0d86b3439`. Hugging Face cache
verification checked all 55 selected files against the pinned source. The
download contains about 135 GB, including the text encoder. Duplicate
Diffusers-formatted weights and showcase assets were omitted.

Checkpoint files reside on the external MODELS volume. The release CLI runs
the original BF16 transformer and conditioner, preserves
the checkpoint's FP32 projections, and decodes with the original FP32 image
VAE. Components load and release in stages. Both image sizes use six PDD
evaluations, CFG 1, seed 42, and the same detailed English prompt. Python only
records measurements and inspects the completed PNG; inference runs in the
native CLI process.

`/usr/bin/time -l` measures wall time, maximum resident size, and peak process
footprint. Wall time includes admission, loading, first-use weight reads,
prompt encoding, denoising, VAE decoding, and PNG output. Footprint is an OS
process measurement, rather than an MLX allocation counter. A GiB is 2³⁰ bytes.
Host-wide swap use grew from about 4.35 GiB before qualification to 9.35 GiB
afterward. These global counters include background processes; they do not
attribute the increase to the inference child alone.

These are single runs on a shared host. Repository builds and tests overlapped
parts of the runs, and file caches were not purged. The results establish
observed execution costs, not isolated throughput or a minimum supported RAM
configuration. Binary hashes, inference-source hashes, progress timings,
output hashes, and visual findings are in the
[JSON receipt](/qualification/minimax-h3-image-native-2026-11-10.json).

## Results

| Output | Wall seconds | Denoising seconds | Peak process footprint (GiB) |
| --- | ---: | ---: | ---: |
| 2048 × 2048 | 212.91 | 143.48 | 67.35 |
| 4096 × 4096 | 498.93 | 424.66 | 72.12 |

The 2K measurement precedes the large-sequence scheduling change described
below; its sequence does not enter that branch. The receipt retains the
separate binary and source hashes for both successful runs.

The 2K run encoded its prompt in 54.84 seconds. Its first denoising step took
79.70 seconds, including lazy transformer weight materialization; the remaining
five steps took about 12–15 seconds each. Decode and image preparation took
7.69 seconds. This phase split should not be read as a pure kernel benchmark.

The 4K run encoded its prompt in 54.80 seconds. Its first denoising step took
126.38 seconds including weight materialization; the remaining five took
58–62 seconds each. Decode and image preparation took 12.06 seconds. The
output is generated at 4096 × 4096 directly, with no upscaling stage.

### Upstream speed comparison

The [SGLang recipe](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3-Image.mdx#dgx-spark-at-2k-and-4k),
checked on November 10, reports a median of 84.96 seconds for 4736 × 3552
text-to-image generation on one DGX Spark GB10. It uses native precision,
resident components, precomputed AdaLN, and warm conditioning caches. The
median covers 10 warm requests across two servers, excluding startup, HTTP,
PNG encoding, and saving. Its 16 MP canvas has a different aspect ratio from
the square 4K case here, and its hardware differs from the M5 Max.

Our 498.93-second result includes first-use loading and output work, but those
costs explain only part of the gap. The five later denoising steps average
59.66 seconds; six comparable steps would take about 358 seconds before
conditioning and decoding. This extrapolation is not a measured warm
resident-pipeline result. It identifies a substantial execution gap that
requires profiling, including GPU work-unit overhead and memory pressure.
Q8 speed improvements and a 70-second Apple Silicon target remain unverified.

### Image quality

The 2K output follows the main composition: a brass rangefinder-style camera,
one primary lens, textured black covering, a blue cup behind on the left, and
a walnut tabletop. Metal grain, lens reflections, leather texture, and wood
grain are visible. Review of the whole frame and two unscaled 512-pixel crops
found no remaining mosaic corruption or obvious tile seams. The cup is larger
than requested, and the engraved lens markings are invented and malformed.

![Native 2K brass camera image, downsampled for this page.](/qualification/minimax-h3-image-native-2026-11-10/camera-2048-s42.jpg)

![Unscaled 2K lens and tabletop crops, saved as JPEG previews.](/qualification/minimax-h3-image-native-2026-11-10/camera-2048-s42-details.jpg)

The 4K output preserves the same main subjects in a different composition,
with more empty background above the camera. Unscaled lens, grip, and tabletop
crops show fine lens rings, metal wear, and detailed black covering. No mosaic
corruption or obvious tile seams were found in the reviewed frame and crops.
The invented markings and oversized cup remain limitations.

![Native 4K brass camera image, downsampled for this page.](/qualification/minimax-h3-image-native-2026-11-10/camera-4096-s42.jpg)

![Unscaled 4K lens, grip, and tabletop crops, saved as JPEG previews.](/qualification/minimax-h3-image-native-2026-11-10/camera-4096-s42-details.jpg)

## Failures found during qualification

The first 2K image was a colored mosaic. The transformer predicts data-ward
velocity, but the native loop used `(sigma_next - sigma_current) * velocity`.
The corrected update adds `(sigma_current - sigma_next) * velocity` in FP32.
The initial trajectory fixture repeated the same sign error; its generator now
calls the pinned upstream scheduler directly, and a separate regression checks
that positive data-ward velocity increases the sample as sigma falls. See the
[pinned reference scheduler](https://github.com/huggingface/diffusers/blob/f598e6e6a99528858b962c1399c45b59a6c25484/src/diffusers/schedulers/scheduling_minimax_h3.py).

The first 4K attempt encountered a Metal GPU command-buffer timeout during
the first denoising pass, after 179.60 seconds, at a 70.70 GiB peak footprint.
The large-sequence path now uses projection chunks of at most 4096 rows and
attention chunks of at most 640 queries and eight heads, preserving attention
to every key. One real block and all 50 real transformer layers have completed
probes at the 4K token count with this schedule.
The subsequent full six-step 4K generation completed successfully.

Auxiliary real-weight probes compared the VAE against the pinned PyTorch
reference: CPU RGB maximum absolute error was 0.00000441. Exact small FP32
and mixed-BF16 numerical fixtures pass on CPU and Metal with
`MLX_ENABLE_TF32=0`; MLX's default M5 TF32 matmuls have larger rounding error.
The image generation runs use the normal MLX precision defaults.

## Prompt and reproduction

The exact prompt was:

> A high-resolution photorealistic studio product photograph of a single vintage brass rangefinder camera on a dark walnut tabletop. The camera is viewed from a three-quarter front angle and has one round black glass lens, a textured black leather grip, a silver shutter button, and engraved mechanical details. A small cobalt-blue ceramic cup sits behind the camera on the left. Soft daylight enters from the right, casting consistent gentle shadows to the left. A pale blue seamless background is out of focus. Show crisp brushed-metal grain, subtle fingerprints and realistic wood pores, with natural reflections and no artificial plastic smoothing. Balanced composition, shallow depth of field, physically plausible proportions, no illustration.

After reviewing and accepting the upstream license, install the model and run
the prompt above with these settings:

```bash
mere.run model pull image-minimax-h3 --accept-model-license
mere.run image generate --model image-minimax-h3 \
  --prompt "$PROMPT" --width 2048 --height 2048 \
  --steps 6 --cfg 1 --seed 42 --output camera-2048.png
```

For retained process measurements, use
`scripts/validation/minimax-h3-image-native-run.py` with a release binary, an
installed model directory, and a fresh case ID. It retains preflight output,
progress events, logs, source and output hashes, and image statistics.

## Validation boundary

`./scripts/check.sh` passed after the sampler and GPU scheduling fixes. The
five independent numerical tests also passed directly on CPU and Metal.
`pnpm docs:build` validates this report and its published previews.

This evidence covers one text-to-image prompt and seed at two sizes. Reference
editing, typography accuracy, additional seeds, other aspect ratios, and other hardware
remain unqualified. Native MLX random streams differ from PyTorch CPU streams;
the same integer seed does not imply identical images across runtimes.

These measurements precede native Q8 support and the BF16 rotary precision
correction. The [Q8 follow-up](minimax-h3-image-q8-2026-11-10.md) records the
separate export, subsequent optimizations, and cold/warm measurements. This
page preserves the original baseline and its artifacts.
