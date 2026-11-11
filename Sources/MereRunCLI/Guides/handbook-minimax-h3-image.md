# MiniMax-H3-Image (MiniMax)

`image-minimax-h3` runs the original BF16 checkpoint with native Swift and
MLX. It generates images from text and edits up to 16 ordered references.
The sampler requires six model evaluations and CFG 1. Each step uses its
own PDD output head.

## Install

Review the MiniMax-H3-Image Community License before pulling the model. It
excludes the United States, European Union, United Kingdom, and Republic of
Korea. Revenue above USD 1 million or hosted services require separate
authorization from MiniMax. The weights have a separate license from mere.run.

```bash
mere.run model pull image-minimax-h3 --accept-model-license
```

The pinned snapshot is
`MiniMaxAI/MiniMax-H3-Image@492fd8c4eca666f1085f57e30738c1e0d86b3439`.
The selected files total about 135 GB, including the Qwen3-VL text encoder.
The installer omits the duplicate Diffusers-formatted transformer and VAE.
Catalog memory estimates are 96 GB minimum and 128 GB recommended. A real
checkpoint run on a 128 GiB M5 Max measured peak process footprints of 67.35
GiB at 2K and 72.12 GiB at 4K; these shared-host measurements do not establish
the minimum RAM requirement.

## Example to adapt

```bash
mere.run image generate --model image-minimax-h3 \
  --prompt "A detailed studio photograph of a brass camera on a walnut desk, with soft window light and a pale blue background." \
  --width 4096 --height 4096 --seed 42 --output ./camera.png
```

Use a detailed English prompt. The runtime passes the prompt directly to
the text encoder, truncated to 4096 tokens. Prompt enhancement is optional
and is not part of this model's native inference path.

Images smaller than 4 MP are generated on a larger canvas with the same
aspect ratio, then resized to the requested dimensions with Lanczos.
Canvas sides round to multiples of 32. Native requests accept 32–8192 pixel
sides and up to 20 MP. Output must be PNG.

## Native Q8 export

Create a separate Q8 checkpoint from an installed original BF16 model:

```bash
mere.run model optimize image-minimax-h3 --output ./image-minimax-h3-q8 --json
mere.run image generate --model ./image-minimax-h3-q8 \
  --prompt "A detailed studio photograph of a brass camera on a walnut desk." \
  --width 4096 --height 4096 --seed 42 --output ./camera-q8.png
```

The exporter uses native MLX affine 8-bit/group-64 linear weights for the
transformer and text encoder. It retains the released precision for the VAE,
FP32 projections, and vision projections whose widths do not align to 64.
Exact timestep tables replace the large schedule-only AdaLN weights. Output
must be a new directory; the source checkpoint is preserved. The exported
checkpoint includes its license, configs, tokenizer, source identity, and
weight hashes. Q8 can change image details and needs its own quality review.

The qualified export occupies 52.04 GB, including its text encoder and original
VAE. This reduction combines Q8, cached AdaLN outputs, and encoder trimming.
The catalog ID `image-minimax-h3-q8` resolves an export in the model store.
For exports elsewhere, pass the directory path to `--model`.
A prepared source build restores affine Q8 matrix acceleration on the tested
M5 Max. See the M5 acceleration report in `docs/benchmarks/` for native 4K
latency, peak memory, and example images.

For contributor timing, compare the first request with repeated requests in
one native session:

```bash
mere.run model benchmark h3-image --model-root ./image-minimax-h3-q8 \
  --prompt "A detailed studio photograph of a brass camera on a walnut desk." \
  --size 4096 --runs 3 --output ./h3-warm-benchmark --json
```

The benchmark retains the transformer and VAE and caches one text-only prompt.
It records each PNG hash and a warm median, including decoding and PNG output.
A new prompt needs conditioner execution. Ordinary `image generate` requests
load and release components for each invocation.

Use `--attention-heads`, `--attention-queries`, and `--vae-tile-batch` to profile
batch sizes. The benchmark defaults to 28 heads, 1024 queries, and 32 decode
tiles, the configuration qualified on the M5 Max. `--vae-tile-batch 1` runs
decode tiles sequentially. `--resident-bf16` materializes the Q8 transformer
as BF16 to compare dense compute. This mode retains more weights in memory
and rounds the reconstructed weights to BF16.

## Controls and variants

```bash
mere.run image generate --model image-minimax-h3 \
  --ref-image ./room.png --ref-image ./chair.png \
  --prompt "Place the chair from <Picture 2> beside the window in <Picture 1>, preserving the room's perspective and lighting." \
  --width 4096 --height 4096 --seed 42 --output ./edited.png
```

References follow flag order. An `--input` image comes first and counts
toward the 16-image limit. References condition the Qwen vision encoder and
the image VAE. Their latent rows remain fixed throughout denoising.

Negative prompts, custom sigma schedules, `--sigma-shift`, edit strength,
and LoRA adapters are unsupported. Leave steps and guidance at their model
defaults, or pass `--steps 6 --cfg 1`.

## Sources and validation

The [official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
describes the released architecture.

Small upstream PyTorch fixtures test FP32 and mixed-BF16 transformer
predictions, all six output heads, complete six-step trajectories, and the
2D VAE decoder. Real-checkpoint camera generations at 2048 and 4096 pixels
originally completed in 213 and 499 seconds on a 128 GiB M5 Max, with coherent images and
invented fine markings. Reference editing and broader quality remain
unqualified. The qualification report in `docs/benchmarks/` retains previews,
process measurements, and output hashes. Native MLX random streams are
repeatable but do not reproduce PyTorch CPU noise from the same integer seed.
