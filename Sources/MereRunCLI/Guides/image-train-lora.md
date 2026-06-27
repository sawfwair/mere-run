# Image Train LoRA

## Purpose

Train a local text-to-image LoRA adapter. Use this when you have a small image
and caption dataset and want an adapter that can be loaded with
`mere.run image generate --lora`.

## Required Models

- Train on `image-krea2-raw`.
- Preview and run the adapter on `image-krea2-turbo`.
- For FLUX.2 Klein LoRAs, train on a Klein base model such as
  `image-klein-base-9b`, then run the adapter on distilled Klein models such as
  `image-klein-9b` for practical inference. If distilled output looks weak,
  compare with base/checkpoint previews and the exact sampling recipe before
  judging the training run.

## Install And Check

```bash
mere.run model pull image-krea2-raw
mere.run model pull image-krea2-turbo
mere.run model pull image-klein-base-9b
mere.run image train-lora --help
```

## Dataset Layout

Use image files with matching UTF-8 `.txt` captions:

```text
dataset/
  001.png
  001.txt
  002.jpg
  002.txt
```

Captions should be concrete and visual. For style training, use a distinct
trigger token in every caption and describe only the image content: subject,
medium, lighting, and composition. Do not describe the style or concept you want
the adapter to learn.

## Parameters

- `--data`, `-d`: dataset directory.
- `--output`, `-o`: output `.safetensors` adapter path.
- `--model`, `-m`: Raw/base model id or local model root. Defaults to `image-krea2-raw`.
- `--width`, `-W`, `--height`, `-H`: fixed training resolution, divisible by 16.
- `--training-steps`, `--steps`: optimizer steps.
- `--batch-size`: usually `1` on local Apple Silicon.
- `--learning-rate`, `--lr`: start around `0.0001`.
- `--rank`: LoRA rank. Start with `16` for local experiments; high-capacity
  Klein style runs commonly need larger ranks such as `128`.
- `--alpha`: LoRA alpha. Defaults to rank.
- `--max-text-length`: prompt token budget.
- `--scheduler-steps`: FlowMatch training timestep count.
- `--caption-dropout`: probability of empty prompt conditioning.
- `--lite`: train only attention Q/V layers for lower memory use.
- `--exclude-preview-images`: skip `preview*` images in the dataset folder.
- `--checkpoint-interval`: for FLUX.2 Klein, save intermediate adapters every N
  steps so you can compare visual quality before the final checkpoint.
- `--max-resolution`: for FLUX.2 Klein, preserve each source image aspect ratio
  and bucket it up to this maximum side length. This cannot be combined with
  `--progressive`.
- `--low-ram`: for FLUX.2 Klein, cache encoded latents on disk to reduce peak
  memory.
- `--no-compile`: for FLUX.2 Klein, skip the compiled train-step graph when
  graph compilation spikes GPU memory.
- `--gradient-checkpointing`: for FLUX.2 Klein, recompute transformer block
  activations during backprop to reduce peak memory for high-rank/high-res runs.
- `--sample-interval`, `--sample-prompt`, `--sample-model`, `--sample-steps`,
  `--sample-cfg`, `--sample-lora-scale`, `--sample-seed`: for FLUX.2 Klein,
  generate checkpoint previews during training. The default preview model is
  `image-klein-9b`.
- `--lora-rank-preset flux2-style-128`: expand FLUX.2 transformer targets to
  rank `128` with alpha `64`.
- `--lora-target-mode`: choose `suffix` for the default FLUX.2 allowlist or
  `transformer-linear-walk` to train every transformer Linear/QuantizedLinear
  layer.
- `--lora-target-ranks`: custom FLUX.2 target suffix ranks, for example
  `.attn.to_q=128,.ff.linear_in=64`.
- Klein-only recipe controls: `--timestep-sampling`, `--timestep-loss-weighting`,
  `--loss-weighting`, `--timestep-low`, `--timestep-high`,
  `--lr-warmup-steps`, `--no-cosine-scheduler`, `--lr-min-factor`, and
  `--adam-weight-decay`.
- `--quiet`, `-q`: print only the final adapter path.

## Train

```bash
mere.run image train-lora \
  --data ./dataset \
  --output ./my-krea2-style.safetensors \
  --training-steps 1000 \
  --rank 16 \
  --lite
```

For high-capacity Klein style LoRAs, use the base model, preserve the target
aspect ratio, keep the trigger token in the captions and prompts, and save
intermediate checkpoints. `--timestep-sampling shift` uses dynamic FlowMatch
sigma shifting based on each resolution bucket:

```bash
mere.run image train-lora \
  --model image-klein-base-9b \
  --data ./dataset \
  --output ./my-klein-style.safetensors \
  --training-steps 1500 \
  --learning-rate 0.000095 \
  --max-resolution 1536 \
  --lora-rank-preset flux2-style-128 \
  --timestep-sampling shift \
  --timestep-loss-weighting weighted \
  --adam-weight-decay 0.00015 \
  --low-ram \
  --gradient-checkpointing \
  --no-compile \
  --checkpoint-interval 150 \
  --sample-interval 150 \
  --sample-prompt "TRIGGER_TOKEN portrait in a diner" \
  --sample-lora-scale 1.0
```

The command writes the adapter plus sidecar training artifacts beside it:

- `*.safetensors`: LoRA weights and optimizer state.
- `*.manifest.json`: training manifest.
- `*.checkpoint.json`: final checkpoint state.
- `run.json`: run manifest.
- `*.loss.csv` and `*.loss.html`: loss history.
- `*.zip`: portable bundle when archive creation succeeds.
- `checkpoints/*-stepN.safetensors`: intermediate Klein checkpoints when
  `--checkpoint-interval` is set.
- `samples/*-stepN-sample.png`: intermediate Klein previews when
  `--sample-interval` is set.

## Preview

```bash
mere.run image generate \
  --model image-krea2-turbo \
  --prompt "a portrait in my trained style" \
  --lora ./my-krea2-style.safetensors \
  --lora-scale 1.0 \
  --output ./preview.png
```

## Troubleshooting

- Missing Raw model: run `mere.run model pull image-krea2-raw`.
- LoRA has no visible effect: confirm you are generating with the exact `--lora`
  path and that the prompt includes the trained trigger token. For Klein LoRAs,
  train on `image-klein-base-9b` and run on distilled `image-klein-9b`, but
  compare base/checkpoint previews when diagnosing a weak adapter.
- Klein LoRA loses structure: preserve the source aspect ratio, avoid `--lite`
  unless memory requires it, use `--max-resolution` or exact source dimensions,
  use a higher-capacity rank for broad styles, and compare intermediate
  checkpoints instead of assuming the final step is best.
- Out of memory: use `--low-ram --gradient-checkpointing --no-compile`, lower
  resolution, lower rank, `--lite`, or fewer concurrent apps.
- Dataset rejected: check every image has a same-stem `.txt` caption.

## Sources

- https://huggingface.co/krea/Krea-2-Raw
- https://huggingface.co/krea/Krea-2-Turbo
- https://www.krea.ai/krea-2-open-source
- https://docs.bfl.ml/flux_2/flux2_klein_training
- https://docs.bfl.ml/flux_2/flux2_klein_training_example
