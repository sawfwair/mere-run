# Image Train LoRA

## Purpose

Train a local text-to-image LoRA adapter. Use this when you have a small image
and caption dataset and want an adapter that can be loaded with
`mere.run image generate --lora`.

## Required Models

- Train on `image-krea2-raw`.
- Preview and run the adapter on `image-krea2-turbo`.

## Install And Check

```bash
mere.run model pull image-krea2-raw
mere.run model pull image-krea2-turbo
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

Captions should be concrete and visual. For style training, include subject,
medium, lighting, composition, and any repeatable style tokens you plan to use
at inference time.

## Parameters

- `--data`, `-d`: dataset directory.
- `--output`, `-o`: output `.safetensors` adapter path.
- `--model`, `-m`: Raw/base model id or local model root. Defaults to `image-krea2-raw`.
- `--width`, `-W`, `--height`, `-H`: fixed training resolution, divisible by 16.
- `--training-steps`, `--steps`: optimizer steps.
- `--batch-size`: usually `1` on local Apple Silicon.
- `--learning-rate`, `--lr`: start around `0.0001`.
- `--rank`: LoRA rank. Start with `16`; lower for smaller adapters.
- `--alpha`: LoRA alpha. Defaults to rank.
- `--max-text-length`: prompt token budget.
- `--scheduler-steps`: FlowMatch training timestep count.
- `--caption-dropout`: probability of empty prompt conditioning.
- `--lite`: train only attention Q/V layers for lower memory use.
- `--exclude-preview-images`: skip `preview*` images in the dataset folder.
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

The command writes the adapter plus sidecar training artifacts beside it:

- `*.safetensors`: LoRA weights and optimizer state.
- `*.manifest.json`: training manifest.
- `*.checkpoint.json`: final checkpoint state.
- `run.json`: run manifest.
- `*.loss.csv` and `*.loss.html`: loss history.
- `*.zip`: portable bundle when archive creation succeeds.

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
- LoRA has no visible effect: confirm you are generating with `image-krea2-turbo` and the exact `--lora` path.
- Out of memory: use `--lite`, lower resolution, lower rank, or fewer concurrent apps.
- Dataset rejected: check every image has a same-stem `.txt` caption.

## Sources

- https://huggingface.co/krea/Krea-2-Raw
- https://huggingface.co/krea/Krea-2-Turbo
- https://www.krea.ai/krea-2-open-source
