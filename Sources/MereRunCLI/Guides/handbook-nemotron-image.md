# Nemotron Labs Image 30B-A3B (NVIDIA)

Use `image-nemotron-labs-30b-a3b-q6` to generate a PNG from a text prompt. The recommended 30.75 GB download includes the mixed 6-bit-expert, 8-bit-dense generator and the pixel decoder.

## Example to adapt

For non-commercial research or evaluation, review NVIDIA's license before
accepting it and installing the checkpoint. [NVIDIA license](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B/blob/d7c2ba6a59a384080f0bdd5b1e846da03804cdd1/LICENSE)

```bash
mere.run model pull image-nemotron-labs-30b-a3b-q6 --accept-model-license
mere.run image generate --model image-nemotron-labs-30b-a3b-q6 \
  --prompt "Two orange kayaks on a lake, viewed from above" \
  --width 1184 --height 880 --seed 42 --output ./kayaks.png
```

For the managed Q8 checkpoint, use `image-nemotron-labs-30b-a3b-q8` in both commands.
It downloads about 40.3 GB including the decoder and uses the same NVIDIA license.
Use `image-nemotron-labs-30b-a3b` for the original FP32 download (about 133 GB) and BF16 inference.

## Controls and variants

Choose 1024×1024, 1184×880, 1248×832, or 880×1184 for a recommended canvas.
Other sizes must be multiples of 16 and fit the 4,352-token budget. Each
16×16 patch uses one code and each row begins with a marker. Sampling uses
temperature 0.9. If the generated grid has the wrong shape, try a different
seed or a recommended size. Use `--max-sequence-length` for longer prompts.

## Sources and validation

The runtime follows the standalone raster generation protocol at revision
`d7c2ba6a59a384080f0bdd5b1e846da03804cdd1`. Synthetic tests cover expert
loading, grid framing, and decoding to RGB. Q6 peaked at 26.90 GiB in six native 1024 × 1024 attempts on a 128 GiB M5 Max. Fit on a 36 GiB Mac is estimated. [Model card](https://huggingface.co/nvidia/Nemotron-Labs-Image-30B-A3B) · [VisualCreation paper](https://huggingface.co/papers/2611.31358)
