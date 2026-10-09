# Qwen Image 2.1 (Qwen)

Use `image-qwen-21` for native Swift/MLX text-to-image generation, editing with
up to 10 ordered references, and transparent RGBA output. Pull the model before
running inference. The Qwen Research License restricts use to non-commercial
research and evaluation; commercial use requires a separate license from Qwen.
Review the license before acknowledging its terms during the managed pull.

## Example to adapt

```bash
mere.run model pull image-qwen-21 --accept-license-terms
mere.run image generate --model image-qwen-21 \
  --prompt 'A ceramic teapot on a wooden table' \
  --width 1024 --height 1024 --steps 40 --seed 42 --output teapot.png
```

To edit an image, add `--input photo.png`. To compose multiple images, append
ordered `--ref-image` arguments. To request transparency, describe an RGBA image
with a transparent background in the prompt. Save as PNG to preserve alpha.

## Controls and variants

### Turbo

`image-qwen-21-turbo` installs the official Turbo checkpoint in a separate model
directory. It uses the same native Swift/MLX layers as the base model and reads
the original BF16 safetensors directly. No Python setup or MLX weight conversion
is required. Pulling Turbo does not replace an existing `image-qwen-21` install.

```bash
mere.run model pull image-qwen-21-turbo --accept-license-terms
mere.run image generate --model image-qwen-21-turbo \
  --prompt 'A ceramic teapot on a wooden table' \
  --steps 8 --seed 42 --output teapot-turbo.png
```

Turbo defaults to eight steps and CFG 1. The checkpoint's `sample_sigmas` grid in
`model_index.json` passes through its static flow scheduler (shift 1, no terminal
stretch), followed by a terminal zero. This differs from running the base model
with `--steps 8`. The native Turbo recipe rejects other step counts and
`--sigma-shift`; custom `--sigmas` remain unsupported. References, PNG alpha,
and optional negative-prompt guidance use the shared native runtime.

Turbo pins `Qwen/Qwen-Image-2.1-Turbo@d65dbc9a7e8f6b5479e33dee6030eaab2a906509`
and downloads approximately 32.45 GB. It carries the Qwen Research License.
Its checkpoint hashes differ from the base model, so existing base weights are
not substituted for Turbo components. The native loader also accepts local mixed
MLX checkpoints with affine Q4/Q8 group-64 weights. The separately published
[Sawfwair mixed quantization](https://huggingface.co/Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit)
uses Q4 transformer blocks, Q8 encoder linears, and BF16 remaining layers. Pass its
downloaded directory to `--model`; packed weights execute without expansion or a
Python inference process.

The saved-schedule behavior follows
[Diffusers PR #14950](https://github.com/huggingface/diffusers/pull/14950), pinned
to `da1d3829cf08d4f329b526d89e17cc035c049d8d`. Local tests check the saved grid,
configuration decoding, original BF16 tensor schemas, CLI/API defaults, and
base-model regression behavior. Dense BF16 Turbo checkpoint inference and
numerical parity remain unqualified. The mixed checkpoint separately passed
bounded native 512/1024 generation, deterministic replay, text-rendering, and
one-reference editing cases; broader quality, 2048-pixel output, multiple
references, and transparency remain unqualified. The base model's qualification
below applies only to that checkpoint.

### Base model

- The default schedule uses 40 steps and guidance 1. Dimensions must be positive
  multiples of 32. The minimum supported step count is two.
- `--cfg` greater than 1 enables classifier-free guidance when you supply
  `--negative-prompt`. Positive and negative conditions have separate caches.
- `--sigma-shift` overrides the flow scheduler's dynamic shift parameter.
- References use a 1024-squared area with their aspect ratios preserved,
  independently of output dimensions. Their alpha reaches the VAE; the vision
  encoder receives a copy composited over white.
- `--strength`, LoRA adapters, and custom `--sigmas` lists are unsupported.
- The runtime uses original dense safetensors. It does not invoke Python,
  Diffusers, or a hosted API. Prompt rewriting is separate from image inference;
  this model does not automatically load Qwen's optional prompt rewriters.
- Seeds select MLX noise. Equal seeds across PyTorch and MLX do not guarantee
  equal initial noise or pixel output.

## Sources and validation

The checkpoint is pinned to
`Qwen/Qwen-Image-2.1@b3179ad355be050328e483a9dfdd9e60cd62adfa`.
The native transformer and VAE follow the
[Diffusers reference](https://github.com/huggingface/diffusers/pull/14804), pinned
to `8d3c30bfda9b511c00992f40cff4170a5502814d`.
See the [official release](https://github.com/QwenLM/Qwen-Image-2.1) and
[checkpoint license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE).

Bounded trained-checkpoint checks on an M4 Max with 128 GiB cover generation,
editing, RGBA output, and exact same-seed replay. The 1024-square, 40-step
generation took about six minutes. This is one-host evidence, not a guarantee
for other prompts or smaller machines.

A tiny image-conditioned BF16 numerical probe exceeds the 5% reference
tolerance; full-precision and 512-pixel transformer probes pass their respective
tolerances. Transparent output can contain edge speckles. An eight-step CFG 3
probe overexposed and distorted the subject. Consult
`docs/benchmarks/qwen-image-21-native-qualification-2026-09-20.md` in the source
repository for retained results and limitations.
