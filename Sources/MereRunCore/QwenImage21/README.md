# Qwen Image 2.1 execution

This directory owns the image-generation adapter for `image-qwen-21`.

- `QwenImage21Resources` pins the official checkpoint and loads indexed tensors.
- `QwenImage21Conditioner` uses the native Qwen3-VL encoder, deepstack vision
  features, and the final decoder activation before RMS normalization.
- `QwenImage21ImageIO` preserves straight RGBA, resizes references with Lanczos,
  and composites a separate vision-encoder copy over white.
- `QwenImage21Generator` stages encoder, VAE, and transformer loading, uses
  dynamic flow shifting, and writes RGBA PNG output.

The existing image plan, API operation, managed catalog, and Studio model
inventory expose this runtime. Reference images retain their order. Each image
uses a 1024-squared reference area, matching the pinned Diffusers pipeline.
Output dimensions are independent and must be multiples of 32.

Caches belong to one generation. Classifier-free guidance uses separate positive
and negative prefix caches. The seed controls MLX noise; identical seeds do not
imply identical random arrays across MLX and PyTorch.

See the [trained-checkpoint qualification report](../../../docs/benchmarks/qwen-image-21-native-qualification-2026-09-20.md)
for bounded M4 Max results and numerical and visual limitations. The local gate
and small reference fixtures establish narrower code contracts.

Regenerate numerical fixtures with `scripts/validation/qwen-image-21-reference.py`
using the dependencies and revisions documented in that script. Refresh checkpoint
schemas with `scripts/validation/qwen-image-21-headers.py`; it reads only pinned
configuration files and safetensors headers. Neither tool is a runtime dependency.

## Trained-checkpoint qualification

Install the pinned model after accepting its license. Prepare tokenizer cases
with `scripts/validation/qwen-image-21-trained-parity.py --prepare-tokenizer`,
passing `--model` as the installed checkpoint directory and `--output` as a fresh
evidence directory. Set `QWEN21_MODEL_ROOT` and `QWEN21_QUALIFICATION_ROOT` to
those same directories, then run:

```bash
MERERUN_TEST_MLX_DEVICE=gpu swift test --filter QwenImage21QualificationTests
```

The opt-in tests check tokenizer parity and export finite trained-weight text,
vision-conditioned text, transformer/cache, and VAE tensors. Run the Python
script again without `--prepare-tokenizer` to compare those exports against the
pinned reference libraries. Comparisons use common inputs and BF16 weights;
the predefined component tolerance is 5% relative L2. These small spatial inputs
do not qualify image quality or full-resolution memory use.

Use `scripts/validation/qwen-image-21-native-run.py` with a JSON case plan for
separate CLI runs. It preserves preflights, commands, receipts, output hashes,
alpha statistics, elapsed time, maximum resident size, and OS peak memory footprint. Visual inspection
and repeatability comparisons remain separate checks.

The checked-in `scripts/validation/qwen-image-21-qualification-plan.json` defines
CLI cases, including reference dependencies. Run them in order into one fresh
output directory. The two-step cases test execution rather than image quality.

For a full-precision transformer diagnostic, set `QWEN21_TRANSFORMER_FP32=1`
when running the native tests. Compare `native-transformer-fp32.safetensors`
with `--native-file`, `--transformer-only`, and `--tolerance 0.0001`. For a
512-pixel target grid, set `QWEN21_TARGET_GRID=32` and compare
`native-components-grid32.safetensors`. Use distinct `--report-name` values to
preserve each comparison. These diagnostic overrides affect only the opt-in
tests, not the public generation runtime.
