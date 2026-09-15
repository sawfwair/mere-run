# Marigold V2 component diagnostics

Use the installed `vision-depth-marigold-v2` checkpoint and an upstream example
image to compare the native VAE and first transformer block with Diffusers
0.38.0. The scripts pin their Python dependencies and run the reference on MPS.
They resize the image to 256 by 176 pixels so both implementations receive the
same tensor. They do not run the full CUDA NF4 reference or measure depth accuracy.

```bash
export MERERUN_TEST_MARIGOLD_ROOT="$HOME/Library/Application Support/MereRun/models/vision-depth-marigold-v2"

uv run scripts/reference-parity/export_marigold_v2_vae_fixture.py \
  --root "$MERERUN_TEST_MARIGOLD_ROOT" --image /path/to/church.jpg \
  --output /tmp/marigold-vae --dtype float32

MERERUN_TEST_MARIGOLD_VAE_FIXTURE=/tmp/marigold-vae/reference-float32.safetensors \
MERERUN_TEST_MARIGOLD_VAE_OUTPUT=/tmp/marigold-vae/native-float32.safetensors \
swift test --filter MarigoldV2VAEParityTests

uv run scripts/reference-parity/export_marigold_v2_first_block_fixture.py \
  --root "$MERERUN_TEST_MARIGOLD_ROOT" \
  --fixture /tmp/marigold-vae/reference-float32.safetensors \
  --output /tmp/marigold-vae/reference-block.safetensors

MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_MARIGOLD_BLOCK_FIXTURE=/tmp/marigold-vae/reference-block.safetensors \
MERERUN_TEST_MARIGOLD_BLOCK_OUTPUT=/tmp/marigold-vae/native-block.safetensors \
swift test --filter MarigoldV2FirstBlockParityTests
```

For the VAE BF16 comparison, export with `--dtype bfloat16`, use the resulting
fixture, and set `MERERUN_TEST_MARIGOLD_VAE_BF16=1` on the Swift invocation.
The first-block comparison uses float32 arithmetic with the real base and
adapter tensors to isolate model structure from quantization differences.

## Reproduce the quantization experiment

`MarigoldV2TransformerDiagnosticTests` accepts the same model root and VAE
fixture. Set `MERERUN_TEST_MARIGOLD_QUANTIZATION` to one of:

- `affine`: quantize every eligible projection, reproducing the original defect.
- `affine-skip`: preserve the first image-modulation projection.
- `nf4`: reconstruct dense BF16 weights from the NF4 codebook and preserve the
  same projection. This is an effective-weight experiment, not a packed NF4
  kernel or a bit-identical bitsandbytes comparison.
- `none`: keep the base weights unquantized.

Set `MERERUN_TEST_MARIGOLD_TRANSFORMER_OUTPUT` to a `.safetensors` output path
and `MERERUN_TEST_MLX_DEVICE=gpu`. The test writes the packed input, transformer
prediction, stepped latent, decoded pixels, and a PNG preview. Its finiteness
check does not assert visual quality: the `affine` control intentionally emits
degraded output. NF4 and unquantized runs retain the full dense transformer and
need substantially more memory than the production quantized path.

For a resident end-to-end check on all four upstream examples, set
`MERERUN_TEST_MARIGOLD_EXAMPLES` to their directory and
`MERERUN_TEST_MARIGOLD_EXAMPLES_OUTPUT` to an output directory, then run
`swift test --filter MarigoldV2InstalledModelTests`. Inspect the resulting depth
maps as well as the artifact checks; a successful test is not an accuracy score.
