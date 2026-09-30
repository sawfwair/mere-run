# 4DAnyone reference fixtures

These fixtures execute the upstream Base graph on CPU with FP32 synthetic
weights. They contain no trained weights, source video, or motion assets.
The transformer has two blocks and 24 hidden channels. The pose encoder uses
its complete production channel configuration; its synthetic weights are
recreated from the recipe in `manifest.json`.

- `transformer.safetensors`: synthetic weights in upstream checkpoint layout.
- `reference.safetensors`: inputs, intermediate tensors, predictions, grouped
  denoising outputs, schedules, and pose encoder outputs.
- `checkpoint-schema.json`: all 1,182 released checkpoint names and shapes.
- `manifest.json`: source revision and hashes, artifact hashes, dimensions,
  routing expectations, and the pose weight recipe.
- `activations.safetensors` and `activations.json`: 16,384 BF16 inputs spanning
  -12 through 12, independent PyTorch GELU and SiLU outputs, and their receipt.
  This fixture contains no checkpoint weights. The native path evaluates the
  activations in FP32 before casting back; the maximum absolute tolerance is
  `2e-5` on both CPU and Metal.

To reproduce, clone `https://github.com/ant-research/4DAnyone` and check out
`8cd60c40d90882de07645cc435dcf24bc9b4fbd1`. Supply the safetensors header JSON
from `4danyone/model.safetensors` at `AntResearch/4DAnyone` revision
`4c80e87b805a5f8461cf339cdbe2fb4249e585aa`.

```bash
uv run --python 3.11 --with torch==2.8.0 --with numpy==1.26.4 \
  --with einops==0.8.1 --with safetensors==0.6.2 --with packaging==25.0 python \
  scripts/reference-parity/export_4danyone_fixture.py \
  --upstream /path/to/4DAnyone --checkpoint-header /path/to/model-header.json \
  --output Tests/MereRunCoreTests/Fixtures/FourDAnyone
swift test --filter FourDAnyone
```

The exporter rejects a different or modified upstream revision. Numerical
comparisons use maximum absolute tolerances of `3e-5` for transformer stages,
`5e-5` for the four-step denoising result, `2e-4` for the full pose encoder,
and `1e-7` for schedules. These are small CPU arithmetic comparisons; they do
not validate CUDA/BF16, trained outputs, VAE parity, memory use, or speed.

See [third-party notices](../../../../THIRD_PARTY_NOTICES.md) for Apache-2.0
source attribution and [the license text](../../../../licenses/4danyone-apache-2.0.txt).

To regenerate the activation fixture alongside the frozen operation cases,
use the dependencies listed in `export_4danyone_operations.py`:

```bash
python scripts/reference-parity/export_4danyone_operations.py \
  --checkpoint /path/to/model.safetensors --output /path/to/operations \
  --activation-fixture Tests/MereRunCoreTests/Fixtures/FourDAnyone/activations.safetensors
```
