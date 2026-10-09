# Released D1 checkpoint references

These inputs and original-code answers cover text/image decisions on D1-3B
(BF16) and text/image/audio decisions on D1 omni-600M (FP32). Each request has
one noul, choice, and score. They establish agreement on these cases, not
benchmark accuracy or comprehensive numerical equivalence.

The 224×224 solid red PNG was generated locally. The 16 kHz mono PCM WAV was
generated with the macOS Samantha voice saying: “The parcel arrived on time,
but the glass inside was broken.” No personal data or checkpoint weights are
included. Omni's reference also misclassifies the damaged item as a book and
answers the arrival question negatively; this fixture preserves that behavior
rather than asserting semantic correctness.

Original checkpoint revisions and weight SHA-256:

- `LiquidAI/d1-3B@da1fe36a861f24690f27f622dca1d8688503d113`:
  `50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c`
- `LiquidAI/d1-omni-600M@414f8d6438174f5b2133a9c21a478fc42625e308`:
  `0713bb05270c2685ad106522f4092bceeeb3a93cf79b401f399a712296c911e1`

References use original checkpoint Python source with PyTorch 2.14.1,
Transformers 5.19.0, torchvision 0.29.1, Pillow 12.3.0 and soundfile 0.14.0 on
CPU. No native output contributes to the expected values. Export each family:

```bash
python scripts/fixtures/export-d1-release-reference.py \
  --checkpoints /path/to/original-checkpoints --family omni \
  --output Tests/MereRunCoreTests/Fixtures/D1/released
python scripts/fixtures/export-d1-release-reference.py \
  --checkpoints /path/to/original-checkpoints --family causal \
  --output Tests/MereRunCoreTests/Fixtures/D1/released
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_TEST_D1_RELEASED_CHECKPOINTS=/path/to/original-checkpoints \
  swift test --filter D1ReleasedCheckpointTests
```

The checkpoint root contains `causal/` and `omni/`. Native regression execution
requires only the original configs, tokenizers, and safetensors; Python is used
only to regenerate the independent references. Probability tolerances are
0.0001 for FP32 omni and 0.02 for BF16 D1-3B. Choice identity and token usage
must agree exactly.
