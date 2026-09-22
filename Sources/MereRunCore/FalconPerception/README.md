# FalconPerception

Vision grounding and perception runtime.

- `FalconPerceptionConfig.swift`: typed model configuration.
- `FalconPerceptionTokenizer.swift`: tokenizer compatibility boundary.
- `FalconPerceptionModel.swift`: native model layers.
- `FalconPerceptionGrounder.swift`: user-facing grounding pipeline.
- `FalconPerceptionProcessor.swift`: image and prompt preprocessing.

Keep tokenizer/config quirks isolated in boundary files and cover grounding
output contracts with focused tests.


## Functional normalization

Attention input, query/key, and feed-forward input normalization use the
Float32 accumulation epsilon (`Float.ulpOfOne`), matching functional
`torch.nn.functional.rms_norm` in the pinned upstream implementation at revision
`54916b3dec58565fafc6d82eb3051fe7246ab666`. The final learned normalization keeps
the configured `norm_eps`; that value is not shared with the functional norms.

The regression fixture records PyTorch 2.13 reference values for small-magnitude
inputs, pre-attention projections, the interleaved feed-forward gate, and a
separately configured final normalization. It validates these numerical stages,
not full checkpoint parity or detection accuracy.

To reproduce the fixture, install the generator's pinned Python dependencies in
an isolated environment and run:

```bash
python scripts/reference-parity/export_falcon_normalization_fixture.py \
  --reference /path/to/pinned/modeling_falcon_perception.py
```

The generator verifies the reference source hash before extracting the two
attention methods. Feed-forward values use the equivalent PyTorch interleaved
gate equation; they do not validate the upstream Triton kernel.
