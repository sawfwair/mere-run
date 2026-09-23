# FalconPerception

Vision grounding and perception runtime.

- `FalconPerceptionConfig.swift`: typed model configuration.
- `FalconPerceptionTokenizer.swift`: tokenizer compatibility boundary.
- `FalconPerceptionModel.swift`: native model layers.
- `FalconPerceptionGrounder.swift`: user-facing grounding pipeline.
- `FalconPerceptionProcessor.swift`: image and prompt preprocessing.

Keep tokenizer/config quirks isolated in boundary files and cover grounding
output contracts with focused tests.

## File and batch image resizing

The MediaImage preprocessing path used by CLI file inputs and batched grounding
uses RGB bicubic resampling at both resize stages. The separable filter,
anti-aliasing support, fixed-point coefficients, intermediate byte rounding,
and clipping follow Pillow 12.3.0. RGB values ignore alpha, matching conversion
to RGB before the reference resizes. Other model families keep their existing
shared resize behavior.

Eight generated fixtures compare RGB bytes for downscaling, upscaling,
single-axis changes, and unchanged dimensions. Regenerate them with the pinned
Pillow version and the verified Falcon processor source:

```bash
python scripts/reference-parity/export_falcon_resize_fixture.py \
  --reference /path/to/processing_falcon_perception.py \
  --output Tests/MereRunCoreTests/Fixtures/falcon-bicubic-resize.json
```

The optional full-frame test reads a declaration from
`MERERUN_FALCON_RESIZE_PARITY_CASE`. Each frame specifies `sourcePath`,
`referencePath`, and `outputPath`; it writes the native result and compares every
RGB byte. These tests establish resize-stage parity, not checkpoint accuracy,
all image decoder/color-management behavior, or the separate CGImage overload.
Dimension rounding policy remains a separate parity concern.

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

## Coordinate token selection

`FalconPerceptionCoordinateDecoder` selects coordinate bins before the selected
coordinate is embedded into the next token. It follows the generation policy in
upstream `modeling_falcon_perception.py` at revision
`54916b3dec58565fafc6d82eb3051fe7246ab666`:

- Keep every coordinate token in the current query's history, including
  coordinates that have no completed detection. Batch slots have separate histories.
- Select the first maximum on each axis. A repeat requires both normalized axes
  to differ from an earlier coordinate by strictly less than 0.01.
- Suppress both selected bins for a repeat and select again. Try at most 100
  candidates, retaining the final candidate even if it repeats.
- Preserve double-precision bin ratios in history before casting the chosen
  coordinate for the model's embedding. Float rounding can change the strict
  threshold decision.

This policy does not deduplicate final boxes or masks. Preprocessing, tensor
numerics, token limits, mask generation, and task-specific detection quality
require separate validation. The pure Swift policy tests do not load weights or
establish full model parity.

Direct and batched generation stop at either the model-configured EOS or the
`<|end_of_query|>` token resolved from the tokenizer vocabulary. Each batch slot
stops independently; remaining slots continue. The query token ID is not
hardcoded. Tokenizers without that marker retain model-EOS stopping behavior.
