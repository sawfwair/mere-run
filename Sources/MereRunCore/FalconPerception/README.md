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
