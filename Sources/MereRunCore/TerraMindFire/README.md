# TerraMind Fire Source Map

This module owns the immutable source and conversion contract for IBM and
ESA's TerraMind ImpactMesh fire checkpoint. The released Fire and Flood
checkpoints use the same base encoder, temporal wrapper, feature pyramid,
UNet decoder, and two-class head graph, so Fire intentionally reuses the
audited native `TerraMindFloodModel` implementation with separately pinned
weights and provenance.

The public boundary is `mere.run geo fire`. It accepts normalized `S2L2A`,
`S1RTC`, and `DEM` arrays shaped `[batch, channels, 4, 256, 256]` and emits
float32 logits. Raster discovery, normalization policy, reprojection, tiling,
stitching, georeferencing, and evidence review remain provider responsibilities.

Native code must never interpret the upstream Lightning/pickle checkpoint.
Use `scripts/convert-terramind-fire-mlx.py` to produce the checksum-pinned,
float32 safetensors package, and
`scripts/validate-terramind-fire-reference.py` to compare native logits and
candidate masks with the official TerraTorch task.
