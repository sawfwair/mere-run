# TerraMind Flood Source Map

This module owns the native, inference-only Swift/MLX implementation of the
pinned IBM/ESA TerraMind ImpactMesh flood checkpoint.

- `TerraMindFloodModel.swift` implements the exact multimodal encoder, selected
  feature pyramid, temporal UNet decoder, and two-class segmentation head.
- `TerraMindFloodResources.swift` validates the immutable upstream source pin
  and the deterministic float32 safetensors conversion before loading weights.

The public CLI boundary is `mere.run geo flood`. It accepts already normalized
and tiled `S2L2A`, `S1RTC`, and `DEM` arrays and emits logits. Raster discovery,
reprojection, tiling policy, reconstruction, COG export, and evidence
classification belong to a geospatial provider such as `mere-geo-tools`, not
this model module.

Keep the runtime float32-only unless a new real-raster parity gate proves an
alternative precision preserves the decision boundary. Native code must never
interpret the upstream Lightning/pickle checkpoint; use
`scripts/convert-terramind-flood-mlx.py` to produce the pinned safetensors
package.
