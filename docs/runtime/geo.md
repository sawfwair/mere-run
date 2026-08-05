# Geospatial Runtime

`mere.run geo` is the native inference boundary for Earth-observation models.
Acquisition, reprojection, raster tiling, COG export, and analyst evidence can
remain in geospatial workflow providers while the neural forward pass runs
locally through Swift, MLX, and Apple Metal.

## TerraMind Flood

`mere.run geo flood` runs IBM and ESA's ImpactMesh flood checkpoint with the
native TerraMind encoder, feature pyramid, temporal UNet decoder, and binary
head. The command accepts normalized, pre-tiled safetensors and emits float32
logits:

```bash
mere.run geo flood flood-input.safetensors \
  --output flood-logits.safetensors \
  --model /path/to/converted-terramind-flood \
  --json
```

The input must contain `S2L2A`, `S1RTC`, and `DEM` arrays shaped
`[batch, channels, 4, 256, 256]`. This low-level tensor contract keeps raster
policy outside the model runtime. The `mere-geo-tools` graph provider prepares
those arrays, reconstructs the tiled output, and writes georeferenced
candidate-only artifacts.

## Model conversion

The managed model id is `vision-flood-terramind-base`. Its source is pinned to
the official `ibm-esa-geospatial/TerraMind-base-Flood` revision and remains a
non-executable Lightning checkpoint until explicit conversion:

```bash
python3 scripts/convert-terramind-flood-mlx.py \
  --checkpoint /path/to/TerraMind_v1_base_ImpactMesh_flood.pt \
  --configuration /path/to/terramind_v1_base_impactmesh_flood.yaml \
  --output /path/to/converted-terramind-flood
```

Conversion is float32-only. FP32 reproduces the reference Helene candidate mask
exactly; FP16 changed the decision boundary and is rejected by the runtime.
The converted package is checksum-pinned and contains safetensors only, so
native inference never interprets the upstream pickle checkpoint.

Flood outputs are candidates, not authoritative findings. Promotion requires
independent corroboration and local validation.
