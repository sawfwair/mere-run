# Geospatial runtime

`mere.run geo` is a local-first inference boundary for Earth-observation and
humanitarian workflows. Any
workflow can prepare the documented tensors, run the native model locally, and
carry the resulting candidates or embeddings into its own evidence process.

Acquisition, cloud masking, reprojection, tiling, georeferencing, source
citations, and analyst review remain workflow responsibilities. `mere.run` owns
immutable model provenance, native Swift/MLX execution, hardware-aware model
selection, and machine-readable outputs.

## Model families

| Command | Capability | Managed tiers | Automatic tier policy |
| --- | --- | --- | --- |
| `geo flood` | Two-class flood logits from four S2/S1/DEM observations | TerraMind Base | Base |
| `geo fire` | Two-class fire logits from four S2/S1/DEM observations | TerraMind Base | Base |
| `geo tessera` | Per-pixel Sentinel-1/2 time-series embeddings | Nano, Small, Medium, Large, 2.06B Teacher | `<6 GB` Nano, `6-7 GB` Small, `8-11 GB` Medium, `12-31 GB` Large, `32+ GB` Teacher |
| `geo olmoearth` | Multisensor spatial embeddings | Nano, Tiny, Small, Base | `<8 GB` Nano, `8-11 GB` Tiny, `12-15 GB` Small, `16+ GB` Base |

When `--model` is omitted, the runtime chooses the strongest tier recommended
for the machine. If that tier is not installed, it selects the strongest
installed tier below it. It does not silently download model weights. An
explicit model ID or converted model path always overrides automatic choice.

### TerraMind Fire

`mere.run geo fire` runs IBM and ESA's official ImpactMesh Fire checkpoint
through the same audited TerraMind Base temporal segmentation graph used by the
Flood checkpoint, but with independent immutable weights and provenance:

```bash
mere.run geo fire fire-input.safetensors \
  --output fire-logits.safetensors \
  --model /path/to/converted-terramind-fire \
  --json
```

The input must contain `S2L2A`, `S1RTC`, and `DEM` arrays shaped
`[batch, channels, 4, 256, 256]`, already normalized under the workflow's
declared raster policy. Output `logits` is float32 `[batch, 2, 256, 256]`.

### TESSERA v2

`mere.run geo tessera` encodes raw annual Sentinel-2 and Sentinel-1 observation
sequences. The four students emit Matryoshka embeddings; the full Teacher emits
its native 1,024-dimensional representation:

```bash
# Hardware-selected tier and native output width.
mere.run geo tessera pixel-series.safetensors \
  --output tessera-embeddings.safetensors \
  --json

# Force the 2.06B high-memory tier.
mere.run geo tessera pixel-series.safetensors \
  --model vision-embed-tessera-v2-teacher \
  --dimensions 1024 \
  --output tessera-teacher-embeddings.safetensors
```

Inputs use raw values and the exact upstream band orders. Required tensors are
`S2` `[batch, time, 10]`, `S2_DOY` `[batch, time]`, and at least one complete
`S1_ASC`/`S1_ASC_DOY` or `S1_DESC`/`S1_DESC_DOY` pair. Sentinel-1 arrays use
two bands. Day of year is an unnormalized integer from 1 through 365.

Students support `--dimensions 16`, `32`, `64`, or `128`. Teacher requires
`1024`. Teacher uses the upstream pooled normalization after merging ascending
and descending Sentinel-1 observations; students use separate source-specific
statistics. The runtime selects the correct contract from the immutable model
variant.

The Teacher evaluates 2.064B parameters per pixel. It is valuable for focused
research and distillation but is not the recommended bulk-embedding default. The
Large student is the automatic ceiling below 32 GB unified memory.

### OlmoEarth v1.2

`mere.run geo olmoearth` accepts one or more primary OlmoEarth imagery
modalities and emits a spatial feature grid for each one:

```bash
mere.run geo olmoearth observations.safetensors \
  --output spatial-embeddings.safetensors \
  --patch-size 4 \
  --input-resolution 10 \
  --json
```

The input must contain `TIMESTAMPS` shaped `[batch, time, 3]` as
`(day, zero-indexed month, year)`, plus one or more of:

- `S2L2A`: `[batch, height, width, time, 12]`
- `S1RTC`: `[batch, height, width, time, 2]`
- `LANDSAT`: `[batch, height, width, time, 11]`

`--patch-size` accepts `1`, `2`, `4`, or `8`. Smaller patches retain more
spatial detail and require more attention work. `--include-tokens` preserves
the full time axis in addition to the default time-pooled grids.

OlmoEarth's artifact license permits broad environmental and humanitarian
work, but prohibits military and defense applications, intelligence gathering,
human surveillance and policing, and listed extractive activities. Managed
pulls require `--accept-model-license`; the flag records review and acceptance
of upstream terms, not a `mere.run` judgment that a proposed use qualifies.

## Immutable conversion

Managed pulls download the pinned upstream checkpoint and source metadata.
The Swift runtime does not interpret the Python checkpoint. Convert it
once into the checksum-pinned float32 safetensors package:

```bash
python3 scripts/convert-terramind-fire-mlx.py \
  --checkpoint /path/to/TerraMind_v1_base_ImpactMesh_fire.pt \
  --configuration /path/to/terramind_v1_base_impactmesh_fire.yaml \
  --output /path/to/converted-terramind-fire

python3 scripts/convert-tessera-v2-mlx.py \
  --variant teacher \
  --checkpoint /path/to/tessera_v2_2B_teacher.pt \
  --output /path/to/converted-tessera-teacher

python3 scripts/convert-olmoearth-v12-mlx.py \
  --variant base \
  --weights /path/to/weights.pth \
  --configuration /path/to/source-config.json \
  --output /path/to/converted-olmoearth-base
```

Each converter rejects the wrong source hash, architecture, precision, tensor
inventory, or scalar count. Each runtime loader then verifies the exact output
artifact bytes and typed conversion receipt before loading weights.

For numerical verification against the official implementations, use
`scripts/validate-terramind-fire-reference.py`,
`scripts/validate-tessera-v2-reference.py`, and
`scripts/validate-olmoearth-v12-reference.py`. The repository's installed-model
gate constructs real input tensors and exercises the public `geo` commands.

## Why THOR is deferred

[THOR](https://github.com/FM4CS/THOR) was evaluated at source commit
`7ded3cea673ac21fe2bcadf9f3d9d4506eb6ab5f`. Its flexible 10-1,000 m
Sentinel-1/2/3 backbone is promising, especially for coarse Sentinel-3 climate
and ocean observations. The evaluated release is an embedding backbone rather
than a humanitarian decision head, however, and its S1/S2 role overlaps the
validated TESSERA and OlmoEarth routes.

THOR is therefore deferred until a workflow needs its genuinely distinct
Sentinel-3/native-resolution contract and can supply a usefulness gate for
that output. Adding another generic encoder without that consumer would expand
download, preprocessing, ALiBi, and validation surface without improving an
implemented result.

## Responsibility boundary

Flood and fire logits are candidates, not authoritative findings. Embeddings
are features, not conclusions. For humanitarian promotion, retain source
imagery and acquisition time, preprocessing provenance, model ID and immutable
revision, output artifact hashes, corroborating evidence, and accountable
human review.
