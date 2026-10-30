# Geospatial runtime

`mere.run geo` runs Earth-observation models on tensors you prepare. It
produces flood and fire class scores for satellite tiles, and embeddings for
Sentinel time series and multisensor imagery. You can carry the resulting
candidates or embeddings into your own evidence process.

Your workflow owns acquisition, cloud masking, reprojection, tiling,
georeferencing, source citations, and analyst review. mere.run owns immutable
model provenance, native Swift/MLX execution, hardware-aware model selection,
and machine-readable outputs.

## Commands and models

Each command reads one safetensors file and writes one safetensors file:

| Command | What it produces | Managed models | Memory (min / recommended) |
| --- | --- | --- | --- |
| `geo flood` | Two-class flood logits from four S2/S1/DEM observations | `vision-flood-terramind-base` | 16 / 24 GB |
| `geo fire` | Two-class fire logits from four S2/S1/DEM observations | `vision-fire-terramind-base` | 16 / 24 GB |
| `geo tessera` | Per-pixel Sentinel-1/2 time-series embeddings | `vision-embed-tessera-v2-nano`, `-small`, `-medium`, `-large`, `-teacher` | 4 / 6 GB (Nano) to 24 / 32 GB (Teacher) |
| `geo olmoearth` | Multisensor spatial embeddings | `vision-embed-olmoearth-v12-nano`, `-tiny`, `-small`, `-base` | 4 / 8 GB (Nano) to 16 / 24 GB (Base) |

Every command accepts `--model` with a managed ID or a converted model root,
`--preflight` to validate inputs and model availability without loading
weights, and `--json` to print structured execution metadata on stdout.

### Automatic model selection

When you omit `--model`, `geo tessera` and `geo olmoearth` choose the strongest
tier recommended for your machine's unified memory. If that tier isn't
installed, they use the strongest installed tier below it. They never download
weights automatically, and an explicit `--model` always wins.

| Command | Under 6 GB | 6 to 7 GB | 8 to 11 GB | 12 to 15 GB | 16 to 31 GB | 32 GB or more |
| --- | --- | --- | --- | --- | --- | --- |
| `geo tessera` | Nano | Small | Medium | Large | Large | Teacher |
| `geo olmoearth` | Nano | Nano | Tiny | Small | Base | Base |

TerraMind Flood and Fire each have one tier, Base.

## Convert a checkpoint

Managed pulls download the pinned upstream checkpoint and source metadata. The
Swift runtime doesn't interpret the Python checkpoint, so convert it once into
a checksum-pinned float32 safetensors package:

```bash
python3 scripts/convert-terramind-flood-mlx.py \
  --checkpoint /path/to/TerraMind_v1_base_ImpactMesh_flood.pt \
  --configuration /path/to/terramind_v1_base_impactmesh_flood.yaml \
  --output /path/to/converted-terramind-flood

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
bytes and typed conversion receipt before it loads weights.

To check numerical agreement with the official implementations, use
`scripts/validate-terramind-fire-reference.py`,
`scripts/validate-tessera-v2-reference.py`, and
`scripts/validate-olmoearth-v12-reference.py`. The installed-model gate builds
real input tensors and runs the public `geo` commands.

## Segment flood and fire tiles

`geo flood` and `geo fire` run IBM and ESA's official TerraMind Base
ImpactMesh checkpoints. Both use the same audited temporal segmentation graph,
with independent immutable weights and provenance.

```bash
mere.run geo flood flood-input.safetensors \
  --output flood-logits.safetensors \
  --model /path/to/converted-terramind-flood \
  --json

mere.run geo fire fire-input.safetensors \
  --output fire-logits.safetensors \
  --model /path/to/converted-terramind-fire \
  --json
```

The input must contain three arrays shaped `[batch, channels, 4, 256, 256]`:
`S2L2A` with 12 channels, `S1RTC` with 2, and `DEM` with 1. Normalize them
under your workflow's declared raster policy before inference. The output
`logits` array is float32 `[batch, 2, 256, 256]`.

## Encode time series with TESSERA v2

`geo tessera` encodes raw annual Sentinel-2 and Sentinel-1 observation
sequences. The four students emit Matryoshka embeddings, and the Teacher emits
its native 1,024-dimensional representation.

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
`S1_ASC`/`S1_ASC_DOY` or `S1_DESC`/`S1_DESC_DOY` pair. Sentinel-1 arrays have
two bands. Day of year is an unnormalized integer from 1 through 365.

Students accept `--dimensions` of 16, 32, 64, or 128; the Teacher requires
1024. The command rejects a mismatched value before it verifies or loads the
checkpoint, including for the tier it picks automatically. The Teacher uses the
upstream pooled normalization after merging ascending and descending
Sentinel-1 observations; students use separate statistics for each source. The
runtime selects the right contract from the immutable model variant.

The Teacher evaluates 2.064B parameters per pixel. It suits focused research
and distillation, not bulk embedding, and the Large student is the automatic
ceiling below 32 GB of unified memory.

## Encode multisensor observations with OlmoEarth v1.2

`geo olmoearth` accepts one or more primary OlmoEarth imagery modalities and
emits a spatial feature grid for each one.

```bash
mere.run model pull vision-embed-olmoearth-v12-base --accept-model-license
mere.run geo olmoearth observations.safetensors \
  --output spatial-embeddings.safetensors \
  --patch-size 4 \
  --input-resolution 10 \
  --json
```

The input must contain `TIMESTAMPS` shaped `[batch, time, 3]` as
`(day, zero-indexed month, year)`, plus one or more of these arrays:

- `S2L2A`: `[batch, height, width, time, 12]`
- `S1RTC`: `[batch, height, width, time, 2]`
- `LANDSAT`: `[batch, height, width, time, 11]`

`--patch-size` accepts 1, 2, 4 (default), or 8 pixels. Smaller patches keep
more spatial detail and need more attention work. `--input-resolution` is the
input ground sample distance in meters (default 10). `--include-tokens` also
writes the full space-time tokens beside the default time-pooled grids.

The OlmoEarth Artifact License permits broad environmental and humanitarian
work, but prohibits military and defense applications, intelligence
gathering, human surveillance and policing, and extractive activities such as
drilling, mining, and deforestation. Managed pulls require
`--accept-model-license`. The flag records that you reviewed and accepted the
upstream terms; it isn't a mere.run judgment that a proposed use qualifies.

## Responsibility boundary

Flood and fire logits are candidates, not authoritative findings. Embeddings
are features, not conclusions. Before you promote a result for humanitarian
use, retain the source imagery and acquisition time, preprocessing provenance,
model ID and immutable revision, output artifact hashes, corroborating
evidence, and accountable human review.

## Why THOR is deferred

[THOR](https://github.com/FM4CS/THOR) was evaluated at source commit
`7ded3cea673ac21fe2bcadf9f3d9d4506eb6ab5f`. Its flexible 10 to 1,000 m
Sentinel-1/2/3 backbone is promising, especially for coarse Sentinel-3 climate
and ocean observations. The evaluated release is an embedding backbone rather
than a humanitarian decision head, though, and its S1/S2 role overlaps the
validated TESSERA and OlmoEarth routes.

THOR is deferred until a workflow needs its distinct
Sentinel-3/native-resolution contract and can supply a usefulness gate for
that output. Another generic encoder without that consumer adds download,
preprocessing, ALiBi, and validation surface without improving an implemented
result.
