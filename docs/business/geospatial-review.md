# Review pinned satellite observations

Use Geospatial Tools to prepare traceable satellite inputs and run a local
flood-candidate or spatial-embedding workflow. The Valencia examples pin
observations around the October 2024 flood south of Valencia, Spain. They
provide a repeatable starting point for reviewing a small area and its source
lineage.

A flood mask is a candidate for independent review. Spatial embeddings are
numeric features for later analysis, not a damage assessment or a finding.

## Before you begin

Follow [Set up companion workflows](./plugin-workflows.md). Install Geospatial
Tools, a compatible Core runtime, and the relevant native model checkpoint:

```bash
mere.run plugin install mere-geo-tools --yes
mere.run plugin doctor mere-geo-tools
```

Use the [native Geo checkpoint setup](../runtime/geo.md#convert-a-checkpoint)
to obtain and convert the selected checkpoint. For graph execution, keep the
converted files in the resolved managed model directory. Preparation downloads
public imagery; inference uses the native Swift/MLX runtime, with no PyTorch
fallback in the provider.

Review upstream model terms. OlmoEarth has restrictions on military, defense,
intelligence, human-surveillance, policing, and listed extractive uses. Check
that your proposed analysis is permitted before using its artifacts.

## Choose an example

Save the recipe and matching graph from the
[Valencia example directory](https://github.com/sawfwair/mere-run-plugins/tree/main/examples/geo):

| Example | Files | Output |
| --- | --- | --- |
| TerraMind flood | `valencia-flood-sources.json`, `valencia-flood-graph.json` | Candidate mask, probability GeoTIFFs, preview, and provenance |
| OlmoEarth | `valencia-olmoearth-sources.json`, `valencia-olmoearth-graph.json` | Georeferenced spatial embeddings and provenance |

Both use a 10-meter UTM zone 30N grid and exact STAC collection and item IDs.
STAC, the SpatioTemporal Asset Catalog format, identifies the source imagery.
The flood example uses four optical/radar acquisition windows and elevation;
the embedding example uses two paired optical/radar timesteps.

## Prepare and inspect the flood inputs

Use the saved recipe to prepare a bundle:

```bash
mere.run plugin run mere-geo-tools -- prepare   --recipe ./valencia-flood-sources.json --output ./valencia-flood-data --json
mere.run plugin run mere-geo-tools -- inspect ./valencia-flood-data --json
```

Review bounds, coordinate system, resolution, cloud cover, acquisition dates,
and band order. Optical and radar observations don't necessarily share an
acquisition date. The bundle records materialized input hashes and source
identities; the recipe doesn't silently substitute a different scene.

## Run and check the graph

With `jq` installed, write the bundle path, preflight, then execute:

```bash
jq -n --arg bundle "$PWD/valencia-flood-data" '{bundle: $bundle}' > inputs.json
mere.run graph preflight ./valencia-flood-graph.json --inputs-json ./inputs.json --json
mere.run graph run ./valencia-flood-graph.json   --inputs-json ./inputs.json --run-dir ./valencia-flood-run --json
```

For OlmoEarth, use its matching recipe and graph with separate data and run
directories. Inspect the output manifest alongside the raster or embeddings.
Confirm the grid aligns with independent reference data and check cloud,
terrain, and temporal effects. Don't use a candidate mask alone to make
emergency, insurance, or property decisions.

The [Geospatial Tools reference](https://plugins-docs.mere.run/plugins/geo-tools)
explains supported modalities, preparation contracts, and hardware scaling.
