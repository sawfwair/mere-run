# MereRunGeo

Earth-observation and satellite-imagery model runtimes carved out of `MereRunCore`.

- `OlmoEarth/`: OlmoEarth v1.2 multi-sensor encoder and checkpoint loader.
- `TESSERA/`: TESSERA v2 student encoder and checkpoint loader.
- `TerraMindFlood/` and `TerraMindFire/`: TerraMind flood and wildfire segmentation encoders and loaders.

Resource pins and architecture metadata stay in `MereRunCore` so model catalog discovery and validation stay centralized.
