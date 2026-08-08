# OlmoEarth v1.2 Source Map

This module implements native OlmoEarth v1.2 spatial embeddings for the three
primary imagery modalities: Sentinel-2 L2A, Sentinel-1 RTC, and Landsat.
`OlmoEarthModel.swift` owns normalization, flexible patch embedding,
space-time composite encodings, mixed 3D RoPE, transformer inference, and
time-pooled grids. `OlmoEarthResources.swift` owns the immutable Nano, Tiny,
Small, and Base source and conversion pins.

The public boundary is `mere.run geo olmoearth`. With no explicit `--model`,
hardware-aware resolution selects the strongest sensible tier for the machine
and falls back to the strongest installed tier at or below it.

OlmoEarth's artifact license permits many environmental and humanitarian uses
but includes material prohibited-use terms. Managed pulls therefore require
explicit review and `--accept-model-license`; mere.run records acceptance but
does not decide whether a use qualifies. Native code loads only the converted
safetensors package. Use `scripts/convert-olmoearth-v12-mlx.py` for conversion
and `scripts/validate-olmoearth-v12-reference.py` for official-reference parity.
