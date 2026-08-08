# TESSERA v2 Source Map

This module implements the full TESSERA v2 pixel-encoder family in native
Swift/MLX. `TESSERAModel.swift` owns the dual Sentinel-2/Sentinel-1 temporal
encoders, attention pooling, student fusion projection, the Teacher's
QK-normalized modality-fusion transformer, and both upstream normalization
contracts. `TESSERAResources.swift` owns the immutable Nano, Small, Medium,
Large, and 2.06B Teacher source and conversion pins.

The public boundary is `mere.run geo tessera`. It accepts raw band sequences
and day-of-year arrays in safetensors. Students emit Matryoshka 16-, 32-, 64-,
or 128-dimensional embeddings; Teacher emits 1,024 dimensions. With no
explicit `--model`, hardware-aware resolution selects the strongest sensible
tier for the machine and falls back to the strongest installed tier at or
below it. The expensive Teacher is reserved for machines with at least 32 GB
of unified memory by the automatic policy; explicit selection remains
available on any supported machine that meets the 24 GB hard floor.

Use `scripts/convert-tessera-v2-mlx.py` for deterministic float32 conversion
and `scripts/validate-tessera-v2-reference.py` for PyTorch/MLX parity. The
Teacher is primarily a research and distillation artifact upstream; use it
deliberately for high-value analysis rather than bulk tile generation.
