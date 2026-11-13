# Inkling 2 numerical fixtures

`config.json` and `published-tensors.json` describe the original
`thinkingmachines/Inkling-2-Small` checkpoint at
`aac974122acd33ddb0d3e1f54433a9c5dd2249ce` (Apache-2.0), retrieved on
2026-11-13. The tensor fixture contains headers from all 24 shards; it contains
no checkpoint weight data. Its zeroed data offsets aren't file locations.

`generate_reference.py` creates deterministic small weights, a tiny configuration,
and FP64 NumPy decoder and vision outputs. Run it with Python 3 and NumPy to
regenerate `tiny-config.json`, `weights.json`, and `reference.json`. The script
doesn't import Swift, MLX, or the runtime under test. These fixtures validate
the reference equations and cache behavior, not full-checkpoint quality or
the upstream quantized activation kernels.
