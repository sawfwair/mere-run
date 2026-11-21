# Inkling 2 numerical fixtures

`config.json` and `published-tensors.json` describe the original
`thinkingmachines/Inkling-2-Small` checkpoint at
`aac974122acd33ddb0d3e1f54433a9c5dd2249ce` (Apache-2.0), retrieved on
2026-11-13. The tensor fixture contains headers from all 24 shards; it contains
no checkpoint weight data. Its zeroed data offsets aren't file locations.

`mini-config.json` and `mini-published-tensors.json` describe the original
`thinkingmachines/Inkling-2-Mini` checkpoint at
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb` (Apache-2.0), retrieved on
2026-11-20. The tensor fixture contains all 1,244 headers from the main shard
with zeroed offsets and no weight data.

`generate_reference.py` creates deterministic small weights, a tiny configuration,
and FP64 NumPy decoder and vision outputs. Run it with Python 3 and NumPy to
regenerate the tiny configs, `weights.json`, `reference.json`, and
`mini-reference.json`. The Mini fixture reuses the base arrays and adds
decoder layers, controllers, and cross-pass message queries. The script
doesn't import Swift, MLX, or the runtime under test. These fixtures validate
the reference equations and cache behavior, not full-checkpoint quality or
the upstream quantized activation kernels.
