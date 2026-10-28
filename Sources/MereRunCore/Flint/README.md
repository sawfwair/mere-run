# Flint orchestration

Core loads the original BF16 expert layout or the pinned MLX INT4 export,
validates checkpoint coverage, prepares the published chat template, and runs
the shared autoregressive decoder. Model computation belongs to
`Sources/MereRunFlintModel`.

Read `docs/runtime/flint.md` for install commands, source revisions, supported
options, and validation boundaries. The INT4 loader checks the vendor's pinned
file hashes before interpreting U8 storage as packed U32 affine weights.
