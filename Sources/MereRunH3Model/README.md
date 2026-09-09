# MiniMax-H3 model computation

Use this library to change MiniMax-H3 tensor computation independently of
checkpoint discovery, media preparation, and generation requests.

- `MiniMaxH3Transformer.swift` owns model state, construction, and resident weight
  conversion. Its extensions own forward execution, block scheduling, compiled
  runners, and AdaLN precomputation.
- Attention, feed-forward, conditioning, token-reduction, and final-output layers
  retain the checkpoint's module and parameter keys.
- `MiniMaxH3FusedKernels` and `MiniMaxH3FastVSA` separate dispatch from Metal kernel
  definitions. Their platform guards and debug benchmark paths remain explicit.
- `MiniMaxH3Geometry` owns packed audio/video rows, position grids, and temporal
  geometry. `MiniMaxH3Schedule` owns numerical schedules.
- Video encoder, decoder, and audio VAE types consume tensors. The audio VAE uses
  the shared BigVGAN layers from `MereRunAudioModels`.
- `MiniMaxH3AdaLNCache` owns in-memory schedule tensors and interpolation. Core
  owns cache files, source identity validation, and adapter cache installation.

The target depends on MLX, `MereRunTensor`, and `MereRunAudioModels`. Core
re-exports public model types and retains checkpoint mapping, loading, request
policies, adapter installation, media I/O, and generation cleanup. Package access
connects those owners without making runtime hooks public API.

Run `H3RuntimeTests` for layout, model, cache, projection, and audio/video
computation coverage. Metal-specific tests retain explicit skips on CPU runs.
Layer fixtures do not establish published-checkpoint quality or throughput.
