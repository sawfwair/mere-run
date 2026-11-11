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
- `MiniMaxH3ImageTransformer`, image geometry, and the image VAE reuse H3 layers
  for six-step image generation. Image-specific PDD heads and two-axis VAE
  rotary embeddings remain separate from video and audio execution.
- Image decoding batches up to 32 existing overlapping tiles while preserving
  the original FP32 VAE, tile geometry, and assembly order. The encoder keeps
  sequential tiling. Computation fixtures compare sequential and batched decode,
  including multiple samples and a final partial batch.
- Large image attention accepts separate query and head limits while attending
  to every key. Core selects the qualified 28-head/1024-query schedule for
  native Q8 on `applegpu_g17s`; other models and architectures keep their
  previous attention schedule.
- `MiniMaxH3ImageModulationCache` holds exact outputs of the released six-step
  image schedule. Installing it releases timestep-only weights; prepared image
  layouts reuse rotary tables throughout denoising.
- `MiniMaxH3AdaLNCache` owns in-memory schedule tensors and interpolation. Core
  owns cache files, source identity validation, and adapter cache installation.

The target depends on MLX, `MereRunTensor`, and `MereRunAudioModels`. Core
re-exports public model types and retains checkpoint mapping, loading, request
policies, adapter installation, media I/O, and generation cleanup. Package access
connects those owners without making runtime hooks public API.

Run `H3RuntimeTests` for layout, model, cache, projection, and audio/video
computation coverage. Metal-specific tests retain explicit skips on CPU runs.
Layer fixtures do not establish published-checkpoint quality or throughput.
