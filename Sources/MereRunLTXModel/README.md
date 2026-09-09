# LTX model computation

Use this module when editing LTX tensor computation. It depends on MLX, MLXFast,
and MLXNN. It builds without Core, tokenizers, model storage, AudioCodecs,
MediaIO, or CLI dependencies.

- `LTXDistilledTransformer.swift`: shared attention, feed-forward, and timestep layers
- `LTXUnifiedAVTransformer*.swift`: legacy and LTX 2.3/2.5 joint A/V transformers
- `LTXAudioOnlyTransformerV2.swift`: audio-only transformer stages
- `LTXVideoEncoder.swift` and `LTXVideoDecoder.swift`: convolutional video VAE
- `LTXDiffVAE*.swift` and `LTXDiffusionVideoDecoder.swift`: diffusion video decoder,
  neighborhood attention, and temporal/spatial layouts
- `LTXLatentUpsampler.swift`: spatial and temporal latent upsampling
- `LTXAudioVAE.swift`: audio encoding and decoding
- `LTX25DurationHead.swift` and `LTX25FrameGeometry.swift`: duration prediction
- `LTXVideoDecodeTiling.swift`: decode geometry and weighted tile accumulation
- `LTXTeaCache.swift`: synchronized guidance-branch residual reuse and diagnostics
- `LTXSamplerSupport.swift`: sampler types and numerical updates
- `LTXParityDiagnostics.swift`: optional tensor dumps and parity-noise fixtures

Core retains checkpoint selection, weight mapping, loading, text encoding,
media input, generation actors, and output. Public model types remain available
through Core imports. Package hooks support existing generation code without
making model internals part of the public API.

Run `swift build --target MereRunLTXModel`, `swift test --filter LTXRuntimeTests`,
and `bash scripts/check-model-boundaries.sh` after changes. Complete the repository
gate before opening a PR. Fixture tests do not qualify published checkpoints,
GPU numerics, or end-to-end video quality.
