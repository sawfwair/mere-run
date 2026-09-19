# AuK native model computation

This directory owns the AuK inference layers. Core owns tokenization, audio
features, model discovery, and sampling orchestration. The CLI publishes WAV output.

- `AuKCheckpoint.swift` reads original safetensors and maps convolution layouts.
- `AuKThinker.swift` implements the Qwen2.5-Omni-3B text and windowed audio encoder.
- `AuKDiT.swift` implements double-stream and single-stream diffusion blocks.
- `AuKVAE.swift` implements the posterior-mean encoder and BigVGAN decoder.
- `AuKSampling.swift` defines the base sway schedule and fixed Flash schedule.

The native implementation follows Tencent-Hunyuan/AuK revision
`6943a1e967409e8c73139a7a345f2a611cfb3dd6`. Preserve checkpoint rotary
frequencies, independent text/audio rotary positions in double-stream blocks,
final Qwen normalization before layer fusion, and centered decoder `conv_pre`.

Only original floating-point checkpoints are accepted. Quantized checkpoints
and training flows are outside this runtime's input contract. Reference encoding
uses the deterministic posterior mean, matching the upstream MLX inference path.
Trained-weight parity and bounded speech-generation/editing checks passed for
base and Flash on Apple Silicon. See the [qualification report](../../../docs/benchmarks/auk-native-qualification-2026-09-18.md)
for numerical tolerances, audio measurements, and untested scope.
