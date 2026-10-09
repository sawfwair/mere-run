# LightOnOCR

LightOnOCR 3 runtime: the 1B checkpoint uses Pixtral-style vision encoding and
Qwen3 text generation. The 0.8B and 4B checkpoints reuse the native Qwen3.5 runtime
through `LightOnOCRGenerator+Qwen.swift`. The typed architecture discriminator
selects the runtime for managed IDs and local paths. Plain mode sends no text;
grounding sends the exact trained prompt and preserves the model's raw output.

- `LightOnOCRGenerator*.swift`: loading and inference.
- `LightOnOCRSupport.swift`: shared decoding/model helpers.
- `PixtralVisionEncoder.swift`: vision encoder path.

Keep OCR output formatting close to the generator and CLI file handling in
`MereRunCLI/Commands/VisionOCRCommand.swift`.

Supported attention shapes use MLX fused SDPA by default. Set
`MERERUN_FUSED_SDPA=0` to restore materialized attention for diagnosis.

OCR decoding reclaims unused MLX buffers when the reusable pool reaches 1 GiB.
Growing sequence lengths can leave buffers that later tokens cannot reuse. The
check follows prefill and each decode step. It does not discard live model or KV
state, change global allocator limits, or change sampling and token limits.

Released BF16 checkpoints have a [bounded native qualification](../../../docs/benchmarks/lightonocr3-native-qualification-2026-10-08.md).
1B and 4B passed ten checked output cases each; 0.8B passed eight, with a
handwritten-word transcription and grounding failure. The opt-in
`LightOnOCR3CheckpointTests` records external-assets runs on Metal; ordinary
unit tests skip it. See the report for saved outputs and reproduction commands.

OCR inference is built for macOS and Linux. iOS retains the shared catalog
resources and mode metadata, without linking this CLI-only generator.
