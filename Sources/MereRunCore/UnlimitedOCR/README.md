# Unlimited-OCR-2 orchestration

`UnlimitedOCRGenerator` loads the pinned original BF16 shards, validates every
tensor name, shape, and dtype, and builds the trained `<PAGE><image>` prompt.
All supplied images form one ordered document. It preserves output page markers
and enforces the upstream page-count and per-page token guards.

`UnlimitedOCRImage` prepares Pillow-compatible bicubic global views with gray
letterboxing and optional 640px tiles. `MereRunUnlimitedOCRModel` owns SAM,
CLIP, the projector, DeepSeek MoE, page positions, and sparse DASH attention.

See [the runtime guide](../../../docs/runtime/unlimited-ocr-2.md) for usage,
memory limits, and validation status. Inference is available on macOS and Linux;
iOS keeps catalog metadata without the CLI generator.
