# LightOnOCR

OCR runtime built around Pixtral-style vision encoding and text generation.

- `LightOnOCRGenerator*.swift`: loading and inference.
- `LightOnOCRSupport.swift`: shared decoding/model helpers.
- `PixtralVisionEncoder.swift`: vision encoder path.

Keep OCR output formatting close to the generator and CLI file handling in
`MereRunCLI/Commands/VisionOCRCommand.swift`.

Supported attention shapes use MLX fused SDPA by default. Set
`MERERUN_FUSED_SDPA=0` to restore materialized attention for diagnosis.
