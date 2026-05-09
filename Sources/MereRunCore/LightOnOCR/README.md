# LightOnOCR

OCR runtime built around Pixtral-style vision encoding and text generation.

- `LightOnOCRGenerator*.swift`: loading and inference.
- `LightOnOCRSupport.swift`: shared decoding/model helpers.
- `PixtralVisionEncoder.swift`: vision encoder path.

Keep OCR output formatting close to the generator and CLI file handling in
`MereRunCLI/Commands/VisionOCRCommand.swift`.
