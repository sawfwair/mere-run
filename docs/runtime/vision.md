# Vision Runtime

This page covers captioning, inspection, and OCR.

## Public surface

- `mere.run vision caption`
- `mere.run vision inspect`
- `mere.run vision ocr`

## Model family

- `vision-ocr-lighton`

Captioning and inspect flows also depend on vision-language support code in
`MereRunCore`.

## Typical workflows

### Caption an image

```bash
swift run mere.run vision caption ./image.png
```

### Inspect an image with a question

```bash
swift run mere.run vision inspect ./image.png "What objects are visible?"
```

### OCR

```bash
swift run mere.run vision ocr ./page.png --backend lighton
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/VisionCaptionCommand.swift`
- `Sources/MereRunCLI/Commands/VisionInspectCommand.swift`
- `Sources/MereRunCLI/Commands/VisionOCRCommand.swift`

### OCR runtime

- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Loading.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRGenerator+Inference.swift`
- `Sources/MereRunCore/LightOnOCR/LightOnOCRSupport.swift`

### Vision-language support

- `Sources/MereRunCore/VLM/`
- `Sources/MereRunCore/QwenVLCaptioner.swift`
- `Sources/MereRunCore/Qwen25VLEncoder.swift`
- `Sources/MereRunCore/QwenVisionAttention.swift`

## How the OCR path works

1. the CLI resolves the OCR model
2. the OCR runtime loads the required components
3. the input image is normalized into the expected tensor form
4. OCR inference runs
5. text is emitted without the internal bring-up noise that older builds used
   to print

## How caption and inspect differ

- `caption` is a direct descriptive task
- `inspect` is a question-driven vision-language path

They share some of the same underlying vision support code, but they are
presented as separate public tasks because the user intent differs.
