# Vision OCR

## Purpose

Extract text from one or more images using LightOnOCR or an external GLM-OCR CLI, with optional backend comparison.

## Required Models

Default managed LightOn id: `vision-ocr-lighton`. The GLM backend requires an installed `glmocr` CLI instead of the LightOn model unless `--compare` is used.

## Install And Check

```bash
mere.run model pull vision-ocr-lighton
mere.run vision ocr --help
```

## Parameters

- positional images: one or more image paths.
- `--backend`, `-b`: `lighton` or `glm`.
- `--compare`: run both backends.
- `--model`, `-m`: LightOn managed id or local model directory.
- `--glmocr-cli`: path to the `glmocr` executable.
- `--glm-config`: GLM-OCR config YAML.
- `--output`, `-o`: output directory for `.txt` files.
- `--max-tokens`: generated token cap.
- `--temperature`: lower is more deterministic.
- `--quiet`, `-q`: output only OCR text.

## Usage Patterns

- Use `--backend lighton` for the self-contained path.
- Use `--quiet` for shell pipelines.
- Use `--compare` when evaluating OCR quality on a new document type.
- Preprocess by cropping, deskewing, or improving contrast before changing sampling.

## Examples

```bash
mere.run vision ocr ./receipt.jpg --quiet
```

```bash
mere.run vision ocr ./pages/*.png \
  --backend lighton \
  --output ./ocr-text \
  --temperature 0.1
```

## Iteration Tips

- Keep temperature low for extraction.
- Split large scans into pages or regions when layout is complex.
- Compare backends on a small sample before batch OCR.

## Troubleshooting

- GLM backend fails: confirm `glmocr` is installed and on PATH, or pass `--glmocr-cli`.
- LightOn model missing: run `mere.run model pull vision-ocr-lighton`.
- Text is scrambled: crop/deskew the source image.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionOCRCommand.swift
- https://huggingface.co/lightonai/LightOnOCR-2-1B
- https://github.com/zai-org/GLM-OCR
