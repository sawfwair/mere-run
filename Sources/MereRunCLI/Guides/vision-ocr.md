# Vision OCR

## Purpose

Extract text from one or more images using LightOnOCR, native Infinity-Parser2,
or external comparison CLIs, with optional backend comparison.

## Required Models

Default managed LightOn id: `vision-ocr-lighton`.

Default native Infinity id: `vision-ocr-infinity-pro-int8`. The full BF16
leaderboard Pro model remains
available as `vision-ocr-infinity-pro`, but it is large enough that it requires
an explicit pull and should be treated as a heavyweight compatibility target.

The GLM backend requires an installed `glmocr` CLI instead of the LightOn model
unless `--compare` is used. The Infinity external runtime requires the upstream
`infinity_parser2` Python package and its `parser` CLI; use it only for parity
checks against an upstream Transformers/vLLM run.

## Install And Check

```bash
mere.run model pull vision-ocr-lighton
mere.run model pull vision-ocr-infinity-pro-int8
mere.run vision ocr --help
```

## Parameters

- positional images: one or more image paths.
- `--backend`, `-b`: `lighton`, `glm`, or `infinity`.
- `--compare`: compare LightOn against the selected secondary backend; defaults
  to GLM when `--backend lighton`.
- `--model`, `-m`: LightOn managed id or local model directory.
- `--glmocr-cli`: path to the `glmocr` executable.
- `--glm-config`: GLM-OCR config YAML.
- `--infinity-runtime`: `native` or `external`.
- `--infinity-model`: native managed model id or local path; upstream model or
  server id when `--infinity-runtime external`.
- `--infinity-parser-cli`: path to the Infinity-Parser2 `parser` executable for
  `--infinity-runtime external`.
- `--infinity-backend`: external `vllm-server`, `vllm-engine`, or `transformers`.
- `--infinity-api-url`: external vLLM server chat-completions URL.
- `--infinity-api-key`: optional external vLLM server API key; upstream accepts
  `EMPTY` for local servers.
- `--infinity-task`: `doc2json`, `doc2md`, or `custom`.
- `--infinity-prompt`: custom prompt for `--infinity-task custom`.
- `--infinity-output-format`: `md` or `json`.
- `--output`, `-o`: output directory for `.txt` files.
- `--max-tokens`: generated token cap.
- `--temperature`: lower is more deterministic.
- `--quiet`, `-q`: output only OCR text.

## Usage Patterns

- Use `--backend lighton` for the small managed OCR path.
- Use `--backend infinity` for the native Infinity-Parser2 Pro int8 model.
- Use `--quiet` for shell pipelines.
- Use `--compare --backend infinity` when evaluating Infinity-Parser2 against
  the managed LightOn path on a new document type.
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

```bash
mere.run vision ocr ./page.png \
  --backend infinity \
  --infinity-task doc2md \
  --quiet
```

```bash
mere.run vision ocr ./page.png \
  --compare \
  --backend infinity
```

```bash
mere.run vision ocr ./page.png \
  --backend infinity \
  --infinity-runtime external \
  --infinity-api-url http://127.0.0.1:8000/v1/chat/completions
```

## Iteration Tips

- Keep temperature low for extraction.
- Split large scans into pages or regions when layout is complex.
- Compare backends on a small sample before batch OCR.

## Troubleshooting

- GLM backend fails: confirm `glmocr` is installed and on PATH, or pass `--glmocr-cli`.
- Native Infinity model missing: run
  `mere.run model pull vision-ocr-infinity-pro-int8`.
- External Infinity fails: confirm `parser` is installed and the selected vLLM
  server or CUDA backend is reachable.
- LightOn model missing: run `mere.run model pull vision-ocr-lighton`.
- Text is scrambled: crop/deskew the source image.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionOCRCommand.swift
- https://huggingface.co/lightonai/LightOnOCR-2-1B
- https://github.com/zai-org/GLM-OCR
- https://huggingface.co/infly/Infinity-Parser2-Pro
- https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser2
