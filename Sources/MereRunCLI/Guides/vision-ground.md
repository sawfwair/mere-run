# Vision Ground

## Purpose

Find image regions matching natural-language queries and write an annotated image plus JSON metadata. Use this when the target is open-vocabulary grounding with masks.

## Required Models

Managed id: `vision-ground-falcon-perception`, or a local Falcon Perception model root.

## Install And Check

```bash
mere.run model pull vision-ground-falcon-perception
mere.run vision ground --help
```

## Parameters

- positional image: image file path.
- `--query`, `--prompt`: one or more grounding expressions.
- `--model`, `-m`: managed id or local model root.
- `--output`, `-o`: annotated image path.
- `--json-output`: JSON metadata path.
- `--mask-output-dir`: directory for per-detection mask PNGs.

## Prompting Patterns

- Query with concrete object phrases: `red backpack`, `person holding a phone`.
- Use multiple `--query` values for separate concepts.
- Prefer attributes that are visible: color, shape, material, location.
- For small text-dependent objects, run OCR first or crop closer.

## Examples

```bash
mere.run vision ground ./street.jpg \
  --query "person in red jacket" "blue bicycle" \
  --output ./street-grounded.jpg \
  --json-output ./street-grounded.json
```

## Iteration Tips

- Start broad, then add attributes to reduce false positives.
- Save masks when downstream compositing or measurement needs exact regions.
- For crowded scenes, test one query at a time.

## Troubleshooting

- No detections: simplify the query and check image resolution.
- Too many detections: add visible attributes or spatial wording.
- Model missing: run `mere.run model pull vision-ground-falcon-perception`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/VisionGroundCommand.swift
- https://huggingface.co/tiiuae/Falcon-Perception
