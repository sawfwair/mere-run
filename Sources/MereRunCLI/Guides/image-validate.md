# Image Validate

## Purpose

Run deterministic checks against local image model layouts. This is for runtime debugging and regression checks, not normal image creation.

## Required Models

Use an installed image family:

- `zimage`: resolves the Z-Image managed family.
- `klein`: resolves the FLUX.2 Klein managed family.

## Install And Check

```bash
mere.run model pull image-zimage-nano
mere.run image validate --family zimage --test all
mere.run image validate --help
```

## Parameters

- `--test`, `-t`: `vae`, `encoder`, `transformer`, `pipeline`, or `all`.
- `--family`, `-m`: `zimage` or `klein`.
- `--output`, `-o`: output directory for artifacts.
- `--save-reference`: store reference latents/outputs for later comparison.
- `--compare`: compare a new run with reference outputs.
- `--reference-dir`: reference directory to compare against.

## Usage Patterns

- Use `--test vae` for decode/reconstruction failures.
- Use `--test encoder` for prompt/tokenizer or text encoder issues.
- Use `--test transformer` for denoising shape or weight-layout issues.
- Use `--test pipeline` after component tests pass.

## Examples

```bash
mere.run image validate --family zimage --test all --output ./validation/zimage
```

```bash
mere.run image validate \
  --family klein \
  --test pipeline \
  --save-reference \
  --output ./validation/klein-reference
```

## Iteration Tips

- Save a reference only after a known-good local run.
- Keep validation artifacts outside source directories.
- Compare the smallest failing component before rerunning the full suite.

## Troubleshooting

- Unknown family: use `zimage` or `klein`.
- Model not found: pull the matching managed model first.
- Comparison fails after code changes: inspect the component-level output before assuming the full pipeline is wrong.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ImageValidateCommand.swift
- https://huggingface.co/docs/diffusers/api/pipelines/z_image
- https://docs.bfl.ai/guides/prompting_guide_flux2_klein
