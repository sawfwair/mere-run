# Model Repair Manifests

## Purpose

Write missing `mererun_model.json` files for known model directories in the local model store. Use this for old installs or manually copied managed models.

## Required Models

At least one known model directory should exist in the configured model store.

## Install And Check

```bash
mere.run model repair-manifests --dry-run
mere.run model repair-manifests
```

## Parameters

- `--dry-run`: print what would change without writing files.

## Usage Patterns

- Always start with `--dry-run`.
- Run after moving models into the managed store by hand.
- Follow with `model info <id>` to verify a repaired model.

## Examples

```bash
mere.run model repair-manifests --dry-run
```

```bash
MERERUN_MODELS_DIR=/Volumes/Models/mere.run mere.run model repair-manifests
```

## Iteration Tips

- Repair manifests before debugging component paths.
- Keep a copy of custom manifests if you edited them by hand.
- Do not use this to invent manifests for unknown model families.

## Troubleshooting

- No candidate directories: confirm the model store path.
- Skipped ids: only known managed ids can be repaired.
- Still invalid: inspect `model info --components` and the directory layout.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelRepairManifestsCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/MereRunModelManifest.swift
