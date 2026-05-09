# Model Info

## Purpose

Inspect a managed model id or local model root: manifest, validation status, and component directories.

## Required Models

The target must be either an installed managed id or an existing local model path.

## Install And Check

```bash
mere.run model list
mere.run model info image-zimage-max
mere.run model info image-zimage-max --components
```

## Parameters

- positional target: canonical model id or local model root path.
- `--json`: print raw `mererun_model.json`.
- `--components`: print resolved component directories.

## Usage Patterns

- Use before debugging a failed generation command.
- Use `--components` for structured roots with tokenizer, transformer, VAE, scheduler, or text encoder folders.
- Use `--json` when scripts need the manifest.

## Examples

```bash
mere.run model info music-acestep --components
```

```bash
mere.run model info ~/Models/custom-zimage --json
```

## Iteration Tips

- Compare `model info` output for a working and failing model root.
- If manifest is missing, try `model repair-manifests`.
- For local paths, verify you are pointing at the model root, not a nested component.

## Troubleshooting

- Not a known id: run `model list` for canonical ids.
- Manifest missing: `--json` requires an actual manifest file.
- Gemma alias not installed: point to a local path or let the runtime auto-download where supported.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelInfoCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/MereRunModelManifest.swift
