# Model List

## Purpose

Show all known managed model ids, categories, install status, and resolved local
payload size. Use this before pulling or running models.
Use `mere.run status` when you also need the active API server and currently
served model.

## Required Models

No model is required.

## Install And Check

```bash
mere.run model list
mere.run status
mere.run guide model list
```

## Parameters

This command has no flags. It reads the configured model store.

## Usage Patterns

- Run before telling a user to pull something; it shows what is already installed.
- Prefer `status` for a first support snapshot because it includes the model
  store and server state.
- Pair with `model capabilities` when choosing a first model.
- Use `MERERUN_MODELS_DIR` or `--models-root` before the command when inspecting a non-default store.

## Examples

```bash
mere.run model list
```

```bash
MERERUN_MODELS_DIR=/Volumes/Models/mere.run mere.run model list
```

## Iteration Tips

- Look for `missing` vs `installed`, not just whether the id exists.
- The size follows symlinked payload files and directories in managed model
  stores. If layout is confusing, inspect with `model info`.
- If validation looks wrong, inspect with `model info` or repair manifests.

## Troubleshooting

- Expected model missing: confirm model storage environment variables.
- Installed alias looks confusing: Gemma and code models may resolve through managed aliases.
- Need hardware fit: use `mere.run model capabilities --all`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelListCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/ManagedModelCatalog.swift
