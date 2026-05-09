# Model Remove

## Purpose

Delete an installed managed model from the local model store to reclaim disk space.

## Required Models

The target model must be installed.

## Install And Check

```bash
mere.run model list
mere.run model remove image-zimage-max
```

## Parameters

- positional target: canonical model id.
- `--force`: skip confirmation.

## Usage Patterns

- Run `model list` first to confirm status and size.
- Use interactive confirmation for manual cleanup.
- Use `--force` only in scripts where the target id is known.

## Examples

```bash
mere.run model remove image-zimage-max
```

```bash
mere.run model remove text-chat-gemma4-nano --force
```

## Iteration Tips

- Remove large creative models before pulling a different family.
- Re-run `model list` after deletion to confirm status.
- Remember that hub caches may still consume disk outside the model store.

## Troubleshooting

- Unknown id: run `model list`.
- Not installed: no deletion is needed.
- Disk still full: inspect `MERERUN_HUB_CACHE` or the Hugging Face cache.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelRemoveCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/docs/configuration.md
