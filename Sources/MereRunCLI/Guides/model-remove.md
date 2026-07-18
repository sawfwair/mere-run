# Model Remove

## Purpose

Delete an installed managed model from the local model store to reclaim disk space.

## Required Models

The target model must be installed.

## Install And Check

```bash
mere.run model list
mere.run model remove image-zimage-nano
mere.run status
```

## Parameters

- positional target: canonical model id.
- `--force`: skip confirmation.
- `--keep-cache`: remove install links but retain backing Hub payloads.
- `--json`: emit a structured result; requires `--force`.

## Usage Patterns

- Run `model storage` first to distinguish referenced, shared, and reclaimable bytes.
- Use interactive confirmation for manual cleanup.
- Use `--force` only in scripts where the target id is known.

## Examples

```bash
mere.run model remove image-zimage-nano
```

```bash
mere.run model remove text-chat-gemma4-nano --force
```

## Iteration Tips

- Remove large creative models before pulling a different family.
- Re-run `status` or `model list` after deletion to confirm status.
- The default removes unshared Hub payloads and preserves anything still referenced.
- Use `--keep-cache` only when retaining bytes for a future reinstall is intentional.

## Troubleshooting

- Unknown id: run `model list`.
- Not installed: no deletion is needed.
- Disk still full: run the read-only `mere.run model gc` preview.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelRemoveCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/docs/configuration.md
