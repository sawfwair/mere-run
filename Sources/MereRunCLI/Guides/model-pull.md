# Model Pull

## Purpose

Download a managed Hugging Face-backed model into the local mere.run model store.

## Required Models

No model is required before running pull, but the target id must exist in the managed catalog and have a public download source.

## Install And Check

```bash
mere.run model capabilities
mere.run model pull image-zimage-nano
mere.run status
mere.run model list
```

## Parameters

- positional target: canonical model id. Omit only with `--all`.
- `--all`: pull every model that has a Hugging Face source and passes support checks.
- `--force`: re-download even if already installed.
- `--quiet`, `-q`: suppress progress.
- `--allow-unsupported`: bypass platform and memory support checks.
- `--accept-model-license`: confirm that you reviewed and accept all listed
  third-party model/component terms and agree to comply with them before a
  restricted download begins.
- `--preflight`: inspect support, source, install state, and disk paths without downloading.
- `--json`: with `--preflight`, emit a structured report for scripts and apps.

## Usage Patterns

- Run `model capabilities` before large downloads.
- Use `--preflight --json` before large pulls or `--all` to get structured
  diagnostics and follow-up actions without downloading.
- Prefer a single target over `--all`.
- Use `--force` for a suspected corrupt install.
- Use external storage with `MERERUN_MODELS_DIR` before pulling very large models.

## Examples

```bash
mere.run model pull image-zimage-nano --preflight --json
```

```bash
mere.run model pull text-chat-gemma4-nano
mere.run model pull text-chat-lfm25-2.6b-4bit --accept-model-license
mere.run model pull text-chat-lfm25-a1b-8bit --accept-model-license
mere.run model pull text-chat-laguna-s-2-1
mere.run model pull text-chat-laguna-xs-2-1
```

```bash
MERERUN_MODELS_DIR=/Volumes/Models/mere.run \
  mere.run model pull music-acestep
MERERUN_MODELS_DIR=/Volumes/Models/mere.run \
  mere.run model pull music-magenta-rt2-small
MERERUN_MODELS_DIR=/Volumes/Models/mere.run \
  mere.run model pull sfx-woosh-dflow --accept-model-license
MERERUN_MODELS_DIR=/Volumes/Models/mere.run \
  mere.run model pull sfx-woosh-flow --accept-model-license
```

## Iteration Tips

- Pull the runtime model before opening a creative loop.
- Use `status` to confirm the model landed in the active store.
- Verify with `model info <id>` after a forced re-download.
- Keep hub cache and model store on the same large disk when possible.
- If disk is tight, set both `MERERUN_HUB_CACHE` and `MERERUN_MODELS_DIR`.

## Troubleshooting

- Unknown id: run `mere.run model list`.
- No Hugging Face source: use a local path or choose a pullable model.
- Unsupported machine: pick a smaller recommended model or pass `--allow-unsupported` only when the user accepts the risk.
- Restricted model terms: review every listed license URL and pass
  `--accept-model-license` only to confirm acceptance and agreement to comply.
- Not enough free disk space: free the reported cache volume, run `mere.run model remove <id>`, or move both the Hugging Face cache and model store:
  `MERERUN_HUB_CACHE=/Volumes/Models/huggingface MERERUN_MODELS_DIR=/Volumes/Models/mere.run mere.run model pull <id>`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelPullCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/ManagedModelResolver.swift
- https://huggingface.co/docs/huggingface_hub/guides/download
