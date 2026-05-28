# Model Runtime

## Purpose

Read or update typed per-model API serving settings. These settings feed
`mere.run api serve`, `mere.run status --json`, and the macOS Models sheet.

## Required Models

Use a managed API-capable model id such as `text-chat-gemma4`,
`text-chat-q35`, `text-agent-deepseek-v4-flash`, `text-agent-qwen35-9b`, or
`text-chat-mebot`.

## Install And Check

```bash
mere.run model runtime get text-chat-gemma4
mere.run model runtime set text-chat-gemma4 --alias chat-default --pinned
mere.run status --json
```

## Parameters

`get`:

- `--json`: emit the typed settings document for the selected model.

`set`:

- `--alias` / `--clear-alias`
- `--pinned` / `--unpinned`
- `--ttl-seconds` / `--clear-ttl`
- `--max-context-tokens` / `--clear-max-context-tokens`
- `--max-tokens` / `--clear-max-tokens`
- `--temperature` / `--clear-temperature`
- `--top-p` / `--clear-top-p`
- `--engine` / `--clear-engine`
- `--kv-cache-mode` / `--clear-kv-cache-mode`
- `--json`

## Usage Patterns

- Configure aliases before starting `api serve` so clients can request a stable
  short model name.
- Keep settings in the model store by using global `--models-root` or
  `MERERUN_MODELS_DIR` when you use a custom store.
- Use engine overrides only for catalog-compatible engines. Incompatible
  overrides are rejected immediately.
- Use `--kv-cache-mode auto` on Gemma4 to keep the default KV path for shorter
  prompts and switch to packed 2-bit PolarKV for prompts at or above 1024
  tokens. `polar2` forces that path for every request, and non-Gemma4 models
  reject the setting.
- Treat TTL/LRU as forward-compatible settings. The first runtime-pool
  milestone records them and exposes them in status; later schedulers can act
  on them.

## Examples

```bash
mere.run model runtime set text-chat-gemma4 \
  --alias chat-default \
  --pinned \
  --ttl-seconds 3600 \
  --max-context-tokens 8192 \
  --max-tokens 1024 \
  --temperature 0.6 \
  --top-p 0.9 \
  --kv-cache-mode auto
```

```bash
mere.run model runtime get chat-default --json
```

## Troubleshooting

- Unknown model: run `mere.run model list` and use a managed id or an existing
  alias.
- Unsupported model: choose an API-capable text model rather than image, speech,
  vision, music, or video ids.
- Missing model at request time: run `mere.run model pull <id>` before selecting
  it through `/v1/chat/completions`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/RuntimeModelSettings.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelRuntimeCommand.swift
