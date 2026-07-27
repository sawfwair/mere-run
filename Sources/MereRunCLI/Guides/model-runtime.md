# Model Runtime

## Purpose

Read or update typed per-model API serving settings. These settings feed
`mere.run api serve`, `mere.run status --json`, and the macOS Models sheet.

## Required Models

Use a managed API-capable model id such as `text-chat-gemma4`,
`text-chat-q36-nano`, `text-chat-lfm25-a1b-8bit`, `text-agent-deepseek-v4-flash`,
`text-agent-qwen35-9b`, or `text-chat-mebot`.

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
- `--min-p` / `--clear-min-p`
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
- Use `--kv-cache-mode affine8` on Gemma4, Qwen-family, or LFM2 as a
  long-context memory control relative to full-precision KV. Qwen-family and
  LFM2 dequantize the generic cache for attention, so compare against `default`
  on the real checkpoint before keeping it enabled. Gemma uses its existing
  model-specific quantized path; `text-chat-gemma4-turbo` already defaults to a
  smaller 4-bit TurboQuant cache, so forcing affine 8-bit can increase its KV
  residency. `default` restores the engine/model/server default, not necessarily
  full precision.
- `ttlSeconds` unloads an idle, loaded model after that many seconds during the
  pool's opportunistic eviction passes. `pinned` exempts the model from
  automatic TTL/LRU eviction, but explicit unload still works.
- Managed embedding, image, TTS, and ASR API sidecars accept only `pinned` and
  `ttlSeconds`. They use an autonomous 300-second idle TTL when none is
  configured, re-read settings changes while idle, and reject aliases plus
  text-only context, sampling, engine, and KV controls. The special
  `qwen-image-edit` repository lane is resident but is not a managed
  runtime-settings target, so it currently uses the default lifecycle policy.
- Memory-pressure LRU uses the API server's `--memory-guard` tier. The guard
  derives soft/hard ceilings from Darwin physical footprint (RSS elsewhere),
  host memory
  headroom, and a tier reserve (`safe`, `balanced`, `aggressive`, or
  `custom`). Elevated pressure evicts the least-recently-used idle unpinned
  model; critical pressure evicts every idle unpinned model. Active requests
  are never evicted, and pinned models are skipped.

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
  --min-p 0.05 \
  --kv-cache-mode auto
```

```bash
mere.run model runtime get chat-default --json
```

```bash
mere.run model runtime set image-zimage-nano --ttl-seconds 45
mere.run model runtime set text-embed-qwen3-0.6b --ttl-seconds 120
mere.run model runtime set speech-asr-qwen3 --pinned
```

## Troubleshooting

- Unknown model: run `mere.run model list` and use a managed id or an existing
  alias.
- Unsupported model: residency settings apply only to API-resident text models
  and supported embedding, image, TTS, or ASR sidecars; vision, music, and
  video ids are rejected.
- Missing model at request time: run `mere.run model pull <id>` before selecting
  it through the matching chat, image, speech, or transcription endpoint.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCore/RuntimeModelSettings.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/ModelRuntimeCommand.swift
