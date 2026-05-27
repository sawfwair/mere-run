# Status

## Purpose

Show one local snapshot of the mere.run runtime: API server reachability, the
model currently reported by `/v1/models`, the active model-store path, and the
managed models installed there.
When `/runtime/status` is available, the snapshot also includes runtime pool
entries, active request counts, request admission queue depth, memory pressure,
runtime capability flags, aggregate cache stats, per-model prefix KV cache
stats, per-model decode batching stats when enabled, and the settings file path.

## Required Models

No model is required.

## Install And Check

```bash
mere.run status
mere.run status --json
mere.run guide status
```

## Parameters

- `--host`: local API host to check, default `127.0.0.1`.
- `--port`: local API port to check, default `8080`.
- `--api-key`: bearer token for `/v1/models`, also read from `MERERUN_API_KEY`.
- `--timeout-seconds`: network probe timeout, default `1.0`.
- `--json`: emit a structured snapshot.

## Usage Patterns

- Run before connecting an editor, agent, or local integration to the API server.
- Run after `api serve` starts to confirm the health endpoint and served model.
- Use it instead of separate `curl /health`, `curl /v1/models`, and `model list`
  checks when you only need the current local state.
- Use `--json` in scripts, setup agents, or support tooling.
- Use it after manual load/unload or runtime settings changes to confirm the
  server sees the same control-plane state.
- During overlapping API traffic, check `request admission` to see how many
  requests are running and how many are queued behind `--max-active-requests`.
  JSON status also includes admitted, completed, and cancelled admission totals.
- Use `continuous batching` and `prefix KV reuse` to see which scheduler/cache
  features are actually enabled instead of assuming they are active.
- Use `cache stats` to check aggregate prefix hits, reused tokens, and batched
  decode steps across loaded models, including same-position versus
  variable-position decode batches when the selected engine can report them.
- Use `benchmark stats` to compare completed request counts, generated tokens,
  and average load/prefill/decode timings while cache and batching flags are on.

## Examples

```bash
mere.run status
```

```bash
mere.run status --host 127.0.0.1 --port 11434
```

```bash
MERERUN_API_KEY=change-me mere.run status --json
```

## Iteration Tips

- `server: down` means nothing answered the configured `/health` URL.
- `loaded models: unavailable (requires API key)` means the server is up, but
  `/v1/models` needs `--api-key` or `MERERUN_API_KEY`.
- The loaded model list is what the API server reports; it is not a system-wide
  RAM/process scan for every runtime family.
- The installed model list follows the same shared inventory path as
  `mere.run model list`.
- `runtime settings` points at
  `<active model store>/.mere-run/runtime-model-settings.json`.

## Troubleshooting

- Wrong port: rerun with the same `--host` and `--port` passed to `api serve`.
- Server is up but loaded model is unavailable: provide the API key used to
  start the server.
- Installed model missing: run `mere.run model list` or
  `mere.run model info <id>` for deeper model-store diagnostics.
- Non-default model store: pass global `--models-root` before `status` or set
  `MERERUN_MODELS_DIR`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/StatusCommand.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Support/ModelInventory.swift
