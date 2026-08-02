# Status

## Purpose

Show one local snapshot of the mere.run runtime: API server reachability, the
model currently reported by `/v1/models`, the active model-store path, and the
managed models installed there.
The snapshot also reads the machine-wide inference admission ledger even when
the API server is down. It reports weighted permit capacity, active commands,
queued commands, process IDs, resource classes, memory pressure, and disk
headroom without recording prompts or generated content.
When `/runtime/status` is available, the snapshot also includes runtime pool
entries, active request counts, request admission queue depth, memory pressure,
runtime capability flags, aggregate cache stats, per-model prefix KV cache
stats, per-model decode batching stats when enabled, embedding/image/TTS/ASR sidecar
residency and readiness, the settings file path, and additive process/device
telemetry where supported. In JSON, `loaded` is the
backward-compatible resident-object signal; additive `ready: false` means a text
model is still preparing or a sidecar's first operation is loading or failed.
Older payloads may omit `ready` and `process`.

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
- Inspect `capabilities.asrStreamingBackends` before routing a live ASR stream
  to Parakeet or Qwen.
- Use it after manual load/unload or runtime settings changes to confirm the
  server sees the same control-plane state.
- During overlapping API traffic, check `request admission` to see how many
  requests are running and how many are queued behind `--max-active-requests`.
  JSON status also includes admitted, completed, cancelled, pressure, and
  whether admission is currently paused by the memory guard.
- During overlapping apps or CLI processes, check `machine admission` to see
  aggregate work across Studio, Raycast, terminals, scripts, agents, and API
  servers. This is distinct from the per-server request queue.
- Use `memory` to see the active memory guard tier, pressure level, current
  resident usage, host-available estimate, and computed soft/hard/ceiling limits.
- Use `process` to inspect server uptime, sampled process CPU, macOS
  thermal/low-power state, and Metal allocation/working-set values. Metal
  allocation is not presented as a GPU-utilization percentage.
- Use `continuous batching` and `prefix KV reuse` to see which scheduler/cache
  features are actually enabled instead of assuming they are active.
- Use `cache stats` to check aggregate prefix hits, reused tokens, and batched
  decode steps across loaded models, including same-position versus
  variable-position decode batches when the selected engine can report them.
- Use `traffic stats` to inspect completed request counts, generated tokens,
  and average load/prefill/decode timings while cache and batching flags are on.
- Use `sidecar residency` to distinguish an unloaded lane, a `resident (not
  ready)` generator, and a ready image, speech, or transcription runtime. The
  human `loaded models` line includes only residents whose optional readiness is
  not explicitly false.

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
- Machine admission is system-wide for cooperative `mere.run` processes. Small
  work uses one weighted permit, standard work uses two, and video/DS4-class
  work takes the full machine capacity. Capacity scales from one permit below
  48 GiB to two below 96 GiB, four below 192 GiB, and six thereafter.
- Admission state contains command families and process IDs only. A process
  exit removes its ticket; a crash is pruned on the next queue or status read.
- A sidecar can be resident before it is ready because its generator object is
  created before the first operation finishes model loading. Check the per-lane
  state rather than treating residency alone as request readiness.
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
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Support/MachineInferenceAdmission.swift
- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Support/ModelInventory.swift
