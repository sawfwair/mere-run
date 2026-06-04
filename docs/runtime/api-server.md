# Local API Server

This page covers `mere.run api serve`, the local API surface exposed by the package.

## Public surface

- `mere.run api serve`
- `mere.run model runtime get`
- `mere.run model runtime set`
- `mere.run status`

## What it is for

The API server lets you expose supported local engines through a local process
instead of shelling out to the CLI for every request. It is useful for:

- local automation
- editor tooling
- simple local integrations
- experimenting with the runtime through HTTP

It is not a hosted-service or relay layer. This repo keeps the server local and
package-scoped.

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/APIServeCommand.swift`

### Supporting stack

- `Sources/MereRunCLI/Support/`
- `Hummingbird` package dependency declared in `Package.swift`

## Example

```bash
swift run mere.run api serve --engine text-chat-gemma4
```

For the LiquidAI LFM2.5 MLX 8-bit model:

```bash
swift run mere.run model pull text-chat-lfm25-a1b-8bit
swift run mere.run api serve --engine text-chat-lfm2
```

In another terminal, confirm that the server is reachable and which model it
reports:

```bash
swift run mere.run status
```

Optional per-model defaults live with the active model store:

```bash
swift run mere.run model runtime set text-chat-gemma4 \
  --alias chat-default \
  --pinned \
  --ttl-seconds 3600 \
  --max-context-tokens 8192 \
  --max-tokens 1024 \
  --temperature 0.6 \
  --top-p 0.9
```

Network-exposed example:

```bash
export MERERUN_API_KEY=change-me
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --host 0.0.0.0 \
  --port 11434 \
  --api-key "$MERERUN_API_KEY" \
  --rate-limit-per-minute 120 \
  --max-active-requests 1
```

## Design notes

- the API server follows the same model-resolution and model-store rules as the
  rest of the CLI
- `ManagedModelCatalog` is the API model authority; the server does not scan
  arbitrary model folders as request-addressable models
- `--engine` and `--model` still define and preload the startup default, while
  each chat request can select another installed API-capable catalog model with
  the OpenAI `model` field
- chat completions pass through a fair FIFO request admission actor; the default
  `--max-active-requests 1` preserves serialized local inference and exposes
  queue depth in status; queued client cancellations are removed from the FIFO
  instead of being admitted later
- Gemma4 and Qwen-family prefills are chunked with cancellation and progress checkpoints
  before decode; this is cooperative single-request prefill, not continuous
  batching
- Gemma4 has an opt-in in-memory prefix KV reuse prototype behind
  `MERERUN_GEMMA4_PREFIX_KV_CACHE=1`; `/runtime/status` reports cache entries,
  hits, and reused tokens when the Gemma4 model is loaded; the cache stores
  chunk boundaries plus the stable chat prefix before the final message when it
  is an exact token prefix, and pruning keeps that stable prefix ahead of
  ordinary chunk boundaries
- Qwen-family chat has an opt-in text-only prefix KV reuse prototype behind
  `MERERUN_Q35_PREFIX_KV_CACHE=1`; vision prompts are excluded because image
  embeddings alter the effective prefix; text-only requests use the same stable
  chat-prefix checkpoint and pruning rule as Gemma4
- Gemma4 and Qwen-family chat have opt-in decode batching prototypes behind
  `MERERUN_GEMMA4_CONTINUOUS_BATCHING=1` and
  `MERERUN_Q35_CONTINUOUS_BATCHING=1`; set `--max-active-requests` above `1` to
  allow overlapping rows, and `/runtime/status` reports actual batched decode
  steps instead of assuming the scheduler is active; Gemma4 full-attention rows
  remain same-position because that engine still uses scalar RoPE/cache offsets,
  while Qwen-family full-attention rows use row-offset-aware ragged KV caches and Qwen-family
  linear rows use typed recurrent state so compatible Qwen-family rows may batch across
  decode positions; the scheduler services the earliest decode position first,
  batching compatible rows there or advancing a single lower-offset row until it
  can join one
- Gemma4 has an experimental packed PolarKV path behind
  `--kv-quant-scheme polar --kv-bits 2`; use it for memory-pressure and
  long-context synthetic decode testing. It is not the default until checkpoint
  benchmarks prove the end-to-end model path.
- Gemma4 runtime settings can set `kvCacheMode` to `default`, `polar2`, or
  `auto`. `auto` keeps the default KV path below 1024 prompt tokens and switches
  to decode-deferred packed PolarKV at or above that threshold.
- `/runtime/status` aggregates prefix hits, reused tokens, and batched decode
  steps across loaded models under `cacheStats`; it also reports completed chat
  request counts, generated tokens, and average load/prefill/decode timings
  under `benchmarkStats`; SSD KV persistence remains unavailable until the
  in-memory counters justify it
- runtime settings are stored at
  `<active model store>/.mere-run/runtime-model-settings.json`
- `mere.run status` is the preferred quick check before wiring an editor or
  agent to a local server
- it is intentionally local-first
- it should not reintroduce relay, billing, or hosted-infrastructure concerns
- non-loopback binds require an API key, and the OpenAI-compatible chat route
  supports basic rate limiting
- chat requests must use `Content-Type: application/json`; browser-simple
  form/text posts are rejected before the request body is processed
- chat requests are validated before generation; `max_tokens`,
  `max_completion_tokens`, `temperature`, and `top_p` must stay within bounded
  ranges
- LoRA adapters are configured at server startup with `--lora`; request bodies
  cannot select local LoRA paths
- streaming and JSON error paths are sanitized so the local server does not
  reflect raw internal runtime details back to clients

## Runtime control endpoints

The control endpoints use the same bearer-token behavior as `/v1/models` and
`/v1/chat/completions`.

- `GET /runtime/status`: server health, pool entries, active request counts,
  request admission state, runtime capability flags, memory snapshot, settings
  path, aggregate cache stats, per-model prefix KV cache stats, per-model decode
  batching stats when enabled, and aggregate benchmark stats measured from
  completed native chat requests.
- `POST /runtime/models/{id}/load`: explicitly load an installed API-capable
  catalog model.
- `POST /runtime/models/{id}/unload`: unload a model; returns `409` while
  active requests are using it.
- `GET /runtime/models/{id}/settings`: read typed runtime defaults.
- `PATCH /runtime/models/{id}/settings`: replace typed runtime defaults.

`GET /v1/models` returns installed API-servable managed IDs plus configured
aliases. Missing catalog models fail with an OpenAI-style error that tells the
user to pull the model first.

## OpenAI chat compatibility

`POST /v1/chat/completions` accepts the common Chat Completions request shape:

- `system`, `developer`, `user`, `assistant`, and `tool` messages
- string content, text content parts, nullable assistant content, and image
  content parts when the selected engine supports vision
- assistant `tool_calls` and tool response messages
- `tools`, `tool_choice`, `parallel_tool_calls`
- `response_format`
- `stream_options.include_usage`
- `stop`, `seed`, penalties, logprobs, reasoning controls, and
  provider-thinking controls as typed request fields
- `max_completion_tokens` alongside legacy `max_tokens`

The server does not silently drop high-impact fields. Native engines either map
supported fields into `ChatRequest` or return an OpenAI-style
`invalid_request_error` before generation. Metadata-style fields such as
`metadata`, `user`, and `service_tier` are accepted as request context but do
not change local generation.

Engine compatibility:

- `text-chat-deepseek-v4-flash`: raw-proxies the original request body to
  `ds4-server`, preserving DS4's OpenAI-compatible behavior.
- `text-chat-gemma4`: accepts function tools and emits OpenAI tool-call
  responses when the model generates a tool call.
- `text-chat-q36-nano`: uses the Qwen-family serving engine with Qwen3.6
  35B-A3B OptiQ chat weights, accepts function tools, and accepts one image
  content part per message.
- `text-chat-lfm25-a1b-8bit`: uses the LFM2 serving engine with the
  LiquidAI LFM2.5 8B-A1B MLX 8-bit weights, accepts function tools, and rejects
  image content parts.
- `text-chat-klein`: supports `response_format: {"type":"json_object"}` with
  local JSON retry behavior.
- `text-code`: accepts plain text chat requests and rejects tools, images,
  reasoning controls, logprobs, seed, stop sequences, and structured outputs
  with explicit errors.

Streaming responses only emit assistant content tokens. Local progress labels
stay in logs/stderr, and `stream_options.include_usage` adds the final usage
chunk before `[DONE]`.

Fair FIFO request admission is part of the runtime pool now, and Gemma4/Qwen-family chat use
engine-specific chunked prefill checkpoints. Gemma4 and Qwen-family prefix KV reuse are
available as opt-in in-memory prototypes; Qwen-family reuse is limited to text-only
requests. Gemma4 and Qwen-family decode batching are also available as opt-in
prototypes: they merge typed cache rows for actual batched decode calls, then
split the rows back so each request keeps its own state. Gemma4 and Qwen-family
status reports same-position versus variable-position batched steps separately.
Qwen-family full-attention rows can batch across different decode positions through
row-offset-aware ragged KV caches, and Qwen-family linear rows can do the same when typed
linear cache state is compatible. Gemma4 full-attention rows still require
matching scalar offsets. SSD KV persistence remains later, measured work; use
`cacheStats` plus `benchmarkStats` to decide whether that experiment is worth
expanding on a real machine.

If you are working on this area, read [CLI and Runtime Internals](../internals/cli-and-runtime.md) after the command source.

## Troubleshooting

Start with:

```bash
swift run mere.run status
```

- `server: down` means nothing answered the configured `/health` URL.
- `loaded models: unavailable (requires API key)` means `/health` worked but
  `/v1/models` needs `--api-key` or `MERERUN_API_KEY`.
- A wrong loaded model usually means another server is already bound to that
  host and port.
