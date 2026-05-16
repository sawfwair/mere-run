# Local API Server

This page covers `mere.run api serve`, the local API surface exposed by the package.

## Public surface

- `mere.run api serve`
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

In another terminal, confirm that the server is reachable and which model it
reports:

```bash
swift run mere.run status
```

Network-exposed example:

```bash
export MERERUN_API_KEY=change-me
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --host 0.0.0.0 \
  --port 11434 \
  --api-key "$MERERUN_API_KEY" \
  --rate-limit-per-minute 120
```

## Design notes

- the API server follows the same model-resolution and model-store rules as the
  rest of the CLI
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
- `text-chat-q35`: accepts function tools and one image content part per
  message.
- `text-chat-klein`: supports `response_format: {"type":"json_object"}` with
  local JSON retry behavior.
- `text-code`: accepts plain text chat requests and rejects tools, images,
  reasoning controls, logprobs, seed, stop sequences, and structured outputs
  with explicit errors.

Streaming responses only emit assistant content tokens. Local progress labels
stay in logs/stderr, and `stream_options.include_usage` adds the final usage
chunk before `[DONE]`.

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
