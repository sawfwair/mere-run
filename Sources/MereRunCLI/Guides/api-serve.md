# API Serve

## Purpose

Start a local OpenAI-compatible HTTP server for chat completions. Use this when another tool, editor, or agent needs to call mere.run over HTTP.

## Required Models

Supported engines:

- `text-code`: default GGUF code model, usually `text-code-qwen3`.
- `text-chat-gemma4`: Gemma text chat models.
- `text-chat-q35`: Qwen3.5 text chat models.
- `text-chat-q36-nano`: Qwen3.6 35B-A3B OptiQ chat weights served through the Q35-family engine.
- `text-chat-deepseek-v4-flash`: DeepSeek V4 Flash via the bundled DS4 server.
- `text-chat-klein`: local Klein/MeBot chat path when installed.

## Install And Check

```bash
mere.run model capabilities
mere.run model pull text-agent-deepseek-v4-flash
mere.run model runtime get text-chat-gemma4
mere.run api serve --help
mere.run status
```

## Parameters

- `--port`, `-p`: listen port, default `8080`.
- `--host`: bind host, default `127.0.0.1`.
- `--model`, `-m`, `--model-path`: model path or engine-specific model root.
- `--engine`: `text-code`, `text-chat-klein`, `text-chat-gemma4`, `text-chat-q35`, or `text-chat-deepseek-v4-flash`.
- `--lora`: default LoRA adapter path for all requests.
- `--api-key`: bearer token, also read from `MERERUN_API_KEY`.
- `--rate-limit-per-minute`: global chat completions limit.
- `--max-active-requests`: fair FIFO admission limit for concurrent chat completions; default `1`.
- `--context-size`: context limit.
- `--kv-bits`, `--kv-quant-scheme`, `--kv-group-size`, `--quantized-kv-start`:
  Gemma4 KV cache controls. Serving `text-chat-gemma4-turbo` defaults to the
  existing 4-bit affine TurboQuant KV cache from token 0; explicit flags
  override that. `--kv-quant-scheme polar --kv-bits 2` enables the experimental
  packed PolarKV path for memory-pressure and long-context synthetic decode
  testing. Per-model runtime settings can also set `kvCacheMode` to `default`,
  `polar2`, or conservative `auto`.

## Usage Patterns

- Keep loopback binds for local-only tools.
- For non-loopback hosts, always set `MERERUN_API_KEY` or pass `--api-key`.
- Choose the engine first, then the model path/id.
- Use `mere.run status` as the quick `/health` plus `/v1/models` check.
- Use `mere.run model runtime set` to configure aliases, pinning, TTL, and
  default generation limits without starting the server.
- Test `/v1/chat/completions` after status shows the expected served model.
- Request `model` resolves by runtime alias, then curated catalog id, then the
  startup default from `--engine`/`--model`.
- `/v1/models` returns installed API-capable catalog ids plus aliases.
- Requests are admitted through a fair FIFO queue. The default
  `--max-active-requests 1` preserves serialized local inference while making
  queue depth visible in `/runtime/status`. Queued client cancellations are
  removed from the FIFO instead of running later.
- Gemma4 and Q35 use chunked prefill with cancellation/progress checkpoints.
  This improves long-prompt observability without arbitrary batching.
- Gemma4 can opt into in-memory prefix KV reuse with
  `MERERUN_GEMMA4_PREFIX_KV_CACHE=1`; `/runtime/status` reports entries, hits,
  and reused tokens when a Gemma4 model is loaded. The cache records chunk
  boundaries plus the stable chat prefix before the final message when token
  prefixes match exactly.
- Q35 can opt into text-only in-memory prefix KV reuse with
  `MERERUN_Q35_PREFIX_KV_CACHE=1`; vision prompts are excluded from reuse, and
  text-only requests use the same stable chat-prefix checkpoint rule as Gemma4.
- Gemma4 and Q35 can opt into decode batching with
  `MERERUN_GEMMA4_CONTINUOUS_BATCHING=1` or
  `MERERUN_Q35_CONTINUOUS_BATCHING=1`; use `--max-active-requests` above `1` to
  allow overlapping rows, and `/runtime/status` reports actual batched decode
  steps. Gemma4 full-attention rows stay same-position because that engine still
  uses scalar RoPE/cache offsets; Q35 full-attention rows use row-offset-aware
  ragged KV caches, and Q35 linear rows use typed recurrent state, so compatible
  Q35 rows can batch across decode positions. The scheduler services the
  earliest decode position first, batching compatible rows there or advancing a
  single lower-offset row until it can join a compatible batch.
- `/runtime/status` aggregates prefix hits, reused tokens, and batched decode
  steps across loaded models under `cacheStats`; it also reports completed chat
  requests, generated tokens, and average load/prefill/decode timings under
  `benchmarkStats` so these experiments stay measured.
- DS4 raw-proxies the complete OpenAI chat request to `ds4-server`.
- Native engines reject unsupported OpenAI fields explicitly instead of silently dropping them.
- Use `stream_options.include_usage` when a client expects the OpenAI streaming usage chunk.

## Examples

```bash
mere.run api serve --engine text-code --port 8080
```

```bash
mere.run model pull text-chat-gemma4
mere.run model runtime set text-chat-gemma4 --alias chat-default --max-tokens 1024
mere.run api serve --engine text-chat-gemma4 --port 11434
```

```bash
export MERERUN_API_KEY=change-me
mere.run api serve --host 0.0.0.0 --api-key "$MERERUN_API_KEY"
```

## Iteration Tips

- Use `mere.run status` before connecting an editor.
- Start with one client and default rate limit.
- Keep `--max-active-requests 1` unless you have measured that the selected
  engine and machine benefit from overlapping requests.
- For Gemma memory pressure, test KV quantization on a local prompt before long sessions.

## Troubleshooting

- Non-loopback bind rejected: set `--api-key` or `MERERUN_API_KEY`.
- Client cannot connect: run `mere.run status --host <host> --port <port>`, then confirm firewall settings.
- Wrong model family: match `--engine` to the model path.
- Unsupported OpenAI field: choose a compatible engine or remove the field named in the `invalid_request_error`.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/APIServeCommand.swift
- https://platform.openai.com/docs/api-reference/chat
- https://ai.google.dev/gemma/docs/core/prompt-structure
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B
