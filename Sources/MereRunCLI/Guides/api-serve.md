# API Serve

## Purpose

Start a local OpenAI-compatible HTTP server for chat completions. Use this when another tool, editor, or agent needs to call mere.run over HTTP.

## Required Models

Supported engines:

- `text-code`: default GGUF code model, usually `text-code-qwen3`.
- `text-chat-gemma4`: Gemma text chat models.
- `text-chat-q35`: Qwen3.5 text chat models.
- `text-chat-deepseek-v4-flash`: DeepSeek V4 Flash via the bundled DS4 server.
- `text-chat-klein`: local Klein/MeBot chat path when installed.

## Install And Check

```bash
mere.run model capabilities
mere.run model pull text-agent-deepseek-v4-flash
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
- `--context-size`: context limit.
- `--kv-bits`, `--kv-quant-scheme`, `--kv-group-size`, `--quantized-kv-start`: Gemma4 KV cache controls. Serving `text-chat-gemma4-turbo` defaults to 4-bit TurboQuant KV cache from token 0; explicit flags override that.

## Usage Patterns

- Keep loopback binds for local-only tools.
- For non-loopback hosts, always set `MERERUN_API_KEY` or pass `--api-key`.
- Choose the engine first, then the model path/id.
- Use `mere.run status` as the quick `/health` plus `/v1/models` check.
- Test `/v1/chat/completions` after status shows the expected served model.
- DS4 raw-proxies the complete OpenAI chat request to `ds4-server`.
- Native engines reject unsupported OpenAI fields explicitly instead of silently dropping them.
- Use `stream_options.include_usage` when a client expects the OpenAI streaming usage chunk.

## Examples

```bash
mere.run api serve --engine text-code --port 8080
```

```bash
mere.run model pull text-chat-gemma4
mere.run api serve --engine text-chat-gemma4 --port 11434
```

```bash
export MERERUN_API_KEY=change-me
mere.run api serve --host 0.0.0.0 --api-key "$MERERUN_API_KEY"
```

## Iteration Tips

- Use `mere.run status` before connecting an editor.
- Start with one client and default rate limit.
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
