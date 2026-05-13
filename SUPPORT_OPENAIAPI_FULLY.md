# Support The OpenAI Chat API Fully

## Problem

`mere.run api serve` is advertised as OpenAI-compatible, but the current generic
chat path only models the subset required by the first local text engines:

- `model`
- `messages`
- `temperature`
- `top_p`
- `max_tokens`
- `stream`
- a local-only `lora` extension

That subset is enough for simple `/v1/chat/completions` clients, but it is not
enough for coding agents and newer OpenAI-compatible clients. Those clients send
fields such as tools, tool choice, reasoning effort, structured outputs,
developer messages, usage-in-streaming options, and provider-specific thinking
controls. Dropping those fields changes behavior silently.

The DS4 integration exposed this sharply: Pi sent a richer OpenAI-style request,
but the generic wrapper decoded it into `ChatRequest`, lost DS4/Pi-specific
fields, then streamed a local progress label as if it were model output.

## Current State

DS4 now has the right short-term behavior: when `--engine
text-chat-deepseek-v4-flash` is selected, `mere.run api serve` starts or attaches
to `ds4-server` and proxies the raw `/v1/chat/completions` body to DS4. That
preserves native DS4 support for:

- streaming Server-Sent Events
- `think` / thinking controls
- `reasoning_effort`
- tool fields understood by DS4
- the exact model id supplied by the client
- upstream error and chunk shapes

The generic server path now decodes a broader typed Chat Completions request
shape and validates it against per-engine capabilities before generation. DS4
still stays on raw proxy mode because its server already implements the richer
OpenAI-compatible protocol.

Native-engine support added here:

- `developer` messages normalize to local system instructions
- text and image content parts decode without collapsing the whole request
- `max_completion_tokens` is accepted alongside `max_tokens`
- function tools map into local `ToolDefinition` for Gemma4 and Q35
- `response_format: {"type":"json_object"}` maps to the Klein JSON retry path
- `stream_options.include_usage` emits the OpenAI-style usage chunk before
  `[DONE]`
- high-impact unsupported fields fail with `invalid_request_error` instead of
  disappearing

Still open:

- per-token tool-call delta streaming from native engines
- strict JSON schema support
- native seed, stop-sequence, logprobs, and penalty support
- richer `/v1/models` capability metadata

## Goal

Make `mere.run api serve` a reliable local OpenAI Chat Completions server for
supported text engines.

Compatibility should mean:

- accept the common OpenAI `/v1/chat/completions` request shape without rejecting
  or silently corrupting valid client input
- preserve fields an engine can use
- reject unsupported fields clearly when an engine cannot honor them
- stream model text, tool calls, usage, and errors in OpenAI-compatible SSE
  shapes
- avoid leaking local progress/status messages into assistant content

## Request Surface To Model

Add typed request coverage for at least:

- messages with `system`, `developer`, `user`, `assistant`, and `tool` roles
- string content, text content parts, image content parts where the engine
  supports vision, and nullable assistant content
- assistant `tool_calls`
- tool response messages with `tool_call_id`
- `tools`
- `tool_choice`
- `parallel_tool_calls`
- `response_format`
- `stream_options`, including usage in streaming
- `stop`
- `seed`
- `presence_penalty`
- `frequency_penalty`
- `logprobs`
- `top_logprobs`
- `reasoning_effort`
- `max_completion_tokens` alongside legacy `max_tokens`
- provider thinking controls that are common in OpenAI-compatible local servers

Unknown fields should be preserved for proxy-capable engines and explicitly
ignored or rejected for engines that use the typed local `ChatGenerator` path.

## Engine Capability Contract

Each serving engine should declare what it supports. The API server can then
make compatibility decisions before generation starts.

Useful capability flags:

- supports raw proxying
- supports tools
- supports tool choice
- supports developer role
- supports structured outputs
- supports reasoning effort
- supports max completion tokens
- supports usage in streaming
- supports vision content parts
- supports strict JSON schema
- supports stop sequences
- supports seed

This should feed both request validation and `/v1/models` metadata where a
client benefits from provider capability hints.

## Streaming Contract

Streaming needs a stricter rule than "anything emitted during generation is a
token."

Required behavior:

- local progress belongs on stderr/logs, not in SSE assistant deltas
- token deltas become `chat.completion.chunk` events
- tool-call deltas retain ids, names, and argument fragments
- final chunks include `finish_reason`
- optional usage chunks are emitted only when requested and supported
- errors during generation are emitted as compatible error payloads and close
  the stream cleanly

For non-token-streaming engines, either buffer and send a single assistant delta
or return non-streaming output. Do not invent fake token content.

## Migration Plan

1. Expand `OpenAIChatRequest` into a full typed OpenAI request model with
   lossless unknown-field handling where practical. Done for the common Chat
   Completions fields and preserved unknown top-level fields.
2. Introduce an engine capability descriptor and make request validation depend
   on it. Done for the local engines and DS4 raw proxy.
3. Split "local progress" from "assistant output" in the `ChatGenerator`
   interface, or add an explicit token-streaming interface for engines that can
   stream true content.
4. Keep DS4 on raw proxy mode because its server already implements the richer
   OpenAI-compatible protocol.
5. For each native engine, map supported OpenAI fields into the local generation
   request and reject unsupported high-impact fields with a clear
   `invalid_request_error`.
6. Add golden request/response tests for common clients: Pi, OpenAI SDK,
   Continue/Cursor-style streaming, tool calls, structured JSON, and plain curl.
7. Update `docs/runtime/api-server.md`, `docs/cli.md`, and the provider
   extension docs with the exact compatibility guarantees.

## Done When

- Pi can run setup conversations without local progress text appearing as model
  output.
- A tool-capable OpenAI client can send tools and receive valid tool-call
  chunks from every engine that advertises tool support.
- Unsupported fields fail loudly per engine instead of disappearing.
- Streaming and non-streaming responses share the same final semantic content.
- `/v1/models` and docs accurately describe each engine's compatibility level.
