# API Serve

## Purpose

Start a local OpenAI-compatible HTTP server for chat completions, native text
embeddings, image generation/editing, text-to-speech, and speech-to-text. Use
this when another tool, editor, UI, or agent needs to call mere.run over HTTP.

## Required Models

Supported engines:

- `text-code`: GGUF code models such as `text-code-qwen3`,
  `text-code-north-mini`, and `text-agent-ornith-35b`.
- `text-chat-gemma4`: Gemma text chat models, including `text-chat-gemma4-12b`.
- `vision-chat-gemma4-12b`: Gemma 4 12B vision chat over the Gemma4 API serving engine.
- `text-chat-q36`: Qwen-family serving engine; defaults to `text-chat-q36-nano`
  and also serves Qwen-family agent experiments such as `text-agent-ornith-9b`
  and `text-agent-ornith-35b-mlx`.
- `text-chat-lfm2`: LFM2 serving engine; defaults to `text-chat-lfm25-a1b-8bit`.
- `text-chat-deepseek-v4-flash`: DeepSeek V4 Flash via the bundled DS4 server.
- `text-chat-klein`: local Klein/MeBot chat path when installed.
- `text-embed-qwen3-0.6b`: native Qwen3 embedding model served through
  `/v1/embeddings` for RAG and semantic search.
- Image generation: any installed `image-*` generation model, such as
  `image-zimage-nano`.
- Text-to-speech: `speech-tts-qwen3-nano`.
- Speech-to-text: `speech-asr-parakeet` or `speech-asr-qwen3`.

## Install And Check

```bash
mere.run model capabilities
mere.run model pull text-agent-deepseek-v4-flash
mere.run model pull image-zimage-nano
mere.run model pull speech-tts-qwen3-nano
mere.run model pull speech-asr-parakeet
mere.run model runtime get text-chat-gemma4
mere.run api serve --help
mere.run status
```

## Parameters

- `--port`, `-p`: listen port, default `8080`.
- `--host`: bind host, default `127.0.0.1`.
- `--model`, `-m`, `--model-path`: model path or engine-specific model root.
- `--engine`: `text-code`, `text-chat-klein`, `text-chat-gemma4`, `text-chat-q36`, `text-chat-lfm2`, or `text-chat-deepseek-v4-flash`.
- `--lora`: default LoRA adapter path for all requests.
- `--api-key`: bearer token, also read from `MERERUN_API_KEY`.
- `--rate-limit-per-minute`: global OpenAI-compatible request limit.
- `--max-active-requests`: fair FIFO admission limit for concurrent chat completions; default `1`.
- `--memory-guard`: runtime memory guard tier, default `balanced`. Accepted
  values are `off`, `safe`, `balanced`, `aggressive`, and `custom`.
- `--memory-guard-custom-ceiling-gb`: custom process-memory ceiling in GiB.
  Requires `--memory-guard custom`.
- `--context-size`: context limit used for request validation and native prompt truncation.
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
  default generation limits without starting the server. TTL unloads idle
  models during runtime pool operations, and pinned models skip automatic
  TTL/LRU eviction. `--memory-guard` computes tiered soft/hard ceilings from
  process resident memory and host memory headroom. Under elevated pressure,
  chat admission pauses extra concurrent prefills and the pool evicts the
  least-recently-used idle unpinned model; under critical pressure, it evicts
  every idle unpinned model. Active requests are never evicted.
- Test `/v1/chat/completions` after status shows the expected served model.
- Test `/v1/embeddings` with `text-embed-qwen3-0.6b` when wiring a RAG client.
- Test `/v1/images/generations` with `image-zimage-nano` when wiring an image client.
- Test `/v1/images/edits` with multipart `image` uploads when wiring image editing.
- Test `/v1/audio/speech` with `speech-tts-qwen3-nano` when wiring TTS.
- Test `/v1/audio/transcriptions` with `speech-asr-parakeet` when wiring STT.
- Request `model` resolves by runtime alias, then curated catalog id, then the
  startup default from `--engine`/`--model`.
- `/v1/models` returns installed API-capable chat catalog ids, aliases, and
  installed native embedding, image, TTS, and ASR sidecar model ids.
- Requests are admitted through a fair FIFO queue. The default
  `--max-active-requests 1` preserves serialized local inference while making
  queue depth visible in `/runtime/status`. Queued client cancellations are
  removed from the FIFO instead of running later.
- Gemma4 and Qwen-family chat use chunked prefill with cancellation/progress checkpoints.
  This improves long-prompt observability without arbitrary batching.
- Gemma4 uses in-memory prefix KV reuse by default; set
  `MERERUN_GEMMA4_PREFIX_KV_CACHE=0` for a baseline. `/runtime/status` reports
  entries, hits, and reused tokens when a Gemma4 model is loaded. The cache
  records chunk boundaries plus the stable chat prefix before the final message
  when token prefixes match exactly.
- Qwen-family chat uses text-only in-memory prefix KV reuse by default; set
  `MERERUN_Q35_PREFIX_KV_CACHE=0` for a baseline. Vision prompts are excluded
  from reuse, and text-only requests use the same stable chat-prefix checkpoint
  rule as Gemma4.
- Gemma4 and Qwen-family chat can opt into decode batching with
  `MERERUN_GEMMA4_CONTINUOUS_BATCHING=1` or
  `MERERUN_Q35_CONTINUOUS_BATCHING=1`; use `--max-active-requests` above `1` to
  allow overlapping rows, and `/runtime/status` reports actual batched decode
  steps. Gemma4 full-attention rows stay same-position because that engine still
  uses scalar RoPE/cache offsets; Qwen-family full-attention rows use row-offset-aware
  ragged KV caches, and Qwen-family linear rows use typed recurrent state, so compatible
  Qwen-family rows can batch across decode positions. The scheduler services the
  earliest decode position first, batching compatible rows there or advancing a
  single lower-offset row until it can join a compatible batch.
- `/runtime/status` aggregates prefix hits, reused tokens, and batched decode
  steps across loaded models under `cacheStats`; it also reports completed chat
  requests, generated tokens, and average load/prefill/decode timings under
  `benchmarkStats` so these experiments stay measured.
- DS4 raw-proxies the complete OpenAI chat request to `ds4-server`.
- Native engines reject unsupported OpenAI fields explicitly instead of silently dropping them.
- `text-code` maps OpenAI `stop` sequences into native generation stops.
- Function `tool_choice` values are accepted for native tool-capable engines;
  specific function choices narrow the advertised tools to the named function.
- `/v1/embeddings` accepts OpenAI-compatible string or string-array `input`
  payloads and returns float embeddings from `text-embed-qwen3-0.6b`.
- `/v1/images/generations` accepts `prompt`, `model`, `size`, `n`, and
  `response_format`. It supports `n=1`, returns base64 PNG JSON by default,
  and can return a local `file://` URL when `response_format` is `url`.
- `/v1/images/edits` accepts multipart `image` or Open WebUI-style `image[]`,
  optional `mask`, `prompt`, `model`, `size`, `n`, and `response_format`. It
  uses the same native image runtime with `inputImage` conditioning and accepts
  local extensions such as `strength`, `seed`, `steps`, and `guidance_scale`.
  Masks are accepted for client compatibility; current native edit models use
  whole-image conditioning rather than strict masked inpainting.
- `/v1/audio/speech` accepts `input`, `model`, `voice`, `speed`, and
  `response_format`. It returns WAV by default and can transcode to `mp3`,
  `opus`, `aac`, or `flac` when `ffmpeg` is available. OpenAI model names such
  as `tts-1` map to the local default.
- `/v1/audio/transcriptions` accepts multipart `file`, `model`, `language`,
  `task`, and `response_format`. OpenAI model names such as `whisper-1` map to
  `speech-asr-parakeet`; response formats are `json`, `text`, `verbose_json`,
  `srt`, and `vtt`.
- `vision-chat-gemma4-12b` accepts one OpenAI image content part per message through `/v1/chat/completions`; use a file path, `file://` URL, or base64 data URL because the local runtime does not fetch remote images.
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
mere.run model pull vision-chat-gemma4-12b
mere.run api serve --engine text-chat-gemma4 --model vision-chat-gemma4-12b --port 11434
```

```bash
mere.run model pull text-chat-lfm25-a1b-8bit
mere.run api serve --engine text-chat-lfm2 --port 11434
```

```bash
mere.run model pull text-agent-ornith-9b
mere.run api serve --engine text-chat-q36 --model text-agent-ornith-9b --port 11434
```

```bash
export MERERUN_API_KEY=change-me
mere.run api serve --host 0.0.0.0 --api-key "$MERERUN_API_KEY"
```

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "text-embed-qwen3-0.6b",
    "input": ["mere.run native embeddings", "local RAG"]
  }'
```

```bash
curl http://localhost:8080/v1/images/generations \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "image-zimage-nano",
    "prompt": "a compact local AI workstation in morning light",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

```bash
curl http://localhost:8080/v1/images/edits \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -F model=image-zimage-nano \
  -F prompt="make the desk dusk-lit while preserving the layout" \
  -F size=1024x1024 \
  -F response_format=b64_json \
  -F image=@input.png
```

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -H "Content-Type: application/json" \
  --output speech.wav \
  --data '{
    "model": "speech-tts-qwen3-nano",
    "input": "mere.run is serving native speech.",
    "voice": "nova",
    "response_format": "wav"
  }'
```

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -F model=speech-asr-parakeet \
  -F response_format=json \
  -F file=@speech.wav
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
- RAG client cannot embed: confirm `/v1/models` includes `text-embed-qwen3-0.6b`
  and call `/v1/embeddings` with `encoding_format` omitted or set to `float`.
- Image client cannot render: use `response_format` `b64_json` unless the
  client can read local `file://` image URLs.
- TTS client expects MP3: set `response_format` to `mp3`, or force `wav` when
  you want the native Qwen3-TTS output without ffmpeg transcoding.
- STT client fails upload: send `multipart/form-data` with a `file` part and a
  supported local ASR model id.

## Sources

- https://github.com/sawfwair/mere-run/blob/main/Sources/MereRunCLI/Commands/APIServeCommand.swift
- https://platform.openai.com/docs/api-reference/chat
- https://platform.openai.com/docs/api-reference/images/create
- https://platform.openai.com/docs/api-reference/audio/createSpeech
- https://platform.openai.com/docs/api-reference/audio/createTranscription
- https://ai.google.dev/gemma/docs/core/prompt-structure
- https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit
