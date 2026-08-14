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
- `text-chat-laguna`: managed Laguna S 2.1 target with automatic DFlash or
  the released Laguna XS 2.1 target without DFlash; defaults to
  `text-chat-laguna-s-2-1`.
- `vision-chat-gemma4-12b`: Gemma 4 12B vision chat over the Gemma4 API serving engine.
- `text-chat-q36`: Qwen-family serving engine; defaults to `text-chat-q36-nano`
  and also serves `text-chat-bonsai-27b-1bit`, `text-chat-bonsai-27b-2bit`,
  plus Qwen-family agent experiments such as `text-agent-ornith-9b` and
  `text-agent-ornith-35b-mlx`.
- `text-chat-lfm2`: LFM2 serving engine; defaults to `text-chat-lfm25-a1b-8bit`
  and accepts `--model text-chat-lfm25-2.6b-4bit` for the compact dense model.
- `text-chat-deepseek-v4-flash`: DeepSeek V4 Flash via the bundled DS4 server.
- `text-chat-klein`: local Klein/MeBot chat path when installed.
- `text-embed-qwen3-0.6b`: native Qwen3 embedding model served through
  `/v1/embeddings` for RAG and semantic search.
- Image generation: any installed `image-*` generation model, such as
  `image-zimage-nano`.
- Image-to-3D: `image-3d-triposr` through a native uploaded-image mesh route.
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
- `--engine`: `text-code`, `text-chat-klein`, `text-chat-gemma4`, `text-chat-laguna`, `text-chat-q36`, `text-chat-lfm2`, or `text-chat-deepseek-v4-flash`.
- `--lora`: default LoRA adapter path for all requests.
- `--api-key`: bearer token, also read from `MERERUN_API_KEY`.
- `--rate-limit-per-minute`: global OpenAI-compatible request limit.
- `--max-active-requests`: fair FIFO admission limit for concurrent local
  inference across chat, embedding, image, TTS, and ASR requests; default `1`.
  Values above `1` automatically enable supported
  Gemma4, Laguna, Qwen-family, and LFM2 decode batching unless an engine-specific
  environment override forces the serial path.
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
  `polar2`, or conservative `auto`; Gemma4, Qwen-family, and LFM2 also accept
  explicit `affine4` or `affine8` as memory controls relative to full-precision KV. Gemma
  Turbo already defaults to a smaller 4-bit TurboQuant cache, so forcing affine
  8-bit can increase its KV residency; `default` restores the selected
  engine/model/server default, not necessarily full precision.

## Local VFX artifact boundary

The five artifact-producing VFX routes are loopback-only, even when the API
server binds `0.0.0.0`, a LAN address, or another non-loopback interface:

- `/v1/vision/geometry`
- `/v1/vision/geometry/multiview`
- `/v1/vision/image-to-3d`
- `/v1/vision/image-to-3d-multiview`
- `/v1/vision/depth-video`

On authenticated non-loopback servers, authentication is checked first and an
authenticated remote request then receives `403 permission_error` before its
body is collected or any model work begins. The policy uses the connected peer
socket and does not trust forwarded-address headers. `/v1/models` also omits
the MoGe, DA3, TripoSR, InstantMesh, and relative/metric VDA companion model IDs
for non-loopback clients.

These responses contain server-local `file:` URLs rather than downloadable
artifacts. Successful output directories remain available for one hour after
the response is created and are then removed automatically while the server is
running; copy any needed files before that TTL expires. Failed requests remove
their output directory immediately. The API intentionally provides no artifact
download route.

## Usage Patterns

- Keep loopback binds for local-only tools.
- For non-loopback hosts, always set `MERERUN_API_KEY` or pass `--api-key`.
- Use `mere.run api serve --preflight --json` to inspect host/port, auth,
  engine/model availability, LoRA paths, runtime limits, KV settings, companion
  model ids, and redacted follow-up actions before starting the server.
- Choose the engine first, then the model path/id.
- Use `mere.run status` as the quick `/health` plus `/v1/models` check.
- Use `mere.run model runtime set` to configure aliases, pinning, TTL, and
  default generation limits without starting the server. TTL unloads idle
  models during runtime pool operations, and pinned models skip automatic
  TTL/LRU eviction. `--memory-guard` computes tiered soft/hard ceilings from
  Darwin physical footprint (RSS elsewhere) and host memory headroom. Under
  elevated pressure,
  chat admission pauses extra concurrent prefills and the pool evicts the
  least-recently-used idle unpinned model; under critical pressure, it evicts
  every idle unpinned model. Active requests are never evicted.
- Test `/v1/chat/completions` after status shows the expected served model.
- Test `/v1/embeddings` with `text-embed-qwen3-0.6b` when wiring a RAG client.
- Test `/v1/images/generations` with `image-zimage-nano` when wiring an image client.
- Test `/v1/images/edits` with multipart `image` uploads when wiring image editing.
- Test `/v1/vision/image-to-3d` with a multipart `image` upload when wiring
  single-image object reconstruction.
- Test `/v1/vision/image-to-3d-multiview` with exactly four or six multipart
  `image[]` uploads when wiring InstantMesh reconstruction.
- Test `/v1/audio/speech` with `speech-tts-qwen3-nano` when wiring TTS.
- Test `/v1/audio/transcriptions` with `speech-asr-parakeet` when wiring STT.
- Request `model` resolves by runtime alias, then curated catalog id, then the
  startup default from `--engine`/`--model`.
- `/v1/models` returns installed API-capable chat catalog ids, aliases, and
  installed native sidecars. Mere-specific additive fields describe each
  model's task, tool-call and reasoning support, thinking levels, input/output
  modalities, context/output limits, and OpenAI compatibility dialect so agent
  harnesses do not have to guess from an id.
- Managed-model metadata comes from the typed catalog API profile. The running
  server overlays aliases and configured context/output defaults; explicit
  uncataloged model paths use a conservative profile for their serving engine.
- The embedding, image/image-edit, TTS, and ASR endpoints each retain one
  bounded most-recently-used runtime. Their autonomous idle timers default to 300
  seconds and re-read managed `pinned`/`ttlSeconds` settings while idle. Active
  and queued work is never evicted. The special `qwen-image-edit` repository ID
  is resident but currently uses default lifecycle settings because it is not a
  `model runtime` target.
- `/runtime/status` reports sidecar residency separately from readiness.
  `loaded` remains the compatibility field for a resident generator;
  `ready: false` means its first operation is still loading or failed. The human
  status formatter prints `resident (not ready)` for that state.
- Cold sidecar loads are exclusive across lanes. Catalog or local-directory
  size estimates are projected against the hard memory guard; the server first
  releases eligible idle residents and returns a memory-pressure error rather
  than starting a load that still lacks headroom.
- The HTTP `/runtime/models/{id}` load, unload, and settings operations address
  the chat/text pool only. Configure managed sidecar pinning and TTL with
  `mere.run model runtime set`; sidecars remain visible in `/v1/models` and
  `/runtime/status`.
- Every local inference request is admitted through a fair FIFO queue. The
  default `--max-active-requests 1` preserves serialized inference across text
  and allocation-heavy media paths while making queue depth visible in
  `/runtime/status`. Queued client cancellations are removed from the FIFO
  instead of running later. Raising the limit is an explicit throughput and
  unified-memory tradeoff. Explicit runtime model load/unload maintenance uses
  the same admission queue so it cannot race default-serialized inference.
- Each API-server process also holds a weighted machine-wide reservation. This
  coordinates the server with direct CLI calls and other server instances
  without disabling its resident model pool, prefix reuse, sidecar reuse,
  continuous batching, or `--max-active-requests` concurrency. Ordinary
  servers use a standard reservation; DeepSeek V4 Flash servers reserve the
  full machine capacity.
- Gemma4, Qwen-family, and LFM2 chat use chunked prefill with
  cancellation/progress checkpoints. This improves long-prompt observability
  without turning prefill itself into continuous batching.
- Gemma4 uses in-memory prefix KV reuse by default; set
  `MERERUN_GEMMA4_PREFIX_KV_CACHE=0` for a baseline. `/runtime/status` reports
  entries, hits, and reused tokens when a Gemma4 model is loaded. The cache
  records chunk boundaries plus the stable chat prefix before the final message
  when token prefixes match exactly.
- Qwen-family chat uses text-only in-memory prefix KV reuse by default; set
  `MERERUN_Q35_PREFIX_KV_CACHE=0` for a baseline. Vision prompts are excluded
  from reuse, and text-only requests use the same stable chat-prefix checkpoint
  rule as Gemma4.
- LFM2 chat uses in-memory prefix KV reuse by default; set
  `MERERUN_LFM2_PREFIX_KV_CACHE=0` for a baseline. Exact prompts and the stable
  conversation prefix before the final message are retained; intermediate
  prefill chunks are not cloned. Checkpoints fork both attention KV and
  short-convolution state.
- Gemma4, Qwen-family, and LFM2 decode batching engages automatically when
  `--max-active-requests` is above `1`. The corresponding
  `MERERUN_GEMMA4_CONTINUOUS_BATCHING`,
  `MERERUN_Q35_CONTINUOUS_BATCHING`, and
  `MERERUN_LFM2_CONTINUOUS_BATCHING` variables can force the implementation on
  or force the serial path. `/runtime/status` reports actual batched decode
  steps. Gemma4 full-attention rows stay same-position because that engine still
  uses scalar RoPE/cache offsets. Qwen-family and LFM2 full-attention rows use
  row-offset-aware ragged KV caches; Qwen-family linear attention and LFM2
  short-convolution layers use typed recurrent state, so compatible rows may
  batch across decode positions. The scheduler services the earliest decode
  position first, batching compatible rows there or advancing one lower-offset
  row until it can join a compatible batch.
- `/runtime/status` aggregates prefix hits, reused tokens, and batched decode
  steps across loaded models under `cacheStats`; it also reports completed chat
  requests, generated tokens, and average load/prefill/decode timings under
  `benchmarkStats` so these experiments stay measured.
- DS4 raw-proxies the complete OpenAI chat request to `ds4-server`.
- Native engines reject unsupported OpenAI fields explicitly instead of silently dropping them.
- `text-code` maps OpenAI `stop` sequences into native generation stops.
- Function `tool_choice` values are accepted for native tool-capable engines;
  specific function choices narrow the advertised tools to the named function.
- Native Gemma and Qwen-family chat engines accept
  `response_format: {"type":"json_object"}`. JSON-object generation uses a
  token-level prefix grammar, forces thinking off, and permits EOS only after
  the root object closes. Qwen-family requests bypass MTP, continuous batching,
  and pipelined decoding in this mode. `json_schema` and strict structured
  output are not supported.
- `/v1/embeddings` accepts OpenAI-compatible string or string-array `input`
  payloads and returns float embeddings from `text-embed-qwen3-0.6b`. A request
  may contain up to 256 texts and 2 MiB of UTF-8 content; inference truncates
  each row to 8,192 tokens and packs rows into bounded padded-token batches.
- `/v1/images/generations` accepts `prompt`, `model`, `size`, `n`, and
  `response_format`. It supports `n=1`, returns base64 PNG JSON by default,
  and can return a local `file://` URL when `response_format` is `url`. Width
  and height must each be a multiple of 16 from 16 through 4,096 pixels, with
  at most 4,194,304 total pixels per image. Explicit `steps` must be from 1
  through 100.
- `/v1/images/edits` accepts multipart `image` or Open WebUI-style `image[]`,
  optional `mask`, `prompt`, `model`, `size`, `n`, and `response_format`. It
  uses the same native image runtime with `inputImage` conditioning and accepts
  local extensions such as `strength`, `seed`, `steps`, and `guidance_scale`.
  The same dimension, total-pixel, and step limits apply. Masks are accepted for
  client compatibility; current native edit models use whole-image
  conditioning rather than strict masked inpainting.
- `/v1/vision/image-to-3d` accepts one uploaded `image` plus the managed
  `image-3d-triposr` id. Server-owned OBJ, PLY, GLB, and manifest artifacts are
  returned as local URLs with byte counts and SHA-256 values. Request fields
  cannot select client filesystem input, output, or checkpoint paths. The
  authoritative run `manifest` records every reconstruction control and exact
  checkpoint identity; `mesh_manifest` exposes the shared mesh contract.
- `/v1/vision/image-to-3d-multiview` accepts exactly four or six ordered,
  user-supplied `image[]` uploads plus the managed `image-3d-instantmesh-base`
  id. The managed source `.ckpt` must first be converted offline into its
  `native` child package; the server never interprets Pickle. Client paths,
  uploaded checkpoints, view generation, Zero123++, runtime Python, and
  proprietary FlexiCubes code are excluded. Responses return hashed OBJ, PLY,
  GLB, and manifest artifacts and state that native marching-tetrahedra
  topology is not upstream FlexiCubes topology.
- Native geometry and reconstruction routes enforce decoded-image admission
  independently of multipart byte limits: at most 16,384 pixels per side, 64
  million pixels per image, and 256 million aggregate pixels. Oversized image
  headers return `400 invalid_request_error` before full decode or model load.
- `/v1/audio/speech` accepts `input`, `model`, `voice`, `speed`, and
  `response_format`. It returns WAV by default and can transcode to `mp3`,
  `opus`, `aac`, or `flac` when `ffmpeg` is available. OpenAI model names such
  as `tts-1` map to the local default. Input plus voice instructions may total
  at most 32 KiB of UTF-8 text.
- `/v1/audio/transcriptions` accepts multipart `file`, `model`, `language`,
  `task`, and `response_format`. OpenAI model names such as `whisper-1` map to
  `speech-asr-parakeet`; response formats are `json`, `text`, `verbose_json`,
  `srt`, and `vtt`; `max_tokens` is limited to 1 through 4,096.
- `vision-chat-gemma4-12b` accepts one OpenAI image content part per message through `/v1/chat/completions`; use a file path, `file://` URL, or base64 data URL because the local runtime does not fetch remote images.
- Use `stream_options.include_usage` when a client expects the OpenAI streaming usage chunk.

## Examples

```bash
mere.run api serve --preflight --json
mere.run api serve --host 0.0.0.0 --preflight --json
```

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
mere.run model pull text-chat-lfm25-2.6b-4bit --accept-model-license
mere.run api serve --engine text-chat-lfm2 --model text-chat-lfm25-2.6b-4bit --port 11434
```

```bash
mere.run model pull text-chat-laguna-s-2-1
mere.run api serve --engine text-chat-laguna --max-active-requests 2 --port 11434
```

```bash
mere.run model pull text-agent-ornith-9b
mere.run api serve --engine text-chat-q36 --model text-agent-ornith-9b --port 11434
```

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "text-chat-q36-nano",
    "messages": [{"role": "user", "content": "Return an object with a name and an array of tags."}],
    "response_format": {"type": "json_object"}
  }'
```

The Q36 JSON capability is native-MLX-only in this release. The Linux/GGUF
Q36 runtime uses llama.cpp and continues to reject structured output until a
llama.cpp JSON grammar is wired.

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
curl http://localhost:8080/v1/vision/image-to-3d \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -F model=image-3d-triposr \
  -F resolution=256 \
  -F image=@chair.png
```

```bash
curl http://localhost:8080/v1/vision/image-to-3d-multiview \
  -H "Authorization: Bearer $MERERUN_API_KEY" \
  -F model=image-3d-instantmesh-base \
  -F resolution=128 \
  -F 'image[]=@view-0.png' \
  -F 'image[]=@view-1.png' \
  -F 'image[]=@view-2.png' \
  -F 'image[]=@view-3.png'
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
