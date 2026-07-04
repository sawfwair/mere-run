# Local API Server

This page covers `mere.run api serve`, the local API surface exposed by the package.

## Public surface

- `mere.run api serve`
- `mere.run model runtime get`
- `mere.run model runtime set`
- `mere.run status`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`

## What it is for

The API server lets you expose supported local engines through a local process
instead of shelling out to the CLI for every request. It is useful for:

- local automation
- editor tooling
- local RAG and knowledge-base embeddings
- local image generation
- local text-to-speech and speech-to-text
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

For Gemma 4 12B vision chat:

```bash
swift run mere.run model pull vision-chat-gemma4-12b
swift run mere.run api serve \
  --engine text-chat-gemma4 \
  --model vision-chat-gemma4-12b
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

Native embeddings use the same OpenAI-compatible base URL:

```bash
swift run mere.run model pull text-embed-qwen3-0.6b
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  --data '{
    "model": "text-embed-qwen3-0.6b",
    "input": ["mere.run native embeddings", "local RAG"]
}'
```

Image generation and editing return base64 PNG JSON by default:

```bash
swift run mere.run model pull image-zimage-nano
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  --data '{
    "model": "image-zimage-nano",
    "prompt": "a compact local AI workstation in morning light",
    "size": "1024x1024"
  }'

curl http://127.0.0.1:8080/v1/images/edits \
  -F model=image-zimage-nano \
  -F prompt="make the workstation dusk-lit while preserving the layout" \
  -F image=@input.png
```

Audio endpoints use the same base URL:

```bash
swift run mere.run model pull speech-tts-qwen3-nano
swift run mere.run model pull speech-asr-parakeet

curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  --output speech.wav \
  --data '{
    "model": "speech-tts-qwen3-nano",
    "input": "mere.run is serving native audio.",
    "voice": "nova",
    "response_format": "wav"
  }'

curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=speech-asr-parakeet \
  -F response_format=json \
  -F file=@speech.wav
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
- `/v1/embeddings` uses the native `text-embed-qwen3-0.6b` sidecar model; it is
  listed in `/v1/models` when installed even though it is not a chat-serving engine
- `/v1/images/generations`, `/v1/images/edits`, `/v1/audio/speech`, and
  `/v1/audio/transcriptions` are sidecar routes over existing native CLI
  runtime paths. Installed image, TTS, and ASR sidecar catalog ids are listed in
  `/v1/models` even though they are not chat-serving engines.
- chat completions pass through a fair FIFO request admission actor; the default
  `--max-active-requests 1` preserves serialized local inference and exposes
  queue depth in status; queued client cancellations are removed from the FIFO
  instead of being admitted later
- Gemma4 and Qwen-family prefills are chunked with cancellation and progress checkpoints
  before decode; this is cooperative single-request prefill, not continuous
  batching
- Gemma4 uses in-memory prefix KV reuse by default in `api serve`; set
  `MERERUN_GEMMA4_PREFIX_KV_CACHE=0` for a baseline. `/runtime/status` reports
  cache entries, hits, and reused tokens when the Gemma4 model is loaded; the
  cache stores chunk boundaries plus the stable chat prefix before the final
  message when it is an exact token prefix, and pruning keeps that stable prefix
  ahead of ordinary chunk boundaries.
- Qwen-family chat uses text-only prefix KV reuse by default in `api serve`; set
  `MERERUN_Q35_PREFIX_KV_CACHE=0` for a baseline. Vision prompts are excluded
  because image embeddings alter the effective prefix; text-only requests use
  the same stable chat-prefix checkpoint and pruning rule as Gemma4.
- Managed Gemma4 12B text and vision installs include the
  `text-chat-gemma4-12b-mtp` assistant companion. The API server uses it only
  for greedy serial decode-tail speculation after prefill; sampled requests,
  raw local model paths, prefix-KV seeded requests, and continuous batching stay
  on baseline decode.
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
- the runtime pool applies `ttlSeconds` opportunistically when handling pool
  operations; expired idle models unload automatically unless their settings are
  pinned. Explicit unload remains available for pinned models.
- memory-pressure LRU uses the API server's `--memory-guard` tier. The guard
  derives soft/hard ceilings from process resident memory, host memory
  headroom, and a tier reserve (`safe`, `balanced`, `aggressive`, or
  `custom`). Elevated pressure pauses extra concurrent admissions and evicts the
  least-recently-used idle unpinned model; critical pressure evicts every idle
  unpinned model. Active requests are never evicted.
- `mere.run status` is the preferred quick check before wiring an editor or
  agent to a local server
- it is intentionally local-first
- it should not reintroduce relay, billing, or hosted-infrastructure concerns
- non-loopback binds require an API key, and the OpenAI-compatible routes
  support basic rate limiting
- chat, embedding, image generation, and TTS requests must use
  `Content-Type: application/json`; image editing and STT requests must use
  `multipart/form-data`; browser-simple form/text posts are rejected before the
  request body is processed
- chat requests are validated before generation; `max_tokens`,
  `max_completion_tokens`, `temperature`, and `top_p` must stay within bounded
  ranges
- LoRA adapters are configured at server startup with `--lora`; request bodies
  cannot select local LoRA paths
- streaming and JSON error paths are sanitized so the local server does not
  reflect raw internal runtime details back to clients

## Runtime control endpoints

The control endpoints use the same bearer-token behavior as `/v1/models`,
`/v1/chat/completions`, and `/v1/embeddings`.

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

`GET /v1/models` returns installed API-servable chat managed IDs, configured
aliases, and installed native sidecar model ids for embeddings, image, TTS, and
ASR. Missing catalog models fail with an OpenAI-style error that tells the user
to pull the model first.

## OpenAI chat compatibility

`POST /v1/chat/completions` accepts the common Chat Completions request shape:

- `system`, `developer`, `user`, `assistant`, and `tool` messages
- string content, text content parts, nullable assistant content, and image
  content parts when the selected engine supports vision
- assistant `tool_calls` and tool response messages
- `tools`, `tool_choice`, `parallel_tool_calls`; `tool_choice` accepts `none`,
  `auto`, `required`, and specific function choices by narrowing the advertised
  tool list to the named function
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

For non-streaming chat responses, native runtimes split `<think>...</think>`
blocks out of `message.content` and expose them as OpenAI-compatible
`message.reasoning_content` when present. Streaming responses still emit token
chunks as they are produced.

Engine compatibility:

- `text-chat-deepseek-v4-flash`: raw-proxies the original request body to
  `ds4-server`, preserving DS4's OpenAI-compatible behavior.
- `text-chat-gemma4`: accepts function tools and emits OpenAI tool-call
  responses when the model generates a tool call.
- `vision-chat-gemma4-12b`: uses the Gemma4 serving engine and accepts one
  OpenAI image content part per message. The native runtime accepts local file
  paths, `file://` URLs, or base64 data URLs; it does not fetch remote images.
- `text-chat-q36-nano`: uses the Qwen-family serving engine with Qwen3.6
  35B-A3B OptiQ chat weights, accepts function tools, and accepts one image
  content part per message.
- `text-agent-ornith-9b`: uses the same Qwen-family serving engine for the
  Ornith 1.0 9B OptiQ coding-agent experiment; start it with
  `api serve --engine text-chat-q36 --model text-agent-ornith-9b`.
- `text-agent-ornith-35b-mlx`: uses the Qwen-family serving engine for a local
  converted Ornith 1.0 35B Q4 MLX snapshot; start it with
  `api serve --engine text-chat-q36 --model text-agent-ornith-35b-mlx`.
- Both Ornith lanes serve with thinking-enabled generation by default (the
  models degenerate without it); the reasoning arrives in the response's
  `reasoning_content` field while `content` carries only the visible answer.
  When a request sets no explicit `temperature`/`top_p`, these lanes also apply
  the model's published top-k of 20.
- `text-chat-lfm25-a1b-8bit`: uses the LFM2 serving engine with the
  LiquidAI LFM2.5 8B-A1B MLX 8-bit weights, accepts function tools, and rejects
  image content parts.
- `text-chat-klein`: supports `response_format: {"type":"json_object"}` with
  local JSON retry behavior.
- `text-code`: accepts plain text chat requests and OpenAI `stop` sequences;
  use it for GGUF code models such as `text-code-qwen3`,
  `text-code-north-mini`, and `text-agent-ornith-35b`. It rejects tools,
  images, reasoning controls, logprobs, seed, and structured outputs with
  explicit errors.

Streaming responses only emit assistant content tokens. Local progress labels
stay in logs/stderr, and `stream_options.include_usage` adds the final usage
chunk before `[DONE]`.

## OpenAI embeddings compatibility

`POST /v1/embeddings` accepts the common Embeddings request shape:

- `model`: `text-embed-qwen3-0.6b` or a local Qwen3 embedding model path
- `input`: a string or an array of strings
- `encoding_format`: omitted or `float`
- `user`: accepted as request context

The route returns `object: "list"`, one `embedding` object per input, and
`prompt_tokens`/`total_tokens` usage counts. Dimension overrides and base64
embedding encoding are rejected with `invalid_request_error` because the native
Qwen3 embedding model has a fixed vector size and returns float vectors.

## OpenAI image/audio compatibility

`POST /v1/images/generations` accepts:

- `model`: a mere.run image model id, local image model path, or an OpenAI image
  model name such as `dall-e-3` mapped to `image-zimage-nano`
- `prompt`: required text prompt
- `size`: `WIDTHxHEIGHT`; defaults to `1024x1024`
- `n`: only `1`
- `response_format`: `b64_json` by default, or `url` for a local `file://` URL
- local extensions: `seed`, `negative_prompt`, `steps`, and `guidance_scale`

`POST /v1/images/edits` accepts multipart form fields:

- `image`: required input image file part; Open WebUI-style `image[]` repeated
  parts are accepted for multi-image edit requests
- `mask`: optional mask file part
- `model`: a mere.run image model id, local image model path, `qwen-image-edit`,
  or an OpenAI image model name such as `gpt-image-1` mapped to
  `image-zimage-nano`
- `prompt`: required edit instruction
- `size`: `WIDTHxHEIGHT`; defaults to `1024x1024`
- `n`: only `1`
- `response_format`: `b64_json` by default, or `url` for a local `file://` URL
- local extensions: `strength`, `seed`, `negative_prompt`, `steps`, and
  `guidance_scale`

Masks are accepted for client compatibility. Current native edit models use
whole-image conditioning rather than strict masked inpainting.

`POST /v1/audio/speech` accepts:

- `model`: `speech-tts-qwen3-nano`, a local Qwen3-TTS model path, or OpenAI
  names such as `tts-1` mapped to the local default
- `input`: required text
- `voice`: OpenAI voice names are translated to Qwen3-TTS style descriptions;
  custom descriptions are passed through
- `speed`, `instructions`, and `temperature`
- `response_format`: `wav`, `mp3`, `opus`, `aac`, or `flac`; non-WAV formats
  require `ffmpeg`

`POST /v1/audio/transcriptions` accepts multipart form fields:

- `file`: required audio file part
- `model`: `speech-asr-parakeet`, `speech-asr-qwen3`, a local ASR model path,
  or OpenAI names such as `whisper-1` mapped to `speech-asr-parakeet`
- `language`
- `task`: `transcribe` or `translate`
- `response_format`: `json`, `text`, `verbose_json`, `srt`, or `vtt`
- `max_tokens`

Unsupported format choices return OpenAI-style `invalid_request_error` payloads
instead of being ignored.

## Open WebUI companion

Open WebUI can use mere.run as an OpenAI-compatible provider without being
vendored into this repo. Start mere.run, run the official Open WebUI Docker or
pip install, then configure Open WebUI with the mere.run `/v1` base URL.

One-command Docker quickstart:

```bash
mere.run open-webui quickstart --pull
```

That starts `api serve`, runs the official Open WebUI container, configures the
OpenAI provider, filters the chat picker to the configured text and vision chat
models, points RAG at `/v1/embeddings`, and keeps image editing disabled for the
live-smoke path. Preview the exact commands with:

```bash
mere.run open-webui quickstart --dry-run
```

Repeatable smoke harness:

```bash
export MERERUN_API_KEY=change-me
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-12b \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$MERERUN_API_KEY"

MERERUN_OPENWEBUI_TEXT_MODEL=text-chat-gemma4-12b \
MERERUN_OPENWEBUI_RESET=1 scripts/smoke-open-webui.sh docker-run
MERERUN_OPENWEBUI_TEXT_MODEL=text-chat-gemma4-12b \
scripts/smoke-open-webui.sh live-smoke
```

`live-smoke` waits for Open WebUI, signs in to the disposable no-auth smoke
admin user, filters the OpenAI chat connection to the configured text and vision
chat ids, imports per-model metadata wrappers, sends text and vision chat
through Open WebUI's own proxy, then exercises the direct mere.run API surface
and the Open WebUI model list. Use `configure`, `proxy-smoke`, `api-smoke`, and
`ui-smoke` separately when debugging one stage.

Docker bridge path:

```bash
export MERERUN_API_KEY=change-me
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-12b \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$MERERUN_API_KEY"

DEFAULT_MODEL_METADATA='{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
docker run -d \
  --name open-webui \
  --restart unless-stopped \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e OPENAI_API_BASE_URLS=http://host.docker.internal:8080/v1 \
  -e OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e OPENAI_API_KEYS="$MERERUN_API_KEY" \
  -e DEFAULT_MODELS=text-chat-gemma4-12b \
  -e DEFAULT_MODEL_PARAMS='{"function_calling":"native"}' \
  -e DEFAULT_MODEL_METADATA="$DEFAULT_MODEL_METADATA" \
  -e RAG_EMBEDDING_ENGINE=openai \
  -e RAG_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e RAG_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e RAG_EMBEDDING_MODEL=text-embed-qwen3-0.6b \
  -e ENABLE_IMAGE_GENERATION=True \
  -e ENABLE_IMAGE_EDIT=False \
  -e IMAGE_GENERATION_ENGINE=openai \
  -e IMAGE_GENERATION_MODEL=image-zimage-nano \
  -e IMAGE_SIZE=1024x1024 \
  -e IMAGE_EDIT_ENGINE=openai \
  -e IMAGE_EDIT_MODEL=qwen-image-edit \
  -e IMAGE_EDIT_SIZE=1024x1024 \
  -e IMAGES_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e IMAGES_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e IMAGES_EDIT_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e IMAGES_EDIT_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e AUDIO_TTS_ENGINE=openai \
  -e AUDIO_TTS_MODEL=speech-tts-qwen3-nano \
  -e AUDIO_TTS_VOICE=nova \
  -e AUDIO_TTS_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e AUDIO_TTS_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e AUDIO_TTS_OPENAI_PARAMS='{"response_format":"wav"}' \
  -e AUDIO_STT_ENGINE=openai \
  -e AUDIO_STT_MODEL=speech-asr-parakeet \
  -e AUDIO_STT_OPENAI_API_BASE_URL=http://host.docker.internal:8080/v1 \
  -e AUDIO_STT_OPENAI_API_KEY="$MERERUN_API_KEY" \
  -e ENABLE_PERSISTENT_CONFIG=False \
  -e WEBUI_AUTH=False \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

The environment variables above preconfigure Open WebUI on a fresh data volume.
Run `scripts/smoke-open-webui.sh configure` after the container is healthy to
set Open WebUI's OpenAI connection `model_ids` filter and import per-model
capability wrappers. That keeps image, embedding, TTS, and STT sidecars out of
the chat selector while still using them in their own settings. You can also set
the same values in the admin UI:

- Base URL: `http://host.docker.internal:8080/v1`
- API key: the value of `MERERUN_API_KEY`
- Chat model: an installed `text-chat-*` model from `/v1/models`
- Vision model: `vision-chat-gemma4-12b`
- RAG embedding model: `text-embed-qwen3-0.6b`
- Image generation engine/model: `openai` / `image-zimage-nano`
- Image editing: disabled for the live smoke path with `ENABLE_IMAGE_EDIT=False`
- Image edit engine/model when enabled: `openai` / `qwen-image-edit`
- TTS engine/model: `openai` / `speech-tts-qwen3-nano`
- STT engine/model: `openai` / `speech-asr-parakeet`
- Function calling: native mode with `{"function_calling":"native"}`

Docker Compose / DGX Spark path:

Keep mere.run on the host and run only Open WebUI in Docker. This is the
cleaner path for Linux, DGX Spark, or LAN workstations because the mere.run
process owns the local model store and runtime/GPU setup while Open WebUI keeps
its persistent app data in a Docker volume.

```bash
export MERERUN_API_KEY=change-me
mere.run api serve \
  --engine text-chat-gemma4 \
  --model text-chat-gemma4-12b \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key "$MERERUN_API_KEY"
```

Create `open-webui-mere-run/.env`:

```dotenv
MERERUN_API_KEY=change-me
WEBUI_SECRET_KEY=replace-with-openssl-rand-hex-32
OPEN_WEBUI_IMAGE=ghcr.io/open-webui/open-webui:main
OPEN_WEBUI_BIND=0.0.0.0
OPEN_WEBUI_URL=http://spark.local:3000
MERERUN_OPENWEBUI_API_URL=http://host.docker.internal:8080/v1
MERERUN_OPENWEBUI_TEXT_MODEL=text-chat-gemma4-12b
```

Create `open-webui-mere-run/compose.yaml`:

```yaml
services:
  open-webui:
    image: ${OPEN_WEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main}
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "${OPEN_WEBUI_BIND:-0.0.0.0}:3000:8080"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      WEBUI_URL: ${OPEN_WEBUI_URL:-http://localhost:3000}
      WEBUI_AUTH: "True"
      WEBUI_SECRET_KEY: ${WEBUI_SECRET_KEY}
      ENABLE_PERSISTENT_CONFIG: "True"
      OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      OPENAI_API_BASE_URLS: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      OPENAI_API_KEY: ${MERERUN_API_KEY}
      OPENAI_API_KEYS: ${MERERUN_API_KEY}
      DEFAULT_MODELS: ${MERERUN_OPENWEBUI_TEXT_MODEL:-text-chat-gemma4-12b}
      DEFAULT_MODEL_PARAMS: '{"function_calling":"native"}'
      DEFAULT_MODEL_METADATA: '{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
      RAG_EMBEDDING_ENGINE: openai
      RAG_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      RAG_OPENAI_API_KEY: ${MERERUN_API_KEY}
      RAG_EMBEDDING_MODEL: text-embed-qwen3-0.6b
      ENABLE_IMAGE_GENERATION: "True"
      ENABLE_IMAGE_EDIT: "False"
      IMAGE_GENERATION_ENGINE: openai
      IMAGE_GENERATION_MODEL: image-zimage-nano
      IMAGE_SIZE: 1024x1024
      IMAGES_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      IMAGES_OPENAI_API_KEY: ${MERERUN_API_KEY}
      AUDIO_TTS_ENGINE: openai
      AUDIO_TTS_MODEL: speech-tts-qwen3-nano
      AUDIO_TTS_VOICE: nova
      AUDIO_TTS_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      AUDIO_TTS_OPENAI_API_KEY: ${MERERUN_API_KEY}
      AUDIO_TTS_OPENAI_PARAMS: '{"response_format":"wav"}'
      AUDIO_STT_ENGINE: openai
      AUDIO_STT_MODEL: speech-asr-parakeet
      AUDIO_STT_OPENAI_API_BASE_URL: ${MERERUN_OPENWEBUI_API_URL:-http://host.docker.internal:8080/v1}
      AUDIO_STT_OPENAI_API_KEY: ${MERERUN_API_KEY}
    volumes:
      - open-webui:/app/backend/data

volumes:
  open-webui:
```

Then run:

```bash
cd open-webui-mere-run
docker compose up -d
```

For Open WebUI on a different machine, set `MERERUN_OPENWEBUI_API_URL` to the
mere.run host's LAN URL, for example `http://192.168.1.50:8080/v1`. Keep
`WEBUI_AUTH=True`, keep a stable `WEBUI_SECRET_KEY`, and pin
`OPEN_WEBUI_IMAGE` to a tested release tag before treating the instance as
shared or production-like.

Pip path:

```bash
python3.11 -m pip install --upgrade open-webui
DEFAULT_MODEL_METADATA='{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_BASE_URLS=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY="$MERERUN_API_KEY" \
OPENAI_API_KEYS="$MERERUN_API_KEY" \
DEFAULT_MODELS=text-chat-gemma4-12b \
DEFAULT_MODEL_PARAMS='{"function_calling":"native"}' \
DEFAULT_MODEL_METADATA="$DEFAULT_MODEL_METADATA" \
RAG_EMBEDDING_ENGINE=openai \
RAG_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
RAG_OPENAI_API_KEY="$MERERUN_API_KEY" \
RAG_EMBEDDING_MODEL=text-embed-qwen3-0.6b \
ENABLE_IMAGE_GENERATION=True \
ENABLE_IMAGE_EDIT=False \
IMAGE_GENERATION_ENGINE=openai \
IMAGE_GENERATION_MODEL=image-zimage-nano \
IMAGE_SIZE=1024x1024 \
IMAGE_EDIT_ENGINE=openai \
IMAGE_EDIT_MODEL=qwen-image-edit \
IMAGE_EDIT_SIZE=1024x1024 \
IMAGES_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
IMAGES_OPENAI_API_KEY="$MERERUN_API_KEY" \
IMAGES_EDIT_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
IMAGES_EDIT_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_ENGINE=openai \
AUDIO_TTS_MODEL=speech-tts-qwen3-nano \
AUDIO_TTS_VOICE=nova \
AUDIO_TTS_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_TTS_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_OPENAI_PARAMS='{"response_format":"wav"}' \
AUDIO_STT_ENGINE=openai \
AUDIO_STT_MODEL=speech-asr-parakeet \
AUDIO_STT_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_STT_OPENAI_API_KEY="$MERERUN_API_KEY" \
ENABLE_PERSISTENT_CONFIG=False \
WEBUI_AUTH=False \
open-webui serve --host 127.0.0.1 --port 3000
```

For pip-based smoke configuration, use the same-host provider URL:

```bash
OPEN_WEBUI_MERERUN_API_URL=http://127.0.0.1:8080/v1 \
scripts/smoke-open-webui.sh configure
```

Use Base URL `http://127.0.0.1:8080/v1` for any admin UI edits in the pip
install. If an existing Open WebUI data directory ignores changed environment
variables, update the values in the admin UI, set `ENABLE_PERSISTENT_CONFIG=False`
for smoke, or start with a fresh data directory. Set the
`vision-chat-gemma4-12b` per-model capability `vision=true` before UI vision
upload smoke if you use the conservative global model metadata above.

uv path:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export DATA_DIR="$HOME/.open-webui-mere-run"
export MERERUN_API_KEY=change-me

DEFAULT_MODEL_METADATA='{"capabilities":{"file_context":true,"vision":false,"file_upload":true,"web_search":false,"image_generation":true,"code_interpreter":false,"terminal":false,"citations":true,"status_updates":true,"builtin_tools":true}}'
DATA_DIR="$HOME/.open-webui-mere-run" \
OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_BASE_URLS=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY="$MERERUN_API_KEY" \
OPENAI_API_KEYS="$MERERUN_API_KEY" \
DEFAULT_MODELS=text-chat-gemma4-12b \
DEFAULT_MODEL_PARAMS='{"function_calling":"native"}' \
DEFAULT_MODEL_METADATA="$DEFAULT_MODEL_METADATA" \
RAG_EMBEDDING_ENGINE=openai \
RAG_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
RAG_OPENAI_API_KEY="$MERERUN_API_KEY" \
RAG_EMBEDDING_MODEL=text-embed-qwen3-0.6b \
ENABLE_IMAGE_GENERATION=True \
ENABLE_IMAGE_EDIT=False \
IMAGE_GENERATION_ENGINE=openai \
IMAGE_GENERATION_MODEL=image-zimage-nano \
IMAGES_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
IMAGES_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_ENGINE=openai \
AUDIO_TTS_MODEL=speech-tts-qwen3-nano \
AUDIO_TTS_VOICE=nova \
AUDIO_TTS_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_TTS_OPENAI_API_KEY="$MERERUN_API_KEY" \
AUDIO_TTS_OPENAI_PARAMS='{"response_format":"wav"}' \
AUDIO_STT_ENGINE=openai \
AUDIO_STT_MODEL=speech-asr-parakeet \
AUDIO_STT_OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
AUDIO_STT_OPENAI_API_KEY="$MERERUN_API_KEY" \
ENABLE_PERSISTENT_CONFIG=False \
WEBUI_AUTH=False \
uvx --python 3.11 open-webui@latest serve --host 127.0.0.1 --port 3000
```

Run the mere.run API on `127.0.0.1:8080` in another terminal. The explicit
Open WebUI `--port 3000` avoids colliding with the mere.run API recipe.

The CLI cookbook has the longer copy-paste recipe:

```bash
mere.run guide open-webui
```

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
