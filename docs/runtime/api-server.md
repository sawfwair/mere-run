# Local API Server

The rest of `mere.run`, behind an OpenAI-compatible endpoint on localhost.
Point an existing client, your editor, or Open WebUI at it and nothing about
your setup changes except where the weights live — and where your prompts stop.

## Commands

| Command | What it does |
| --- | --- |
| `mere.run api serve` | Start an OpenAI-compatible API server for supported local text, image, vision, and audio models. |
| `mere.run model runtime get` | Print typed API runtime settings for a managed model. |
| `mere.run model runtime set` | Update typed API runtime settings for a managed model. |
| `mere.run status` | Show local server, loaded model, and installed model status. |

## macOS Serving & Agents Console

MereRun Studio exposes this control plane as a top-level **Serving & Agents**
destination (or `Shift-Command-S`). It is an operational client of the public
CLI and HTTP contracts, not a second runtime. The console provides:

- API preflight, app-owned start/stop/restart, and reconnection to a server
  started outside Studio
- explicit loopback/LAN authentication safety; Studio blocks a non-loopback
  app-owned launch until an API key is configured
- the live text and image/speech/transcription/embedding sidecar pools,
  readiness, activity, queues, pinning, TTLs, aliases, model settings, and
  text-model load/unload controls
- unified-memory guard pressure, process CPU, honest Metal allocation and
  recommended working set, and macOS thermal/power state where available
- observed request totals, failures, tokens, latency/throughput, prefix reuse,
  decode batching, and MTP counters
- typed Pi readiness/install/configure/start flows plus copyable OpenAI SDK,
  curl, BYOA, and Open WebUI connection setup
- a sanitized lifecycle/activity feed that excludes prompts, messages, media,
  and generated content

Server launches and agent setup sessions use the normal Studio Library
lifecycle. API keys still cross the process boundary only through
`MERERUN_API_KEY`.

## HTTP routes

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/vision/geometry`
- `POST /v1/vision/geometry/multiview`
- `POST /v1/vision/image-to-3d`
- `POST /v1/vision/image-to-3d-multiview`
- `POST /v1/vision/depth-video`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`

## What it is for

Serving beats shelling out to the CLI once you are making more than one request
— the model stays loaded between them. It suits:

- editor tooling and local automation
- RAG and knowledge-base embeddings over private documents
- image generation, speech, and transcription from an existing OpenAI client
- trying the runtime out over HTTP before wiring anything permanent

It is not a hosted service or a relay layer. The server binds to loopback by
default. For remote jobs, prefer [portable workflows](../workflows.md); when
you deliberately bind the API server to a non-loopback address, configure its
bearer token and rate limit first.

The server keeps its existing in-process request admission, resident model
pool, prefix caches, and continuous batching. In addition, the server process
holds a weighted machine-wide reservation shared with other `mere.run`
processes. This prevents a terminal, Studio, Raycast, or a second server from
blindly loading incompatible working sets while allowing measured concurrency
inside the resident pool. Standard servers consume two RAM-scaled permits;
DeepSeek V4 Flash servers run with exclusive machine admission.

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

For Muse Glimmer multimodal agent chat (explicit 21.38 GB Q4 target plus the
5.54 GB DFlash2 assistant):

```bash
swift run mere.run model pull vision-chat-muse-glimmer-30b --accept-model-license
swift run mere.run api serve \
  --engine text-chat-muse-glimmer \
  --model vision-chat-muse-glimmer-30b
```

This engine accepts OpenAI image content parts and function tools. It exposes
reasoning effort, but does not claim constrained `json_object` decoding. The
managed pull also installs z-lab's pinned DFlash2 assistant. Requests for
at least 32 output tokens use target-verified speculation and return to
target-only decode when draft acceptance is poor.

The native DFlash2 companion can be refreshed independently:

```bash
swift run mere.run model pull vision-chat-muse-glimmer-30b-dflash2
swift run mere.run api serve \
  --model vision-chat-muse-glimmer-30b
```

For NVIDIA Nemotron 3.5 Lightning with its automatically installed DSpark
companion:

```bash
swift run mere.run model pull text-chat-nemotron-35-lightning
swift run mere.run api serve \
  --engine text-chat-nemotron-h \
  --model text-chat-nemotron-35-lightning
```

The native engine serves text, tools, and stop sequences. Requests with at
least 16 output tokens probe DSpark at NVIDIA's recommended three-token width;
low draft acceptance automatically routes the rest of that request through the
serial target.

For the compact LiquidAI LFM2.5 2.6B MLX 4-bit model:

```bash
swift run mere.run model pull text-chat-lfm25-2.6b-4bit --accept-model-license
swift run mere.run api serve --engine text-chat-lfm2 --model text-chat-lfm25-2.6b-4bit
```

For LFM2.5-VL image content parts, select the pinned vision model explicitly:

```bash
swift run mere.run model pull vision-chat-lfm25-3b-8bit --accept-model-license
swift run mere.run api serve --engine text-chat-lfm2 --model vision-chat-lfm25-3b-8bit
```

For the opt-in Laguna S 2.1 target with its automatically installed DFlash
companion:

```bash
swift run mere.run model pull text-chat-laguna-s-2-1
swift run mere.run api serve \
  --engine text-chat-laguna \
  --max-active-requests 2
```

Laguna uses its validated `1/1/20/0.02` temperature, top-p, top-k, and min-p
recipe when a request does not override sampling. The server automatically
routes output budgets of at least 32 tokens through DFlash and falls back
losslessly when measured draft acceptance is poor.

To serve the smaller experimental XS checkpoint instead, select it explicitly;
the server will not attach the S DFlash companion:

```bash
swift run mere.run model pull text-chat-laguna-xs-2-1
swift run mere.run api serve \
  --engine text-chat-laguna \
  --model text-chat-laguna-xs-2-1
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
  --top-p 0.9 \
  --min-p 0.05
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
  runtime paths. Installed embedding, image, TTS, and ASR sidecar catalog ids are listed in
  `/v1/models` even though they are not chat-serving engines.
- the server keeps one most-recently-used embedding lane, one image lane shared
  by generation and editing, one TTS lane, and one ASR lane resident. Repeated requests for the
  same model reuse loaded components; mutable generators execute exclusively,
  and a model or ASR-backend switch unloads the previous runtime before loading
  the next one so request-selected local paths cannot grow residency without
  bound.
- sidecars default to a 300-second idle TTL. Per-lane autonomous timers expire
  idle residents without waiting for another request and re-read managed
  `pinned` and `ttlSeconds` settings while idle, so live settings changes take
  effect. Managed embedding, image, TTS, and ASR models accept
  `model runtime set <id> --ttl-seconds <seconds>` and `--pinned`;
  sidecar-specific settings reject text-only sampling, engine, alias, context,
  and KV controls. The special `qwen-image-edit` repository lane is resident but
  currently uses the default lifecycle policy because it is not configurable
  through `model runtime`.
- sidecar admission samples the same `--memory-guard` policy before and after an
  operation. Cold sidecar operations are exclusive across lanes, preventing two
  model loads from racing each other or an already-running warm sidecar. Image
  operations remain exclusive because staged image pipelines can reload
  components even when their resident generator is warm. Before a new resident loads, the
  catalog estimate (or resolved local directory size) is projected against the
  hard guard with conservative per-family working-set floors; idle unpinned residents are proactively released, and the request
  fails with a memory-pressure error if adequate headroom cannot be recovered.
  Under pressure, admission gives idle unpinned text runtimes the first
  opportunity to unload, one at a time with pressure re-sampled after each
  eviction; the idle startup default is eligible for this sidecar admission
  path. If pressure remains, eligible idle sidecars are evicted oldest first.
  Active, queued, or pinned residents are protected throughout.
- chat, embedding, image, TTS, and ASR inference pass through a fair FIFO request
  admission actor; the default `--max-active-requests 1` serializes local
  inference across text and media activation peaks and exposes queue depth in
  status. Raising it is an explicit throughput and unified-memory tradeoff;
  queued client cancellations are removed from the FIFO instead of being
  admitted later. Explicit runtime model load/unload maintenance shares the
  same queue
- machine admission complements rather than replaces `--max-active-requests`:
  the server's local limit controls request and batching concurrency inside its
  weighted machine reservation
- Gemma4, Qwen-family, and LFM2 prefills are chunked with cancellation and
  progress checkpoints before decode; this is cooperative single-request
  prefill, not continuous batching. Qwen3.8 defaults to 1,024-token chunks and
  caps them at 512 when live reclaimable memory falls below 16 GiB or a peer is
  admitted, so a live decoder is not held behind a pressure-heavy prefill
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
- LFM2 chat uses prefix KV reuse by default in `api serve`; set
  `MERERUN_LFM2_PREFIX_KV_CACHE=0` for a baseline. It retains exact prompts and
  the stable conversation prefix before the final message, without cloning
  every intermediate prefill chunk. Both attention KV and short-convolution
  state are forked at those retained checkpoints.
- Managed Gemma4 12B text and vision installs include the
  `text-chat-gemma4-12b-mtp` assistant companion. The API server uses it only
  for greedy serial decode-tail speculation after prefill; sampled requests,
  raw local model paths, prefix-KV seeded requests, and continuous batching stay
  on baseline decode.
- Gemma4, Qwen-family, and LFM2 decode batching engages automatically when
  `--max-active-requests` is above `1`. Their engine-specific continuous-batching
  variables can force the implementation on or force the serial path, and
  `/runtime/status` reports actual batched decode steps instead of assuming the
  scheduler is active. An eligible Qwen MTP request takes its speculative lane
  only when no peer is already admitted; contended arrivals use ordinary
  batching, and a late arrival does not migrate an MTP request already in
  progress. Gemma4 full-attention rows remain same-position because
  that engine still uses scalar RoPE/cache offsets. Qwen-family and LFM2
  full-attention rows use row-offset-aware ragged KV caches; Qwen-family linear
  attention and LFM2 short-convolution layers use typed recurrent state, so
  compatible rows may batch across decode positions. The scheduler services the
  earliest decode position first, batching compatible rows there or advancing a
  single lower-offset row until it can join one
- Gemma4 has an experimental packed PolarKV path behind
  `--kv-quant-scheme polar --kv-bits 2`; use it for memory-pressure and
  long-context synthetic decode testing. It is not the default until checkpoint
  benchmarks prove the end-to-end model path.
- Gemma4, Qwen-family, and LFM2 runtime settings can set `kvCacheMode` to
  explicit `affine8` as a memory control relative to full-precision KV.
  Qwen-family and LFM2 dequantize the generic cache for attention. Gemma Turbo
  already defaults to a smaller 4-bit TurboQuant cache, so forcing affine 8-bit
  can increase its KV residency. `default` restores the engine/model/server
  default, not necessarily full precision. Gemma additionally accepts `polar2`
  or `auto`; `auto` keeps the default KV path below 1024 prompt tokens and
  switches to decode-deferred packed PolarKV at or above that threshold.
- `/runtime/status` aggregates prefix hits, reused tokens, and batched decode
  steps across loaded models under `cacheStats`; it also reports completed chat
  request counts, generated tokens, and average load/prefill/decode timings
  under `benchmarkStats`. The additive `sidecars` object reports the embedding,
  image, speech, and transcription resident model/path, active and queued requests,
  load/access/eviction timestamps, TTL/pinned state, readiness, and lifecycle
  counters. For text and sidecar entries, `loaded` remains the compatibility
  signal that a resident object exists; additive `ready: false` means text
  preparation is in progress or a sidecar's first operation is loading or
  failed. Older payloads can omit `ready`, and older clients can ignore it. SSD
  KV persistence remains unavailable until the in-memory counters justify it
- runtime settings are stored at
  `<active model store>/.mere-run/runtime-model-settings.json`
- the runtime pool applies `ttlSeconds` opportunistically when handling pool
  operations; expired idle models unload automatically unless their settings are
  pinned. Explicit unload remains available for pinned models.
- memory-pressure LRU uses the API server's `--memory-guard` tier. The guard
  derives soft/hard ceilings from Darwin physical footprint (RSS elsewhere),
  host memory
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
  `Content-Type: application/json`; image editing, STT, and vision
  geometry/3D/depth requests must use `multipart/form-data`; browser-simple
  form/text posts are rejected before the request body is processed
- chat requests are validated before generation; `max_tokens`,
  `max_completion_tokens`, `temperature`, `top_p`, and the supported `min_p`
  extension must stay within bounded ranges
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
  batching stats when enabled, text/sidecar residency and readiness, and
  aggregate traffic stats measured from completed native chat requests.
  Newer servers also include additive `process` telemetry: PID, start time,
  uptime, sampled process CPU, macOS thermal/low-power state, Metal device,
  current Metal allocation, recommended maximum working set, and unified-memory
  capability. Older clients ignore it and older servers may omit it.
- `POST /runtime/models/{id}/load`: explicitly load an installed API-capable
  text-pool catalog model.
- `POST /runtime/models/{id}/unload`: unload a text-pool model; returns `409`
  while active requests are using it.
- `GET /runtime/models/{id}/settings`: read typed text-pool runtime defaults.
- `PATCH /runtime/models/{id}/settings`: replace typed text-pool runtime
  defaults.

These four `/runtime/models/{id}` HTTP operations address the chat/text runtime
pool only. Managed embedding, image, TTS, and ASR sidecar TTL/pinning is configured with
`mere.run model runtime set` (or the settings file); there is no sidecar HTTP
load/unload/settings endpoint. Sidecars still appear in `/v1/models` and in the
`sidecars` object returned by `/runtime/status`.

`GET /v1/models` returns installed API-servable chat managed IDs, configured
aliases, and installed native sidecars. In addition to the standard OpenAI
fields, each entry can describe its `task`, `tool_call`, `reasoning`,
`thinking_levels`, input/output `modalities`, context/output `limit`, and
`openai_compat` dialect. Agent harnesses can therefore discover what the
running server actually supports instead of inferring capabilities from a
model name. Missing catalog models fail with an OpenAI-style error that tells
the user to pull the model first.

For managed models, these capabilities come from the typed API profile attached
to the managed-model catalog entry. The runtime layer adds the request-facing
ID or alias and applies configured context/output defaults to the reported
limits. A server started from an explicit uncataloged path uses a conservative
profile for its selected serving engine.

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

Native Qwen-family, Nemotron Lightning, and Laguna catalog profiles accept
`logprobs: true` for non-streaming requests, with `top_logprobs` from 0 through
20. Logprob capture currently requires unconstrained text output rather than
`json_object` or `json_schema`. The response keeps the OpenAI
`choices[].logprobs.content[]` shape and adds
explicit `raw_logprob`, `policy_logprob`, entropy, margin, region, summary, and
capture-overhead fields. Raw measures the target model before sampling
transforms; policy measures the exact distribution after temperature/top-k/
top-p/min-p. Reasoning token text is omitted. Capture routes through serial
final-target decode, so speculative draft distributions never appear as model
confidence.

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
  content part per message. It also accepts
  `response_format: {"type":"json_object"}` and reports structured-output
  support with strict mode disabled. JSON-object mode forces thinking off and
  uses token-level constrained serial decoding; it does not implement
  `json_schema`.
- `vision-chat-q38-27b` and `vision-chat-q38-27b-4bit`: use the same native
  Qwen-family serving engine for the official dense Qwen3.8 27B BF16 checkpoint
  or the pinned MLX 4-bit conversion. Both accept function tools, one
  local/base64 image content part per message, and structured JSON output; both
  default to thinking and the published 1.0/0.95/20 sampling. Pull the 55.59 GB
  BF16 lane or 19.47 GB 4-bit-plus-MTP lane explicitly before serving it.
- `text-chat-bonsai-27b-1bit` and `text-chat-bonsai-27b-2bit`: use the same
  native Qwen-family serving engine for Prism ML's dense packed binary and
  ternary 27B checkpoints. They accept function tools and one local/base64
  image content part per message, default to thinking, and use the published
  0.7/0.95/20 sampling when omitted. Start the server with
  `--context-size 262144` to expose their full advertised context; the
  server-wide default remains the conservative 32K limit.
- `text-agent-ornith-9b`: uses the same Qwen-family serving engine for the
  Ornith 1.0 9B OptiQ coding-agent experiment; start it with
  `api serve --engine text-chat-q36 --model text-agent-ornith-9b`.
- `text-agent-ornith-35b-mlx`: uses the Qwen-family serving engine for Ornith's
  official 1.5 35B-A3B BF16 MLX snapshot; install it explicitly, then start it with
  `api serve --engine text-chat-q36 --model text-agent-ornith-35b-mlx`.
- Both Ornith lanes serve with thinking-enabled generation by default (the
  models degenerate without it); the reasoning arrives in the response's
  `reasoning_content` field while `content` carries only the visible answer.
  When a request sets no explicit `temperature`/`top_p`/`min_p`, these lanes
  also apply the model's published top-k of 20.
- `text-chat-lfm25-a1b-8bit`: uses the LFM2 serving engine with the
  LiquidAI LFM2.5 8B-A1B MLX 8-bit weights, accepts function tools, and rejects
  image content parts.
- `text-chat-lfm25-2.6b-4bit`: uses the same native LFM2 serving engine with
  LiquidAI's dense 2.6B MLX 4-bit weights, accepts function tools, and rejects
  image content parts.
- `vision-chat-lfm25-3b-8bit`: uses the native LFM2 serving engine with
  LiquidAI's dense language backbone, SigLIP2 vision tower, and multimodal
  projector. It accepts local-file and base64 image content parts; it does not
  fetch remote image URLs.
- `text-chat-klein`: supports `response_format: {"type":"json_object"}` with
  local JSON retry behavior.
- `text-code`: accepts plain text chat requests and OpenAI `stop` sequences;
  use it for GGUF code models such as `text-code-qwen3`,
  `text-code-north-mini`, and `text-agent-ornith-35b`. It rejects tools,
  images, reasoning controls, logprobs, seed, and structured outputs with
  explicit errors.

The JSON-object capability applies to the native MLX Qwen-family runtime,
including Q35-compatible models that share that generator. The Linux/GGUF Q36
lane still routes through llama.cpp and rejects structured output until its
JSON grammar is wired.

Streaming responses only emit assistant content tokens. Local progress labels
stay in logs/stderr, and `stream_options.include_usage` adds the final usage
chunk before `[DONE]`.

## OpenAI embeddings compatibility

`POST /v1/embeddings` accepts the common Embeddings request shape:

- `model`: `text-embed-qwen3-0.6b` or a local Qwen3 embedding model path
- `input`: a string or an array of strings
- `encoding_format`: omitted or `float`
- `user`: accepted as request context

One request may contain at most 256 texts and 2 MiB of UTF-8 input. The native
runtime caps each row at 8,192 tokens, length-packs rows into sequential batches
with at most 8,192 padded tokens, and restores response order after evaluation.

The route returns `object: "list"`, one `embedding` object per input, and
`prompt_tokens`/`total_tokens` usage counts. Dimension overrides and base64
embedding encoding are rejected with `invalid_request_error` because the native
Qwen3 embedding model has a fixed vector size and returns float vectors.

## OpenAI image/audio compatibility

`POST /v1/images/generations` accepts:

- `model`: a mere.run image model id, local image model path, or an OpenAI image
  model name such as `dall-e-3` mapped to `image-zimage-nano`
- `prompt`: required text prompt
- `size`: `WIDTHxHEIGHT`; defaults to `1024x1024`; each dimension must be a
  multiple of 16 from 16 through 4,096 pixels, with total area limited to
  4,194,304 pixels
- `n`: only `1`
- `response_format`: `b64_json` by default, or `url` for a local `file://` URL
- local extensions: `seed`, `negative_prompt`, `steps` (1 through 100), and
  `guidance_scale`

`POST /v1/images/edits` accepts multipart form fields:

- `image`: required input image file part; Open WebUI-style `image[]` repeated
  parts are accepted for multi-image edit requests
- `mask`: optional mask file part
- `model`: a mere.run image model id, local image model path, `qwen-image-edit`,
  or an OpenAI image model name such as `gpt-image-1` mapped to
  `image-zimage-nano`
- `prompt`: required edit instruction
- `size`: `WIDTHxHEIGHT`; defaults to `1024x1024`; each dimension must be a
  multiple of 16 from 16 through 4,096 pixels, with total area limited to
  4,194,304 pixels
- `n`: only `1`
- `response_format`: `b64_json` by default, or `url` for a local `file://` URL
- local extensions: `strength`, `seed`, `negative_prompt`, `steps` (1 through
  100), and `guidance_scale`

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

The normalized input and voice instructions may total at most 32 KiB of UTF-8
text.

`POST /v1/audio/transcriptions` accepts multipart form fields:

- `file`: required audio file part
- `model`: `speech-asr-parakeet`, `speech-asr-qwen3`, a local ASR model path,
  or OpenAI names such as `whisper-1` mapped to `speech-asr-parakeet`
- `language`
- `task`: `transcribe` or `translate`
- `response_format`: `json`, `text`, `verbose_json`, `srt`, or `vtt`
- `max_tokens`: 1 through 4,096; defaults to 448

Unsupported format choices return OpenAI-style `invalid_request_error` payloads
instead of being ignored.

## Vision geometry and 3D compatibility

The five `/v1/vision/*` routes take `multipart/form-data` and return JSON whose
artifacts are server-local `file://` URLs, retained for one hour. Because those
URLs only make sense on the serving machine, the routes are loopback-only:
authenticated remote clients get an error, and their model ids are filtered out
of `/v1/models` for non-loopback clients. Inputs must be uploaded file parts;
client filesystem paths are rejected. Each route accepts only its managed
default model id (depth video also accepts its metric variant).

`POST /v1/vision/geometry` accepts:

- `image`: required single input image file part
- `model`: only `vision-geometry-moge2-small`
- `resolution_level`: integer 0 through 9; defaults to 9
- `token_count`: optional integer 1 through 3,600
- `max_points`: optional positive point-count cap

`POST /v1/vision/geometry/multiview` accepts:

- `image` / `image[]`: one or more image file parts; multipart order is the
  view order
- `model`: only `vision-geometry-da3-small`
- `process_resolution`: defaults to 504
- `reference_view`: `first`, `middle`, `saddle-balanced`, or
  `saddle-similarity-range`; defaults to `saddle-balanced`
- `confidence_percentile`: 0 through 100; defaults to 40
- `max_points`: defaults to 1,000,000
- `cameras`: optional known-camera JSON, as either one uploaded JSON file or
  one inline field, not both

`POST /v1/vision/image-to-3d` accepts:

- `image`: exactly one input image file part
- `model`: only `image-3d-triposr`
- `resolution`: integer 2 through 512; defaults to 256
- `density_threshold`: defaults to 25
- `foreground_ratio`: greater than 0 and at most 1; defaults to 0.85
- `already_framed`: boolean; defaults to false
- `vertex_colors`: boolean; defaults to true

`POST /v1/vision/image-to-3d-multiview` accepts:

- `image` / `image[]`: exactly 4 or 6 non-empty view file parts
- `model`: only `image-3d-instantmesh-base`
- `resolution`: integer 2 through 256; defaults to 128
- `vertex_colors`: boolean; defaults to true
- `cameras`: optional `schemaVersion` 1 JSON with one 16-value camera per
  uploaded view

`POST /v1/vision/depth-video` accepts:

- `video`: exactly one input video file part
- `model`: `vision-depth-vda-small` (default) or `vision-depth-vda-small-metric`
- `input_size`: integer 14 through 1,008; defaults to 518
- `max_frames`: integer 1 through 2,400; defaults to 240

Uploads are capped at 100 MiB for the single-image geometry and image-to-3d
routes and 512 MiB for the multi-view and depth-video routes.

## Open WebUI companion

Open WebUI can use mere.run as an OpenAI-compatible provider without being
vendored into this repo. Start mere.run, run the official Open WebUI Docker or
pip install, then configure Open WebUI with the mere.run `/v1` base URL.

One-command Docker quickstart:

```bash
mere.run open-webui quickstart --pull --accept-model-license
```

That starts `api serve`, runs the official Open WebUI container, configures the
OpenAI provider, filters the chat picker to the configured text and vision chat
models, points RAG at `/v1/embeddings`, and keeps image editing disabled for the
live-smoke path. Preview the exact commands with:

```bash
mere.run open-webui quickstart --dry-run
```

The macOS Studio passes `MERERUN_API_KEY` and
`MERERUN_OPEN_WEBUI_ADMIN_PASSWORD` in the child environment so neither secret
appears in the process argument list. CLI users may use the same variables;
explicit `--api-key` and `--admin-password` values still take precedence.

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

Fair FIFO request admission is part of the runtime pool now, and Gemma4,
Qwen-family, and LFM2 chat use engine-specific chunked prefill checkpoints.
Matching text prefixes reuse in-memory KV by default for all three families;
Qwen-family vision requests remain excluded, and the engine-specific prefix
variables can disable reuse for an A/B. Decode batching also engages for all
three when `--max-active-requests` is above `1`, with force-on/force-off
environment overrides. The implementations merge compatible typed cache rows
for actual batched decode calls, then split them so each request retains its own
state. Qwen-family and LFM2 full-attention rows can batch across positions with
row-offset-aware ragged KV caches; Qwen-family linear state and LFM2
short-convolution state are batched when their typed shapes are compatible.
Gemma4 full-attention rows still require matching scalar offsets. SSD KV
persistence remains later, measured work; use `cacheStats` plus
`benchmarkStats` to decide whether that experiment is worth expanding on a real
machine.

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
