# Text Runtime

This page covers the text-facing command families: chat, code generation,
embeddings, PII anonymization, and native text LoRA preparation.

## Public surface

- `mere.run text chat`
- `mere.run text code`
- `mere.run text embed`
- `mere.run text anonymize`
- `mere.run text train-lora`

## Model families

### Chat

- `text-chat-gemma4`
- `text-chat-gemma4-12b` (managed dense Google Gemma 4 12B-it snapshot)
- `text-chat-gemma4-12b-4bit` (managed MLX 4-bit Gemma 4 12B-it snapshot)
- `text-chat-gemma4-turbo` (managed MLX NVFP4 Gemma 4 26B-A4B MoE snapshot)
- `text-chat-q36-nano`
- `text-chat-bonsai-27b-1bit` (managed packed 1-bit dense Qwen3.6 27B vision/reasoning snapshot)
- `text-chat-bonsai-27b-2bit` (managed packed 2-bit ternary dense Qwen3.6 27B vision/reasoning snapshot)
- `text-chat-lfm25-a1b-8bit` (managed LiquidAI LFM2.5 8B-A1B MLX 8-bit snapshot)
- `text-agent-ornith-9b` (experimental native MLX/OptiQ coding-agent snapshot)
- `text-agent-ornith-35b-mlx` (local native MLX Q4 coding-agent snapshot)
- `text-agent-deepseek-v4-flash` (API/agent serving)
- `text-chat-mebot`
- `text-chat-psi-agent`

### Code

- `text-agent-ornith-35b`
- `text-code-north-mini`
- `text-code-qwen3`

### Embeddings

- `text-embed-qwen3-0.6b`

### Anonymization

- `text-anonymize-privacy-filter`

## Typical workflows

### Local chat

```bash
swift run mere.run text chat \
  --stream \
  --model text-chat-gemma4 \
  --prompt "Summarize diffusion models in one paragraph."
```

`text-chat-gemma4-12b` runs Google's dense Gemma 4 12B-it checkpoint through
the native Swift Gemma runtime. Managed pulls for `text-chat-gemma4-12b` and
`vision-chat-gemma4-12b` install the `google/gemma-4-12B-it-assistant`
companion as `text-chat-gemma4-12b-mtp`; when it is present, greedy serial
decode can use native MTP on the decode tail after text or multimodal prefill.
Sampled requests, prefix-KV seeded requests, continuous batching, raw local
model paths, and prompts below the MTP threshold fall back to baseline decode.
When no assistant is active, greedy requests use draft-model-free
prompt-lookup speculation by default: repeated context n-grams are drafted
and verified in bursts inside the pipelined loop — output-identical, free on
non-repetitive text, and worth ~2× when generation echoes the prompt
(`MERERUN_GEMMA4_PROMPT_LOOKUP=0` disables; see
[Configuration](../configuration.md#mererun_gemma4_prompt_lookup)).
Use these chat winners by RAM band instead of treating every supported model as
equally recommended:

| Unified memory | Winner | Notes |
| --- | --- | --- |
| 16-23 GB | `text-chat-gemma4-12b-4bit` | Best compact first chat pick; `text-chat-gemma4-nano` is the safer smallest fallback. |
| 24-63 GB | `text-chat-gemma4-12b-4bit` | Default grounded local-assistant tier; `text-chat-gemma4-turbo` is the larger Gemma alternate. |
| 64-95 GB | `text-chat-gemma4-12b-4bit` | Keep the proven Gemma assistant lane; spend headroom on context, concurrency, or larger Gemma alternates. |
| 96+ GB | `text-agent-deepseek-v4-flash` | Premier agent/API chat tier; keep Gemma 12B 4-bit for normal interactive local chat. |

`text-chat-lfm25-a1b-8bit` installs `LiquidAI/LFM2.5-8B-A1B-MLX-8bit`
and runs through the native Swift LFM2 runtime. It is text-only; use
`--model text-chat-lfm25-a1b-8bit` for CLI chat or `api serve --engine text-chat-lfm2`.
Concurrent serve workloads use ragged, cache-safe decode batching when
`--max-active-requests` is above `1` (`MERERUN_LFM2_CONTINUOUS_BATCHING`
overrides); rows at different prompt lengths share a forward only when every
attention and short-conv cache proves compatibility.

Gemma4, Qwen-family, and LFM2 API-serving settings can select explicit
`--kv-cache-mode affine4` or `--kv-cache-mode affine8` as long-context memory
controls relative to full-precision K/V. Qwen-family and LFM2 dequantize the
generic cache for attention, so these modes are not assumed faster. Gemma uses
its model-specific quantized KV path; `text-chat-gemma4-turbo` already defaults
to a smaller 4-bit TurboQuant cache, so forcing affine 8-bit can increase that
model's KV residency. `default` restores the engine/model/server default rather
than promising full precision.

`text-chat-bonsai-27b-1bit` and `text-chat-bonsai-27b-2bit` install the pinned
5.13 GB binary and 8.52 GB ternary Prism ML snapshots. They run packed low-bit
language weights plus a dense vision tower through the native Qwen-family
runtime. Both default to thinking-enabled generation and the published
temperature 0.7, top-p 0.95, and top-k 20 when those values are not set
explicitly. The models advertise 262,144 tokens; `text chat` selects that limit
automatically, and `--context-size` can set a smaller operational bound. Choose
1-bit for lower residency and faster decode or 2-bit for the larger ternary
checkpoint. On Linux CUDA, the pinned MLX fork executes supported 1-bit affine
projections directly from packed weights; the runtime retains its operation-
scoped dense fallback for unsupported shapes and keeps 2-bit and wider model
dispatch independent. For memory-constrained long-context work, opt into the
generic affine cache:

```bash
swift run mere.run model pull text-chat-bonsai-27b-1bit
swift run mere.run model pull text-chat-bonsai-27b-2bit
swift run mere.run text chat \
  --model text-chat-bonsai-27b-2bit \
  --context-size 262144 \
  --kv-bits 4 \
  --prompt "Summarize the key decisions in this context."
```

The local-path-only `text-chat-psi-agent` runtime has guarded compressed-MLA
and fused sparse-MoE controls. They remain opt-in until a repeatable public
checkpoint quality/throughput A/B is available; see the
[guarded acceleration audit](../internals/guarded-acceleration.md).

`text-agent-ornith-9b` installs an Ornith 1.0 9B OptiQ MLX snapshot and runs
through the native Qwen-family runtime. Use `text chat --model text-agent-ornith-9b`
or `api serve --engine text-chat-q36 --model text-agent-ornith-9b` for coding-agent
smoke tests.

`text-agent-ornith-35b-mlx` is the same native Qwen-family runtime lane for a
locally converted Ornith 1.0 35B Q4 MLX directory. It does not auto-download
from Hugging Face until a converted snapshot is published.

Both Ornith lanes are R1-style reasoning tunes and generate with thinking
enabled by default in `text chat` and `api serve` — without it the models
degenerate into repetition loops or signature echo on constrained prompts.
The reasoning stays hidden in `text chat` unless `--thinking` is passed;
`--no-thinking` disables reasoning generation entirely. When sampling options
are not set explicitly, these lanes also use the model's published
`generation_config.json` sampling (temperature 1.0, top-p 0.95, top-k 20)
instead of the generic chat defaults. Other Qwen-family lanes such as
`text-chat-q36-nano` keep the existing no-think default.

Native MLX Gemma and Qwen-family chat models support
`--response-format json_object`. The output must be one complete root object;
the prefix grammar checks nested objects and arrays, strings and escapes,
Unicode, numbers, booleans, null, commas, and colons before each token is
streamed. Qwen-family JSON mode forces thinking off and bypasses MTP,
continuous batching, and pipelined decoding. The GGUF Q36 model used by Linux
packages remains on llama.cpp and rejects JSON-object mode until a llama.cpp
grammar is wired.

```bash
swift run mere.run text chat \
  --model text-chat-q36-nano \
  --response-format json_object \
  --prompt 'Return an object with a name and an array of tags.'
```

### Tool use

`text chat` can run a local agentic tool loop. `--tools` takes a
comma-separated list of built-in tool names (`write_file`, `shell_exec`), and
`--tool-loop` enables the loop: generate, execute tool calls, feed results
back, continue, capped at 10 iterations. Tools execute inside a sandbox
directory — `--sandbox-dir` sets it; the default is a per-process temp
directory — and each call asks for interactive `[y/N]` approval by default.
`shell_exec` additionally requires `--allow-shell-exec`, and `write_file` only
targets absolute paths outside the sandbox with `--allow-absolute-tool-paths`.
`--auto-approve-tools` skips confirmation for `write_file` only; `shell_exec`
always requires interactive approval even when that flag is set.

```bash
swift run mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --prompt "Create hello.py that prints the current time, then run it." \
  --tools write_file,shell_exec \
  --tool-loop \
  --allow-shell-exec \
  --sandbox-dir ./work
```

### Vision input

`text chat` accepts `--image` with a local file path or a `data:image/...` URI
for vision-capable chat models such as Bonsai 27B or `vision-chat-gemma4-12b`.
The image attaches to the user prompt for multimodal prefill.

```bash
swift run mere.run text chat \
  --model text-chat-bonsai-27b-2bit \
  --image ./photo.png \
  --prompt "Describe what is in this photo."
```

### Local code generation

```bash
swift run mere.run text code \
  --prompt "Write a Swift function that reverses a string."
```

`text-code-qwen3`, `text-code-north-mini`, and `text-agent-ornith-35b` are
native `text code` models and run GGUF weights through llama.cpp. North Mini
Code uses the Unsloth `North-Mini-Code-1.0-UD-Q4_K_M.gguf` quant, so it
requires a llama.cpp runtime with `cohere2moe` architecture support. Ornith 35B
uses DeepReinforce's `ornith-1.0-35b-Q4_K_M.gguf` quant for larger coding-agent
evals.

```bash
swift run mere.run model pull text-code-north-mini
swift run mere.run text code \
  --model text-code-north-mini \
  --prompt "Write a Swift function that reverses a string."

swift run mere.run model pull text-agent-ornith-35b
swift run mere.run text code \
  --model text-agent-ornith-35b \
  --prompt "Write a compact Swift Result helper."
```

### Embeddings

```bash
swift run mere.run text embed \
  "semantic search query" \
  --pretty
```

### PII anonymization

```bash
swift run mere.run text anonymize \
  "My name is Alice Smith and my email is alice@example.com"
```

### Native text LoRA training

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --dry-run \
  --json
```

`text train-lora` is the native MereRun entrypoint for chat-style SFT JSONL.
The data format is one JSON object per line with `sources` and `messages`;
messages use the same `system`, `user`, and `assistant` roles as the local chat
runtime. `--dry-run` validates the dataset, fingerprints it, counts optional
eval prompts, and writes a `.manifest.json` next to the requested adapter path.

Without `--dry-run`, the command resolves the Gemma4 text model through the same
managed model store as chat, applies the Gemma chat template to each example,
masks loss to assistant tokens, injects native LoRA layers into attention
projections, and writes a `.safetensors` adapter plus manifest. Add
`--visualize` to start the same loopback LoRA training dashboard used by image
training; text runs write `run.json`, `*.events.jsonl`, `*.loss.csv`, and
`*.loss.html` beside the adapter so loss and training events can be inspected
while the optimizer runs. Keep local text fine-tuning in `mere.run` so the same
model ids, manifests, runtime constraints, and eval artifacts remain under the
MereRun command plane.

The optimizer projects only loss-masked target positions through the lm_head
(prompt and padding rows never contribute loss, so gradients are unchanged),
reads the loss back every 10 steps rather than per step, writes a
`<name>.partial.safetensors` checkpoint every 100 steps so a killed run can be
salvaged, and prints `[text-lora-train] step= loss= step_s= footprint_gb=`
diagnostics at each readback. `docs/configuration.md` documents the
`MERERUN_TEXT_LORA_TRAIN_*` and shared `MERERUN_LORA_TRAIN_*` environment
switches that tune or disable each behavior.

```bash
swift run mere.run text train-lora \
  --data ./pairs.seed.jsonl \
  --eval ./eval.prompts.jsonl \
  --output ./local-assistant.safetensors \
  --model text-chat-gemma4-12b-4bit \
  --visualize \
  --visualize-port 8787
```

Use the resulting adapter with native Gemma chat:

```bash
swift run mere.run text chat \
  --model text-chat-gemma4-12b-4bit \
  --lora ./local-assistant.safetensors \
  --prompt "What should this local assistant know?"
```

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/TextChatCommand.swift`
- `Sources/MereRunCLI/Commands/TextCodeCommand.swift`
- `Sources/MereRunCLI/Commands/TextEmbedCommand.swift`
- `Sources/MereRunCLI/Commands/TextAnonymizeCommand.swift`
- `Sources/MereRunCLI/Commands/TextTrainLoRACommand.swift`

### Chat families

- `Sources/MereRunCore/Q35/`
- `Sources/MereRunCore/Gemma4/`
- `Sources/MereRunCore/LFM2/`
- `Sources/MereRunCore/Psi/`
- `Sources/MereRunCore/MeBot/`

### Code generation

`mere.run text code` uses the vendored `llama.cpp` runtime via
`vendor/llama.xcframework` and the matching support code in `MereRunCore`. On
packaged Linux installs, the command uses the colocated `llama-cli` subprocess
so CUDA GGUF loads stay isolated from the MLX runtime in the Swift process.

### Embeddings

- `Sources/MereRunCore/Embeddings/`

### Anonymization

- `Sources/MereRunCore/PrivacyFilter/`

## Practical distinctions

### `mere.run text chat`

Use this for local assistant-style generation. It supports prompt/system
control, token limits, token streaming with `--stream`, and the chat-oriented
model families. Native MLX Gemma and Qwen-family models also support constrained
JSON objects with `--response-format json_object`.

### `mere.run text code`

Use this when you want a GGUF-backed coding path through the vendored
`llama.cpp` runtime. Linux packages run the same model family through the
bundled `llama-cli` subprocess when it is present.

### `mere.run text embed`

Use this for vector generation, semantic search prep, and other representation
tasks. It is not a generative command.

### `mere.run text anonymize`

Use this for local PII detection and redaction. It runs the OpenAI Privacy
Filter token-classification model through the native MLX runtime and can emit
plain redacted text or structured JSON spans.

### `mere.run text train-lora`

Use this to prepare and train Gemma-family text LoRA adapters from reviewed chat
SFT data. The first supported target lane is `text-chat-gemma4-12b-4bit` for
local assistant tuning.

## Reading the code

If you want to understand the text stack:

1. start at the matching CLI command
2. follow model resolution through `MereRunModelManifest` and `ModelResolver`
3. jump to the family-specific runtime in `MereRunCore`

For repo orientation, pair this page with [CLI and Runtime Internals](../internals/cli-and-runtime.md).
