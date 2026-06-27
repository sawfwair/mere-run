# Text Runtime

This page covers the text-facing command families: chat, code generation,
embeddings, and PII anonymization.

## Public surface

- `mere.run text chat`
- `mere.run text code`
- `mere.run text embed`
- `mere.run text anonymize`

## Model families

### Chat

- `text-chat-gemma4`
- `text-chat-gemma4-12b` (managed dense Google Gemma 4 12B-it snapshot)
- `text-chat-gemma4-12b-4bit` (managed MLX 4-bit Gemma 4 12B-it snapshot)
- `text-chat-gemma4-turbo` (managed MLX NVFP4 Gemma 4 26B-A4B MoE snapshot)
- `text-chat-q36-nano`
- `text-chat-lfm25-a1b-8bit` (managed LiquidAI LFM2.5 8B-A1B MLX 8-bit snapshot)
- `text-agent-deepseek-v4-flash` (API/agent serving)
- `text-chat-mebot`
- `text-chat-psi-agent`

### Code

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
Use these chat winners by RAM band instead of treating every supported model as
equally recommended:

| Unified memory | Winner | Notes |
| --- | --- | --- |
| 16-23 GB | `text-chat-gemma4-12b-4bit` | Best compact first chat pick; `text-chat-gemma4-nano` is the safer smallest fallback. |
| 24-63 GB | `text-chat-q36-nano` | Default strong chat tier; `text-chat-gemma4-turbo` is the Gemma-specific alternative. |
| 64-95 GB | `text-chat-q36-nano` | Extra RAM is better spent on context, concurrency, or code models than dense Gemma 31B as a default. |
| 96+ GB | `text-agent-deepseek-v4-flash` | Premier agent/API chat tier; keep Q36 for lower-latency interactive chat. |

`text-chat-lfm25-a1b-8bit` installs `LiquidAI/LFM2.5-8B-A1B-MLX-8bit`
and runs through the native Swift LFM2 runtime. It is text-only; use
`--model text-chat-lfm25-a1b-8bit` for CLI chat or `api serve --engine text-chat-lfm2`.

### Local code generation

```bash
swift run mere.run text code \
  --prompt "Write a Swift function that reverses a string."
```

`text-code-qwen3` and `text-code-north-mini` are native `text code` models and
run GGUF weights through llama.cpp. North Mini Code uses the Unsloth
`North-Mini-Code-1.0-UD-Q4_K_M.gguf` quant, so it requires a llama.cpp runtime
with `cohere2moe` architecture support.

```bash
swift run mere.run model pull text-code-north-mini
swift run mere.run text code \
  --model text-code-north-mini \
  --prompt "Write a Swift function that reverses a string."
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

## Runtime entrypoints

### CLI

- `Sources/MereRunCLI/Commands/TextChatCommand.swift`
- `Sources/MereRunCLI/Commands/TextCodeCommand.swift`
- `Sources/MereRunCLI/Commands/TextEmbedCommand.swift`
- `Sources/MereRunCLI/Commands/TextAnonymizeCommand.swift`

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
model families.

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

## Reading the code

If you want to understand the text stack:

1. start at the matching CLI command
2. follow model resolution through `MereRunModelManifest` and `ModelResolver`
3. jump to the family-specific runtime in `MereRunCore`

For repo orientation, pair this page with [CLI and Runtime Internals](../internals/cli-and-runtime.md).
