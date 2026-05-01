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
- `text-chat-q35`
- `text-chat-q35-nano`
- `text-chat-mebot`
- `text-chat-psi-agent`

### Code

- `text-code-qwen3`

### Embeddings

- `text-embed-qwen3-0.6b`

### Anonymization

- `text-anonymize-privacy-filter`

## Typical workflows

### Local chat

```bash
swift run mere.run text chat \
  --model text-chat-gemma4 \
  --prompt "Summarize diffusion models in one paragraph."
```

### Local code generation

```bash
swift run mere.run text code \
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
- `Sources/MereRunCore/Psi/`
- `Sources/MereRunCore/MeBot/`

### Code generation

`mere.run text code` uses the vendored `llama.cpp` runtime via
`vendor/llama.xcframework` and the matching support code in `MereRunCore`.

### Embeddings

- `Sources/MereRunCore/Embeddings/`

### Anonymization

- `Sources/MereRunCore/PrivacyFilter/`

## Practical distinctions

### `mere.run text chat`

Use this for local assistant-style generation. It supports prompt/system
control, token limits, and the chat-oriented model families.

### `mere.run text code`

Use this when you want a GGUF-backed coding path through the vendored
`llama.cpp` runtime.

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
