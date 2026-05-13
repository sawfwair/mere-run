# Model Management

This page covers the shared model store, canonical model IDs, manifests, and
the model-management commands.

## Public surface

- `mere.run model list`
- `mere.run model capabilities`
- `mere.run model info`
- `mere.run model pull`
- `mere.run model remove`
- `mere.run model repair-manifests`
- `mere.run setup`

## Default model store

By default:

```text
~/Library/Application Support/MereRun/models
```

Override with:

```bash
export MERERUN_MODELS_DIR=/path/to/models
```

or:

```bash
swift run mere.run --models-root /path/to/models model list
```

## Canonical model IDs

Examples:

- images: `image-klein-nano`, `image-zimage-nano`, `image-klein-max`, `image-zimage-max`
- text: `text-chat-gemma4`, `text-chat-q35`, `text-chat-q35-nano`, `text-agent-deepseek-v4-flash`, `text-agent-qwen35-9b`, `text-code-qwen3`, `text-embed-qwen3-0.6b`
- speech: `speech-tts-qwen3-nano`, `speech-asr-parakeet`
- vision: `vision-ocr-lighton`
- music: `music-acestep`
- video: `video-ltx-av`

The public runtime resolves these IDs directly, so docs and examples should use
the canonical names shown by `mere.run model list`.

## Runtime entrypoints

- `Sources/MereRunCore/MereRunModelPaths.swift`
- `Sources/MereRunCore/MereRunModelManifest.swift`
- `Sources/MereRunCore/ModelResolver.swift`

## Command responsibilities

### `mere.run model list`

Shows the canonical managed model table and installed status.

### `mere.run model capabilities`

Summarizes the current Mac, the managed models it can run, the preferred
setup-agent tier, and cross-modality starter coverage. Pass `--all` to include
models that are blocked by Apple Silicon or unified-memory requirements.

### `mere.run model info`

Shows the resolved local install for one canonical model ID.

### `mere.run model pull`

Downloads a managed model from its cataloged Hugging Face source. Pulls are
checked against the managed capability catalog before download so low-memory
machines do not fetch models they cannot run. Pass `--allow-unsupported` only
when you intentionally accept that risk or are using external hardware.

### `mere.run setup`

Guided onboarding for the shared model store and first local agent. The command
offers a Pi-powered Mere agent, a BYOA prompt for Claude/Codex, or manual
commands. The small local agent model is `text-agent-qwen35-9b`; hardware-tier
setup can select Q35 nano, Qwen3-Coder Next, or DeepSeek V4 Flash. On 96 GB+
Apple Silicon Macs, `text-agent-deepseek-v4-flash` is the preferred managed
setup-agent tier; Q35/Qwen agent models are alternatives, not upgrades.

### `mere.run model remove`

Removes a managed install from the local store.

### `mere.run model repair-manifests`

Repairs manifest metadata in the local store when that metadata is missing or
stale.

## Related docs

- [Model Sources](../model-sources.md)
- [Configuration](../configuration.md)
- [Testing Guide](../testing.md)
