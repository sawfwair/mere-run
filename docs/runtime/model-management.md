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
- `mere.run status`
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

- images: `image-klein-nano`, `image-bonsai-binary`, `image-bonsai-ternary`, `image-zimage-nano`, `image-klein-max`, `image-zimage-max`
- text: `text-chat-gemma4`, `text-chat-q36-nano`, `text-chat-lfm25-a1b-8bit`, `text-agent-deepseek-v4-flash`, `text-agent-qwen35-9b`, `text-agent-ornith-9b`, `text-agent-ornith-35b-mlx`, `text-agent-ornith-35b`, `text-code-north-mini`, `text-code-qwen3`, `text-embed-qwen3-0.6b`
- speech: `speech-tts-qwen3-nano`, `speech-asr-parakeet`
- vision: `vision-ocr-lighton`
- music: `music-acestep`, `music-acestep-xl-turbo`, `music-acestep-xl-turbo-lm4b`, `music-magenta-rt2-small`, `music-magenta-rt2-base`
- sfx: `sfx-woosh-dflow`, `sfx-woosh-flow`, `sfx-woosh-clap`, `sfx-woosh-synchformer`, `sfx-woosh-dvflow-8s`, `sfx-woosh-vflow-8s`
- video: `video-ltx-av`, `video-ltx23-av-mlx`, `video-wan22-ti2v-5b-mlx`

The public runtime resolves these IDs directly, so docs and examples should use
the canonical names shown by `mere.run model list`.

## Runtime entrypoints

- `Sources/MereRunCore/MereRunModelPaths.swift`
- `Sources/MereRunCore/MereRunModelManifest.swift`
- `Sources/MereRunCore/ModelResolver.swift`

## Command responsibilities

### `mere.run model list`

Shows the canonical managed model table and installed status.

### `mere.run status`

Combines the model inventory with a local API probe. It reports the active
model-store path/source, installed managed models, whether the configured API
server answers `/health`, and the model IDs returned by `/v1/models`.

### `mere.run model capabilities`

Summarizes the current machine, the managed models it can run, the preferred
setup-agent tier, chat winners by RAM band, and cross-modality starter coverage.
Pass `--all` to include models that are blocked by platform or memory
requirements.

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
setup can select Qwen3.6 nano, Qwen3-Coder Next, or DeepSeek V4 Flash. On 96 GB+
Apple Silicon Macs, `text-agent-deepseek-v4-flash` is the preferred managed
setup-agent tier; smaller Qwen agent models are alternatives, not upgrades.
`text-code-north-mini` can be pulled, inspected, and run through the native
GGUF code runtime for coding-agent comparisons against `text-code-qwen3`.
`text-agent-ornith-9b` can be pulled, inspected, and run through the native
Qwen-family MLX/OptiQ runtime for coding-agent comparisons.
`text-agent-ornith-35b-mlx` is a local-only converted MLX Q4 Ornith target; it
can be inspected and served once its converted directory exists in the model
store, but `model pull` is disabled until a converted public snapshot exists.
`text-agent-ornith-35b` is the larger GGUF Ornith eval target and runs through
the native `text-code`/llama.cpp path.

### `mere.run model remove`

Removes a managed install from the local store.

### `mere.run model repair-manifests`

Repairs manifest metadata in the local store when that metadata is missing or
stale.

## Related docs

- [Model Sources](../model-sources.md)
- [Configuration](../configuration.md)
- [Testing Guide](../testing.md)
