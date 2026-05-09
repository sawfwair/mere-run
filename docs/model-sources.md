# Model Sources

This repo supports two model paths:

1. Hugging Face snapshots pulled into the local mere.run model store with `mere.run model pull`
2. Explicit local paths passed to commands with `--model`, `--model-root`, or equivalent command-specific options

The public repo no longer supports private model archives, R2 credentials, or a
packaged central model host.

The canonical local model store is:

```text
~/Library/Application Support/MereRun/models
```

Override that with `MERERUN_MODELS_DIR` or `--models-root`.

## Canonical Managed Model IDs

`mere.run model pull` works for catalog entries that have a Hugging Face source:

| Category | Hugging Face pull IDs |
| --- | --- |
| Image | `image-klein-nano`, `image-klein-base`, `image-klein-max`, `image-zimage-nano`, `image-zimage-base`, `image-zimage-max` |
| Text chat | `text-chat-gemma4`, `text-chat-gemma4-nano`, `text-chat-gemma4-max`, `text-chat-q35`, `text-chat-q35-nano` |
| Text code / agents | `text-agent-qwen35-9b`, `text-code-qwen3` |
| Text embed | `text-embed-qwen3-0.6b` |
| Text anonymize | `text-anonymize-privacy-filter` |
| Speech TTS | `speech-tts-qwen3-nano`, `speech-tts-qwen3-customvoice` |
| Speech ASR | `speech-asr-qwen3`, `speech-asr-parakeet` |
| Vision | `vision-ocr-lighton`, `vision-segment-sam31`, `vision-ground-falcon-perception` |
| Music | `music-acestep` |
| Video | `video-ltx-av` |

Some legacy/local IDs remain in the catalog so existing installs and explicit
local paths keep working:

```text
image-klein-shared
text-chat-mebot
text-chat-psi-agent
```

`image-klein-shared` is an internal shared-component install shape, and the
text-chat IDs listed here remain local-path-only until they have public Hugging
Face sources.

## Hugging Face Cache

Hub snapshots use the shared cache location managed by the runtime. Override it
when you want large models on an external disk:

```bash
export MERERUN_HUB_CACHE=/Volumes/Models/huggingface
swift run mere.run model pull image-zimage-max
```

Model pulls are resumable at the file level through the Hugging Face snapshot
cache. The CLI writes a managed-model symlink from the mere.run model store to
the prepared snapshot when needed.

## Hardware Support Checks

Managed pulls are gated by the local capability catalog before any download. The
check uses Apple Silicon macOS plus unified-memory thresholds for each model
family, then blocks models that are unlikely to run reliably on the current Mac.

Inspect the local recommendation first:

```bash
swift run mere.run model capabilities
swift run mere.run model capabilities --all
```

If you are intentionally testing an unsupported setup, pass
`--allow-unsupported` to `mere.run model pull`.

## Model Store Behavior

The CLI resolves models in this order:

1. `--models-root` process override
2. `MERERUN_MODELS_DIR`
3. persisted local model-store setting
4. default `~/Library/Application Support/MereRun/models`

Examples:

```bash
# Pull into the default model store
swift run mere.run model pull image-zimage-max

# Pull into a custom SSD-backed store
MERERUN_MODELS_DIR=/Volumes/Models swift run mere.run model pull text-chat-q35

# Inspect what is currently installed
swift run mere.run model list
swift run mere.run model info image-klein-max
```

## Music And Video Layouts

Two retained surfaces have more structure than a flat model root.

### `music-acestep`

The top-level model root is:

```text
.../models/music-acestep
```

That root may contain:

- `music-acestep-v15-turbo/`
- `music-acestep-5hz-lm-1.7b/` or another supported LM subdirectory
- `Qwen3-Embedding-0.6B/`
- `vae/`

`mere.run music generate` auto-discovers that layout unless you override the root
with `--checkpoints-root` or `MERERUN_MUSIC_ACESTEP_ROOT`.

### `video-ltx-av`

The unified AV model root is:

```text
.../models/video-ltx-av
```

`mere.run video generate --variant unified-av` can use that root directly or
resolve it from `MERERUN_VIDEO_LTX_MODEL_ROOT`.
