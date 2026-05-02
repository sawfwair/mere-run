# Model Management

This page covers the shared model store, canonical model IDs, manifests, and
the model-management commands.

## Public surface

- `mere.run model list`
- `mere.run model info`
- `mere.run model pull`
- `mere.run model remove`
- `mere.run model repair-manifests`

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

- images: `image-klein-max`, `image-zimage-max`
- text: `text-chat-gemma4`, `text-chat-q35`, `text-code-qwen3`, `text-embed-qwen3-0.6b`
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
- `Sources/MereRunCLI/Support/R2ModelRegistry.swift`

## Command responsibilities

### `mere.run model list`

Shows the canonical managed model table and installed status.

### `mere.run model info`

Shows the resolved local install for one canonical model ID.

### `mere.run model pull`

Downloads a managed model from an explicit configured source. In the OSS repo,
it does not silently fall back to a baked-in hosted archive URL.

### `mere.run model remove`

Removes a managed install from the local store.

### `mere.run model repair-manifests`

Repairs manifest metadata in the local store when that metadata is missing or
stale.

## Related docs

- [Model Sources](../model-sources.md)
- [Configuration](../configuration.md)
- [Testing Guide](../testing.md)
