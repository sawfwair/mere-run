# Configuration

These are the public runtime environment variables that matter in the OSS repo.

## Model store

### `MERERUN_MODELS_DIR`

Overrides the shared local model store.

```bash
export MERERUN_MODELS_DIR=/Volumes/Models/mere.run
swift run mere.run model list
```

Default:

```text
~/Library/Application Support/MereRun/models
```

## Managed model downloads

### `MERERUN_HUB_CACHE`

Overrides the Hugging Face snapshot cache used by managed model pulls.

```bash
export MERERUN_HUB_CACHE=/Volumes/Models/huggingface
swift run mere.run model pull image-zimage-max
```

Managed pulls use cataloged Hugging Face repos only. Private archive hosts and
R2 credential flows are not part of the public distribution.

## Specialized model roots

### `MERERUN_VIDEO_LTX_MODEL_ROOT`

Sets the default root used by `mere.run video generate` and `mere.run video export-latents` when `--model-root` is omitted.

### `MERERUN_MUSIC_ACESTEP_ROOT`

Sets the default checkpoint root used by `mere.run music generate` when the command is not resolving from the shared model store.

## API server security

### `MERERUN_API_KEY`

Provides the bearer token accepted by `mere.run api serve` for `/v1/models` and
`/v1/chat/completions`.

This is optional for loopback-only usage and required for non-loopback binds.

## Debug toggles

These are quiet by default and are intended for troubleshooting deeper runtime paths.

- `MERERUN_FLUX2_DEBUG=1`
- `MERERUN_ZIMAGE_DEBUG=1`
- `MERERUN_OCR_DEBUG=1`
- `MERERUN_LORA_DEBUG=1`
- `MERERUN_VIDEO_LTX_DEBUG_DENOISE=1`
- `MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX=/tmp/mererun-ltx`
