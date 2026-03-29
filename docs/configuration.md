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

### `MERERUN_MODEL_SOURCE_BASE_URL`

Sets the explicit base URL used by `mere.run model pull` for unsigned public model archives.

```bash
export MERERUN_MODEL_SOURCE_BASE_URL=https://your-host.example/models/
swift run mere.run model pull image-zimage-max
```

Alternative download configuration is also supported for signed downloads:

- `MERERUN_R2_SIGNED_URL_ENDPOINT`
- `MERERUN_R2_SIGNED_URL_BEARER_TOKEN`
- `MERERUN_R2_SIGNED_URL_REQUIRED`
- `MERERUN_R2_ACCOUNT_ID`
- `MERERUN_R2_ACCESS_KEY_ID`
- `MERERUN_R2_SECRET_ACCESS_KEY`
- `MERERUN_R2_BUCKET`

## Specialized model roots

### `MERERUN_VIDEO_LTX_MODEL_ROOT`

Sets the default root used by `mere.run video generate` and `mere.run video export-latents` when `--model-root` is omitted.

### `MERERUN_MUSIC_ACESTEP_ROOT`

Sets the default checkpoint root used by `mere.run music generate` when the command is not resolving from the shared model store.

## Debug toggles

These are quiet by default and are intended for troubleshooting deeper runtime paths.

- `MERERUN_FLUX2_DEBUG=1`
- `MERERUN_ZIMAGE_DEBUG=1`
- `MERERUN_OCR_DEBUG=1`
- `MERERUN_LORA_DEBUG=1`
- `MERERUN_VIDEO_LTX_DEBUG_DENOISE=1`
- `MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX=/tmp/mererun-ltx`
