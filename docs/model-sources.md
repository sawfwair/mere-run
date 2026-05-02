# Model Sources

This repo supports two model paths:

1. Managed archives pulled into the local mere.run model store with `mere.run model pull`
2. Explicit local paths passed to commands with `--model`, `--model-root`, or equivalent command-specific options

The canonical local model store is:

```text
~/Library/Application Support/MereRun/models
```

Override that with `MERERUN_MODELS_DIR` or `--models-root`.

## Canonical managed model IDs

`mere.run model pull` supports these OSS IDs:

| Category | Canonical IDs |
| --- | --- |
| Image | `image-klein-nano`, `image-klein-base`, `image-klein-max`, `image-zimage-nano`, `image-zimage-base`, `image-zimage-max` |
| Text chat | `text-chat-mebot`, `text-chat-psi-agent`, `text-chat-q35`, `text-chat-q35-nano` |
| Text code | `text-code-qwen3` |
| Text embed | `text-embed-qwen3-0.6b` |
| Speech TTS | `speech-tts-qwen3-nano`, `speech-tts-qwen3-customvoice` |
| Speech ASR | `speech-asr-qwen3`, `speech-asr-parakeet` |
| Vision OCR | `vision-ocr-lighton` |
| Vision segmentation / tracking | `vision-segment-sam31` |
| Music | `music-acestep` |
| Video | `video-ltx-av` |

Some commands also support local or upstream model layouts outside this managed table, but the IDs above are the only canonical managed names in this repo.

`text-chat-gemma4` is also a canonical chat model in the CLI, but it does not use
`mere.run model pull`. It resolves through the native Hugging Face snapshot path
on first use unless you point it at a local model root.

Useful environment variables for that path:

- `MERERUN_HUB_CACHE`: override the native Hugging Face snapshot cache path

`vision-segment-sam31` packages the native SAM 3.1 segmentation and tracking runtime used by `mere.run vision segment`, `mere.run vision track`, and `mere.run vision track-live`. Managed or local SAM roots are expected to contain:

- `config.json`
- `tokenizer/`
- `model.safetensors` or `model.safetensors.index.json`

The manifest for this package advertises both `vision_segmentation` and
`vision_tracking` capabilities.

## Archive resolution

Managed model downloads use the R2 request builder in `MereRunCore` and resolve in this order:

1. Signed URL lookup service
2. Direct SigV4 credentials
3. Explicit public archive base URL

Supported environment variables:

### Signed URL lookup

- `MERERUN_R2_SIGNED_URL_ENDPOINT`
- `MERERUN_R2_SIGNED_URL_BEARER_TOKEN`
- `MERERUN_R2_SIGNED_URL_REQUIRED`
- `MERERUN_R2_SIGNED_URL_ALLOW_FALLBACK`

When a signed URL endpoint is configured, lookup failures stop the download by
default. Set `MERERUN_R2_SIGNED_URL_ALLOW_FALLBACK=1` only when you explicitly
want to fall back to direct credentials or a public archive base URL.

### Direct SigV4 credentials

- `MERERUN_R2_ACCOUNT_ID`
- `MERERUN_R2_ACCESS_KEY_ID`
- `MERERUN_R2_SECRET_ACCESS_KEY`
- `MERERUN_R2_BUCKET`

### Explicit public archive base URL

- `MERERUN_MODEL_SOURCE_BASE_URL`
- `MERERUN_MODEL_SOURCE_SHA256S`

If you do not configure signed downloads or direct credentials, set:

```bash
export MERERUN_MODEL_SOURCE_BASE_URL=https://your-host.example/models/
export MERERUN_MODEL_SOURCE_SHA256S='models/image-zimage-nano.tar.gz=<sha256>'
```

Archive downloads verify SHA-256 digests when the managed catalog entry includes
one or when `MERERUN_MODEL_SOURCE_SHA256S` supplies one. The checksum variable
accepts comma-, semicolon-, or newline-separated `archive/key.tar.gz=<sha256>`
entries.

Without one of the three configuration paths above, `mere.run model pull` fails fast with a clear configuration error.

## Model store behavior

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

## Music and video layouts

Two retained surfaces have more structure than a flat model root:

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

`mere.run music generate` auto-discovers that layout unless you override the root with `--checkpoints-root` or `MERERUN_MUSIC_ACESTEP_ROOT`.

### `video-ltx-av`

The unified AV model root is:

```text
.../models/video-ltx-av
```

`mere.run video generate --variant unified-av` can use that root directly or resolve it from `MERERUN_VIDEO_LTX_MODEL_ROOT`.
