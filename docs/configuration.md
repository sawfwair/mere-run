# Configuration

These are the public runtime environment variables that matter in the OSS repo.

## Model store

### `MERERUN_MODELS_DIR`

Overrides the shared local model store.

```bash
export MERERUN_MODELS_DIR=/Volumes/Models/mere.run
swift run mere.run status
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
swift run mere.run model pull image-zimage-nano
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
`mere.run status` also reads it when probing `/v1/models`.

## Runtime experiments

### `MERERUN_GEMMA4_PREFIX_KV_CACHE`

Set to `1` to enable the in-memory Gemma4 prefix KV reuse prototype in
`mere.run api serve`. This stores bounded, forked Gemma4 prompt-prefix cache
state for matching token prefixes and reports entries, hits, and reused tokens
through `/runtime/status`. When the final chat message changes but the earlier
system/tool/chat prefix is identical, Gemma4 stores that semantic prefix as an
extra checkpoint before continuing normal prefill chunks. The bounded cache
keeps semantic checkpoints ahead of ordinary chunk checkpoints when pruning.

Continuous batching and SSD KV cache are not enabled by this flag.

### `MERERUN_GEMMA4_MTP`

Gemma 4 12B MTP is enabled by default when the managed
`text-chat-gemma4-12b-mtp` assistant companion is installed. Set this to `0`,
`false`, or `off` to force baseline decode. The runtime only uses Gemma MTP for
greedy serial decode after the main Gemma 12B model has prefetched the prompt and
exposed hidden state plus shared KV; sampled requests, continuous batching, raw
local model paths, and prefix-KV seeded requests stay on baseline decode.

### `MERERUN_GEMMA4_MTP_MIN_PROMPT_TOKENS`

Minimum effective prompt length before Gemma 4 12B MTP is considered. Defaults
to `2048`.

### `MERERUN_GEMMA4_MTP_BLOCK_SIZE`

Override the Gemma 4 12B assistant draft block size. Defaults to the assistant
config value, currently `4`, and is clamped to the native runtime's supported
range.

### `MERERUN_Q35_MTP_SPECULATION`

Controls the Qwen-family MTP path used by `text-chat-q36-nano`. Set this to
`1`, `true`, `yes`, or `on` to force consideration when the effective context
window is large enough; set it to `0`, `false`, or `no` to disable MTP. Any other
value, including unset, uses the adaptive long-context threshold.

The `Q35` name is an internal compatibility prefix for the Qwen-family runtime;
the public managed model id is `text-chat-q36-nano`.

### `MERERUN_Q35_MTP_MIN_PROMPT_TOKENS`

Minimum effective prompt length before Qwen-family MTP is considered. Defaults
to `6144`, and the effective request context must also be at least this large.

### `MERERUN_Q35_MTP_BLOCK_SIZE`

Override the Qwen-family greedy MTP draft block size. Defaults to `4` and is
clamped to the native runtime's supported range.

### `MERERUN_Q35_PREFIX_KV_CACHE`

Set to `1` to enable the in-memory Qwen-family text-only prefix KV reuse prototype in
`mere.run api serve`. Vision prompts are excluded because image embeddings
change the effective prefix even when the token ids match. Runtime status uses
the same prefix KV counters as Gemma4. Text-only Qwen-family requests also store the
stable chat prefix before the final message as an extra checkpoint when it is an
exact token prefix of the full prompt, and the bounded cache gives those
semantic checkpoints the same pruning priority as Gemma4.

### `MERERUN_GEMMA4_CONTINUOUS_BATCHING`

Set to `1` to enable the Gemma4 same-offset decode batching prototype in
`mere.run api serve`. It packs overlapping Gemma4 decode rows with equal KV
offsets into typed batched KV caches, splits the cache rows back after each
step, and reports same-position batched decode steps, queued rows, and max
observed batch size through `/runtime/status`. Gemma4 variable-position decode
batching is not enabled because its current attention path applies RoPE with
scalar cache offsets.

### `MERERUN_Q35_CONTINUOUS_BATCHING`

Set to `1` to enable the Qwen-family decode batching prototype in `mere.run api serve`.
It uses the runtime's typed full-attention and linear-attention cache states. Full
attention can batch different decode positions through row-offset-aware ragged
KV caches, and linear attention can batch different decode positions through
typed recurrent state when cache shapes are compatible. Runtime status reports
same-position and variable-position batched decode steps.

This is deliberately narrower than arbitrary continuous batching: prefill still
runs as cancellable per-request chunks, and cache rows batch only when their
typed state proves compatibility. Gemma4 full-attention rows remain
same-position because that engine still uses scalar cache offsets. The scheduler
services the earliest decode position first by batching compatible rows there or
advancing one lower-offset row until it can join a compatible batch. The feature
needs `--max-active-requests` above `1` before requests can overlap.

## Debug toggles

These are quiet by default and are intended for troubleshooting deeper runtime paths.

- `MERERUN_FLUX2_DEBUG=1`
- `MERERUN_ZIMAGE_DEBUG=1`
- `MERERUN_OCR_DEBUG=1`
- `MERERUN_LORA_DEBUG=1`
- `MERERUN_VIDEO_LTX_DEBUG_DENOISE=1`
- `MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX=/tmp/mererun-ltx`
