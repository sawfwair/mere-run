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

Sets the default checkpoint root used by `mere.run music generate` and
`mere.run music analyze` when the command is not resolving from the shared model
store.

## API server security

### `MERERUN_API_KEY`

Provides the bearer token accepted by `mere.run api serve` for `/v1/models`,
`/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`,
`/v1/images/edits`, `/v1/audio/speech`, and `/v1/audio/transcriptions`.

This is optional for loopback-only usage and required for non-loopback binds.
`mere.run status` also reads it when probing `/v1/models`.

## Runtime experiments

### `MERERUN_GEMMA4_PREFIX_KV_CACHE`

Gemma4 in-memory prefix KV reuse is enabled by default in `mere.run api serve`.
Set this to `0`, `false`, `no`, or `off` to disable it for a baseline run. The
runtime stores bounded, forked Gemma4 prompt-prefix cache state for matching
token prefixes and reports entries, hits, and reused tokens through
`/runtime/status`. When the final chat message changes but the earlier
system/tool/chat prefix is identical, Gemma4 stores that semantic prefix as an
extra checkpoint before continuing normal prefill chunks. The bounded cache
keeps semantic checkpoints ahead of ordinary chunk checkpoints when pruning.

Continuous batching and SSD KV cache are not enabled by this flag.

### `MERERUN_LFM2_PREFIX_KV_CACHE`

Opt-in (`1`): in-memory prompt-prefix reuse for the LFM2 chat runtime,
mirroring the Qwen-family implementation. Forked layer caches (attention KV
and short-conv states both support forking) are stored at prefill chunk
boundaries and the longest matching token prefix seeds later requests, so a
repeated or extended prompt re-prefills only its tail. Chunk-boundary
checkpoints only — Gemma4-style semantic chat-template checkpoints are not
yet derived for LFM2. Bounded to 4 entries with the shared retention planner.
Measured on a ~2.9k-token prompt: repeat requests drop from 11.7s to 0.3s
end to end.

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

### `MERERUN_GEMMA4_MTP_SAMPLED`

Opt-in (`1`, `true`, or `on`) Gemma 4 12B MTP for sampled (temperature > 0)
requests. The verify loop samples the target model at every drafted position
and emits either the matching draft token or the target's own sample, so
sampled outputs remain true target-model samples; drafts run greedily to
maximize the match rate. Off by default: with the current assistant the
acceptance economics measured below the pipelined sampled decode path at long
context.

### `MERERUN_Q35_FUSED_SWITCH_GLU`

Qwen-family MoE blocks stack the gate and up expert weights so each
`SwitchGLU` issues one gather matmul instead of two. Enabled by default; set
to `0`, `false`, or `off` to fall back to separate gate/up gathers. The stack
keeps a second resident copy of the gate/up expert weights.

### `MERERUN_Q35_FUSED_QKV`

Qwen-family attention concatenates the q/k/v quantized projection weights so
each attention call issues one fused matmul instead of three. Quantized
packing is per-output-row, so results are bit-identical to the separate
projections (greedy outputs verified byte-equal). Enabled by default; set to
`0`, `false`, or `off` to fall back. On the 35B MoE this measured +1% decode
throughput for roughly 130 MB of additional resident weight copies —
attention is a small slice of a MoE's per-token weights, unlike the dense
Gemma case where the same fusion bought +17%.

### `MERERUN_Q35_BATCHED_GPU_SAMPLING`

Qwen-family continuous-batching decode samples every active request's row on
GPU (the same sampler the serial pipelined path uses, including the on-GPU
repetition window) and reads the whole batch back in a single sync per step.
The legacy path sampled per row on the host — one blocking GPU→CPU readback
per request per token, scaling linearly with serve concurrency. Enabled by
default; set to `0`, `false`, or `off` to restore per-row host sampling.

### `MERERUN_GEMMA4_FUSED_PROJ`

Gemma4 concatenates the q/k/v and gate/up quantized projection weights after
load so decode issues one fused matmul instead of two or three per group
(measured +17% decode throughput on the 12B MTP long-context path). Enabled by
default; set to `0`, `false`, or `off` to fall back to separate projections.
Fusion trades roughly 4 GB of additional resident weight copies for the fused
matmuls on the 12B and is skipped automatically while a text LoRA adapter
wraps the affected projection modules.

### `MERERUN_GEMMA4_FUSED_DECODE_KERNELS`

Opt-in (`1`, `true`, or `on`) custom fused Metal kernels for the elementwise
chains between matmuls on Gemma4 single-token decode (QKV head split plus
q/k/v norms, post-attention norm plus residual plus pre-FFN norm, gelu·up over
the fused gate/up buffer, and post-FFN norm plus residual plus layer scalar).
Off by default: throughput is neutral on an idle GPU and the kernels' float32
single-rounding numerics reduce Gemma MTP speculative acceptance at long
context. They cut per-token kernel dispatches roughly in half, which helps
when the GPU is shared with other heavy work (for example concurrent
training) — enable explicitly for that scenario.

### `MERERUN_GEMMA4_COMPILED_SEGMENTS`

Opt-in (`1`, `true`, or `on`) MLX-compiled per-layer decode segments. Off by
default: with mlx-swift 0.31.4 every compiled call serializes on the global
eval lock, which measured slower than the interpreted path at decode call
rates. Kept for evaluation against future mlx-swift releases.

### `MERERUN_MAGENTA_NONBLOCKING_PROMPT_SWAP`

Opt-in (`1`, `true`, or `on`): Magenta RT2 mid-session prompt swaps keep the
render loop generating frames on the previous prompt while the engine encodes
the new one on its own thread, switching when ready, instead of blocking the
render thread in a status poll for the whole encode (a guaranteed underrun
for live audio). Off by default until the vendored engine's Metal-library
regression is fixed and the gated
`MagentaRT2PromptSwapTests.testNonBlockingPromptSwapKeepsRendering`
integration test can validate the behavior live. The blocking wait's poll
interval is also reduced from 10ms to 1ms.

### `MERERUN_SAMPLER_TOP_P_PREFILTER`

GPU-side top-p sampling prefilters to this many top-logit candidates (via
argPartition) before running the softmax/sort/cumsum chain, replacing a
full-vocabulary sort per sampled token. Defaults to `256`; set `0` for the
exact full-vocabulary sort. The truncation only affects requests whose top-p
nucleus would span more than this many tokens, which does not occur at
practical temperatures.

### `MERERUN_LORA_TRAIN_GRAD_CHECKPOINT`

Gradient checkpointing for image-LoRA training — Krea 2 and FLUX.2 Klein.
Transformer blocks recompute their activations during backward instead of
retaining every intermediate. Unset, the default is resolution-aware: it
engages when the peak training resolution (after `--max-resolution` capping)
reaches 768×768-equivalent pixels, and stays off below that. Activation memory
scales with the token count, so capped-resolution recipes (for example
`klein-fast-style` at 512, or 768×416 runs) fit comfortably without the
recompute overhead, while a single full-injection step at 1024×1024 peaks
around 159 GB on the Krea 2 backbone and crashes outright on Klein base
(183 GB peak observed) on 128 GB machines. Set `1` to force checkpointing on
everywhere, or `0`/`false`/`off` to force it off. The explicit
`--gradient-checkpointing` CLI flag still forces it on regardless of the
environment. Checkpointed training runs the step uncompiled. Each trainer
prints its decision at startup (`grad_checkpoint=` on stderr).

### `MERERUN_LORA_TRAIN_CACHE_LIMIT_GB`

MLX buffer-cache cap during image-LoRA training, in gigabytes (default `16`;
`0` leaves the cache unlimited). The cache grows to the transient high-water
mark and never shrinks, so without a cap the process footprint stays pinned at
the worst spike of the run. Applies to both image-LoRA trainers.

### `MERERUN_LORA_TRAIN_SYNC_EVAL`

Set to `1` to evaluate each image-LoRA training step synchronously. The
default overlapped evaluation lets the next step's graph (and its full
activation set) go live while the current step executes, which can nearly
double the peak memory footprint at training-sized activations. Synchronous
mode caps in-flight activations at one step's worth at the cost of graph-build
overlap, which is negligible next to a training step.

### `MERERUN_LORA_TRAIN_SAVE_EVERY`

Adapter checkpoint cadence, in optimizer steps, for image-LoRA training
(default `100`; `0` disables). A `<name>.partial.safetensors` file is written
next to the output and removed once the final save succeeds, so a run killed
mid-training — for example by memory pressure — no longer loses everything.
The trainer also logs `step_s` and `footprint_gb` diagnostics at the metrics
cadence.

### `MERERUN_TEXT_LORA_TRAIN_GATHERED_LOSS`

Native text LoRA training (`text train-lora`) projects only loss-masked
target positions through the 262k-vocabulary lm_head and computes cross
entropy as logSumExp-minus-gather, instead of materializing full-sequence
logits plus a second full-vocabulary log-probability tensor. Gradients are
identical to the full path — prompt and padding rows never contribute loss —
so this is on by default; set `0`, `false`, or `off` to restore the legacy
full-logits loss. The trainer prints its decision at startup
(`gathered_loss=` on stderr).

### `MERERUN_TEXT_LORA_TRAIN_LOG_EVERY`

Loss-readback cadence for text LoRA training, in optimizer steps (default
`10`). Between boundaries steps are scheduled with asyncEval and the loop
continues without a GPU→CPU sync, so the next step's graph construction
overlaps execution. Boundary steps read the loss, update the metrics CSV and
progress, and print `[text-lora-train] step= loss= step_s= footprint_gb=` to
stderr. Set `1` for the legacy per-step synchronous readback. The shared
image/text training knobs (`MERERUN_LORA_TRAIN_CACHE_LIMIT_GB`,
`MERERUN_LORA_TRAIN_SAVE_EVERY`, `MERERUN_LORA_TRAIN_SYNC_EVAL`) also apply:
the buffer-cache cap defaults to 32 GB for text training (a sub-working-set
cap doubles step time at ~900-token sequences, while uncapped the cache
balloons past 100 GB), and a `<name>.partial.safetensors` adapter checkpoint
is written every 100 steps and removed after the final save.

### `MERERUN_STT_DECODE_WINDOW`

Parakeet transducer decoding (TDT and RNN-T) evaluates the joint network for
this many contiguous encoder frames per batched call (default `16`). The
decoder state only changes when a token is emitted, so blank frames — the
majority — scan host-side from a single readback instead of one (RNN-T) or
two (TDT) scalar readbacks per frame. Greedy semantics are identical to the
per-frame loop. Set `1` to restore the legacy per-frame readback cadence.

### `MERERUN_STT_PIPELINED_DECODE`

Qwen3 ASR token decoding samples on GPU and pipelines the loop at depth 1:
the sampled token feeds the next forward as a GPU array and the previous
step's token is read back while the current step executes (the legacy loop
synchronized twice per token). Enabled by default; set `0`, `false`, or
`off` to restore the legacy loop.

### `MERERUN_GEMMA4_DECODE_TRACE`

Set to `1` to log a per-decode summary to stderr splitting each token's wall
time into graph build, sampling, and eval scheduling, plus the readback wait.
Useful for locating whether decode is CPU-, schedule-, or GPU-bound.

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

Qwen-family text-only prefix KV reuse is enabled by default in
`mere.run api serve`. Set this to `0`, `false`, `no`, or `off` to disable it for
a baseline run. Vision prompts are excluded because image embeddings change the
effective prefix even when the token ids match. Runtime status uses the same
prefix KV counters as Gemma4. Text-only Qwen-family requests also store the
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
