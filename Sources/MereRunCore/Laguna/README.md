# Laguna S 2.1

Native Swift/MLX support for Poolside's official
`Laguna-S-2.1-NVFP4-mlx` checkpoint and DFlash companion. The validated pair
is available as the opt-in managed model `text-chat-laguna-s-2-1`. The same
typed runtime also accepts the official five-shard `Laguna-XS-2.1-NVFP4-mlx`
checkpoint through an explicit local path; shard discovery comes from the
safetensors index, and shared-expert quantization is derived from that index
rather than assumed from the S layout.

## Files

- `LagunaConfig.swift` decodes and validates the official typed model,
  quantization, attention, mixture-of-experts, and YaRN configuration.
- `LagunaModel.swift` implements the hybrid full/sliding attention stack,
  per-head attention gates, dense and routed expert layers, NVFP4 projections,
  rotary embeddings, and decode caches.
- `LagunaTokenizerAndTemplate.swift` loads the official tokenizer and renders
  the checkpoint's chat and tool prompt contract.
- `LagunaGenerator.swift` verifies the local checkpoint layout, loads its
  sharded safetensors, runs chunked prefill and serial or ragged continuous
  decode, and owns streaming generation and acceleration metrics.
- `LagunaDFlashConfig.swift` strictly decodes the official companion model
  contract.
- `LagunaDFlashModel.swift` implements the official six-layer DFlash draft
  model and target-hidden-state context projection.
- `LagunaDFlashDecoder.swift` performs lossless speculative verification,
  including rejection-sampling recovery for non-greedy generation.
- `LagunaToolParser.swift` converts the checkpoint's GLM-style tool markup to
  mere.run's typed `ToolCall` output.

## Supported boundary

Pulling the target installs both immutable checkpoint revisions:

```bash
mere.run model pull text-chat-laguna-s-2-1 --accept-model-license
mere.run text chat \
  --model text-chat-laguna-s-2-1 \
  --prompt "Write a bounded Swift actor queue." \
  --stats
mere.run api serve --engine text-chat-laguna --max-active-requests 2
```

Laguna requires at least 96 GB unified memory and is not a setup or
hardware-aware default. The normal chat and serving routes use temperature
`1`, top-p `1`, top-k `20`, and min-p `0.02`; callers can override those
values explicitly.

The CLI benchmark commands accept `--laguna-path` as an explicit local-only
checkpoint override. Pass `--laguna-dflash-path` to evaluate the official
companion checkpoint and `--laguna-dflash-tokens` to control the proposal
length. The measured default is 12 proposals per round.
The resident DFlash benchmark accepts `--temperature`, `--top-p`, `--top-k`,
and `--min-p` so sampled target-only, forced-DFlash, and automatic-routing
performance can be compared at an exact decode length in one process.
Laguna lanes default to min-p `0.02`, the quality/richness winner on
the M4 Max gate. Pass `--min-p 0` for the official Poolside control. This
recommendation does not change sampling defaults for other managed models.
`--laguna-dflash-min-tokens` controls the length-aware router and defaults to
32 effective output tokens. Requests below that budget use target-only prefill
and decode, without building DFlash prompt context. Routed requests fall back
losslessly to target-only decode when draft acceptance is below 25% after the
first round or below 60% after the second. Reports expose routed, bypassed, and
fallback request counts. The chat lane also accepts `--concurrency`; values
above one enable the ragged target continuous-batching scheduler and report
batching plus DFlash counters in both text and JSON output. DFlash batching
remains a forced evaluation path because fewer physical batch forwards did not
produce lower wall time on the measured two-row workload.

The checkpoint is a catalog model but remains explicitly opt-in: it is never
selected by default, requires acceptance of its published usage terms, and
keeps explicit checkpoint overrides for evaluation and rollback.

Laguna's routed experts sort prefill and multi-token verification routes by
expert before issuing the NVFP4 gather matmuls. This is enabled by default for
64 or more routes; single-token target decode keeps the lower-overhead unsorted
path. Set `MERERUN_LAGUNA_SORTED_MOE=0` to restore the reference routing order
for comparison or rollback. Do not lower the threshold without measuring
short verification and concurrent decode: two route sorts can cost more than
grouped expert locality saves on small batches.

On macOS 26 and the measured M4 Max GPU, BF16 sorted NVFP4 forwards of at
least 64 tokens fuse the gate and up projections with SwiGLU. An
expert-aligned 16-row schedule avoids recomputing tiles that cross expert
boundaries, while separate gate/up threadgroup tiles reduce synchronization.
The down projection uses the same expert-aligned schedule; route weighting and
top-k reduction retain their native MLX order. The guarded kernels support the
measured M4 Max (`applegpu_g16s`) and M5 Max (`applegpu_g17s`) architectures.
Set `MERERUN_LAGUNA_FUSED_SORTED_NVFP4_MOE=0` to restore the portable sorted
gate/up path, `MERERUN_LAGUNA_FUSED_SORTED_NVFP4_DOWN=0` to restore the native
sorted down projection, or `MERERUN_LAGUNA_FAST_SORTED_INVERSE=0` to restore
the second route sort instead of the linear permutation inversion. Unsupported
hardware, dtypes, quantization, or shapes fall back automatically.

The ranked 512-token XS prefill shape also stages each sorted route row and
constructs the sorted expert IDs plus inverse permutation with fixed-shape
Metal copies. `MERERUN_LAGUNA_RANKED_PREFILL_ROUTE_STAGING=0` restores the
native gathers and second sort.

Prompt prefill also reuses one full-attention mask and one sliding-window mask
across the 40 layers and evaluates the graph in an eight-layer asynchronous
ladder. `MERERUN_LAGUNA_SHARED_ATTENTION_MASKS=0` restores per-layer masks;
set `MERERUN_LAGUNA_PREFILL_ASYNC_LADDER=0` to disable the ladder or choose a
stride from 1 through 40. Exact fused residual/RMSNorm and QK-norm/RoPE kernels
are enabled by default on M5 Max and remain opt-in elsewhere. Their independent
controls are `MERERUN_LAGUNA_PREFILL_FUSED_RESIDUAL_RMSNORM` and
`MERERUN_LAGUNA_PREFILL_QK_NORM_ROPE`. The final prompt row is sliced before
the output norm and language-model head; decode math and cache layout are
unchanged.

Single-token target decode fuses the NVFP4 gate and up gather-GEMVs with
SwiGLU, then keeps the native down projection. The fused path is enabled by
default and can be disabled with `MERERUN_LAGUNA_FUSED_NVFP4_MOE=0` for a
controlled A/B or rollback. It applies only on Apple GPU execution with
BF16/FP16 activations, NVFP4 group size 16, 4-bit weights, an input width
divisible by 512, and an output width divisible by 8. Every other layout falls
back to MLX's portable gather path. Prefill and multi-token verification keep
the measured expert-sorted implementation above.

On M5 Max, the exact Laguna XS single-token layout additionally combines the
eight routed down projections, shared-expert down projection, ordered BF16
route reduction, fixed 2.5 scale, and decoder residual add in one dispatch.
This is the one-output-row-per-SIMD layout validated in the paired MLX Fast M5
run. Set `MERERUN_LAGUNA_FUSED_ROUTED_SHARED_DOWN_RESIDUAL=0` to restore the
native projection/reduction chain; set it to `1` to opt in on a supported M4
Max. Any non-XS shape, dtype, quantization layout, or unquantized shared expert
falls back automatically.

The same M5-ranked runtime uses a retained group-32 affine INT8 side layout for
Q/K/V on the first 28 layers of one-token decode. Prefill, batched decode, and
the original BF16 checkpoint parameters are unchanged. Disable it with
`MERERUN_LAGUNA_NATIVE_AFFINE_QKV=0`, opt in on other hardware with `=1`, or
set `MERERUN_LAGUNA_NATIVE_AFFINE_QKV_LAYERS=0...40` for a bounded rollout.
The official XS head pattern makes the default 28-layer side layout 598.5 MiB;
preparation is idempotent and decode retains no additional buffer per token.
The default remains 28 because that exact slice passed the public, hidden,
semantic, and paired-timing M5 gates; the prepared 40-layer continuation is
not a production default until it receives the same validation.

Example:

```bash
mere.run model benchmark chat \
  --laguna-dflash-tokens 12 \
  --laguna-dflash-min-tokens 32 \
  --json
```

Tests cover typed config validation, official tensor inventory, quantized
shape contracts, cached-decode parity, chunked-prefill parity across the
sliding-window boundary, ragged target and draft parity, lossless greedy
verification, tokenizer/template behavior, and tool parsing. When changing
model math or loading, run the focused Laguna tests, `./scripts/check.sh`, and
a real-checkpoint benchmark before claiming runtime compatibility.
