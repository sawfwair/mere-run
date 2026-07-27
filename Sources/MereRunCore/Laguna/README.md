# Laguna S 2.1

Native Swift/MLX evaluation support for Poolside's official
`Laguna-S-2.1-NVFP4-mlx` checkpoint. This module is intentionally isolated
from the managed model catalog while checkpoint compatibility and real-device
quality are being evaluated.

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

## Evaluation boundary

The CLI benchmark commands accept `--laguna-path` as an explicit local-only
checkpoint override. Pass `--laguna-dflash-path` to evaluate the official
companion checkpoint and `--laguna-dflash-tokens` to control the proposal
length. The measured default is 12 proposals per round.
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

The checkpoint is not a catalog model, is never selected by default, and is
not part of public pull or serving behavior. Keep it behind this boundary until
official weights have passed load, deterministic decode, quality, memory, and
performance evaluation on Apple Silicon.

Laguna's routed experts sort prefill and multi-token verification routes by
expert before issuing the NVFP4 gather matmuls. This is enabled by default for
64 or more routes; single-token target decode keeps the lower-overhead unsorted
path. Set `MERERUN_LAGUNA_SORTED_MOE=0` to restore the reference routing order
for comparison or rollback. Do not lower the threshold without measuring
short verification and concurrent decode: two route sorts can cost more than
grouped expert locality saves on small batches.

Example:

```bash
mere.run model benchmark chat \
  --laguna-path /path/to/Laguna-S-2.1-NVFP4-mlx \
  --laguna-dflash-path /path/to/Laguna-S-2.1-DFlash \
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
