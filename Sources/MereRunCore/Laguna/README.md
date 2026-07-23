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
  sharded safetensors, and owns streaming generation and inference metrics.
- `LagunaToolParser.swift` converts the checkpoint's GLM-style tool markup to
  mere.run's typed `ToolCall` output.

## Evaluation boundary

The CLI benchmark commands accept `--laguna-path` as an explicit local-only
checkpoint override. The checkpoint is not a catalog model, is never selected
by default, and is not part of public pull or serving behavior. Keep it behind
this boundary until official weights have passed load, deterministic decode,
quality, memory, and performance evaluation on Apple Silicon.

Tests cover typed config validation, official tensor inventory, quantized
shape contracts, cached-decode parity, tokenizer/template behavior, and tool
parsing. When changing model math or loading, run the focused Laguna tests,
`./scripts/check.sh`, and a real-checkpoint benchmark before claiming runtime
compatibility.
