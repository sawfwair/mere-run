# Laguna model computation

Use this library to change Laguna layers and speculative verification without
loading the CLI, tokenizer, model catalog, or training pipeline.

- Configuration types decode the target and DFlash checkpoint contracts.
- `LagunaCausalLM` and `LagunaLanguageModel` compose attention, dense layers,
  routed experts, and vocabulary projection.
- `LagunaAttention` separates acceleration preparation from forward execution.
  RoPE and ragged caches have their own files.
- `LagunaDFlashModel` owns draft layers and target-context projection.
  `LagunaDFlashDecoder` owns proposal verification, rejection correction, and
  decode callbacks; its state types describe routing and measured outcomes.
- Acceleration policy and fused kernels retain their hardware and shape guards.
  Shared routed quantization primitives live in `MereRunTensor`.

The target reuses attention-cache implementations from `MereRunGemmaModel` and
sampling and token-decoding primitives from `MereRunDecode`. Core retains
checkpoint loading, prompts, LoRA installation and training, request routing,
continuous batching, streaming adapters, and resource cleanup.

Run `LagunaRuntimeTests` for model, cache, quantization, DFlash, and cancellation
fixtures. Core tests retain installed-resource, tokenizer, training, and adapter
integration coverage. CPU fixtures do not qualify the Metal acceleration paths
or published checkpoint performance.
