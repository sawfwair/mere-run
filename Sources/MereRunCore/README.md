# MereRunCore

Shared runtime validation, catalog assembly, generation protocols, and native
runtime families. Model metadata and installed lookup live in `MereRunModelKit`;
Core re-exports those types for existing callers.
Shared full-attention caches live in `MereRunKVCache`. Tensor loading, Qwen
encoder layers, and FLUX.2/ZImage models live in `MereRunTensor`,
`MereRunTextEncoder`, and `MereRunImageModels`. Core re-exports these libraries and `MereRunDecode`, which owns shared token
sampling, decoding, streaming, and logprob diagnostics. Native Qwen model layers
live in `MereRunQwenModel`. Gemma layers, caches, and draft computation live in
`MereRunGemmaModel`. Core retains resource loading and generation for both families.

- `Generation.swift`: image and chat request/response contracts.
- `ImageGenerationOptions.swift` and `ImageGenerationPlan.swift`: typed image
  inputs, model selection, effective sampling, conditioning, and validation.
- `ImageGenerationOperation.swift`: image preparation, executor invocation,
  cleanup, typed events, and results. The caller's executor owns admission and
  runtime residency.
- `ImageMaskEditing.swift`: mask/outpaint preparation and pixel restoration.
- `ManagedAdapterArgumentResolver.swift`: installed adapter lookup and base-model
  compatibility shared by image, text, video, and evaluation adapters.
- `ManagedModel*.swift`: public managed-model catalog and install metadata.
- `ModelResolver.swift`: adapts the catalog and family validators to ModelKit lookup.
- `MereRunModelManifest+Templates.swift`: runtime-dependent manifest templates.
- Runtime family directories own model-specific loading, inference, and decode
  paths.

Keep external config and tokenizer data typed at the boundary. Do not let raw
dynamic JSON move deeper than the compatibility shim that ingests it.
