# MereRunCore

Shared model resolution, manifests, generation protocols, and native runtime
families.

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
- `MereRunModel*.swift`: manifests, paths, and validation.
- Runtime family directories own model-specific loading, inference, and decode
  paths.

Keep external config and tokenizer data typed at the boundary. Do not let raw
dynamic JSON move deeper than the compatibility shim that ingests it.
