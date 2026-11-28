# ACE-Step VAE

Oobleck VAE implementation used by the ACE-Step audio pipeline.

- `OobleckVAEConfig.swift`: typed decoder configuration.
- `OobleckVAE.swift` and `OobleckDecoder.swift`: model entry points.
- `OobleckResBlock.swift` and `WNConv1d.swift`: reusable layers.

Keep checkpoint layout expectations in resource/loading code, not in layer
implementations.
