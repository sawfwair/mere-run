# FLUX.1

Native FLUX.1 text-to-image inference for the gated `FLUX.1-dev` Diffusers
checkpoint.

- `Flux1Generator.swift`: component loading, prompt encoding, denoising, and decode.
- `Flux1Transformer.swift`: FLUX.1 dual-stream and single-stream transformer.
- `Flux1TextEncoders.swift`: CLIP-L pooled and T5-XXL sequence conditioning.
- `Flux1Configs.swift`: typed Diffusers configuration and checkpoint resources.
- `Flux1Scheduler.swift`: flow-match schedule and latent packing.

FLUX.1 and FLUX.2 adapters are architecture-specific. Keep this runtime and
its manifest family separate from `Flux2Klein` even though both use FLUX-style
block names.
