# MarigoldV2

Single-step monocular depth on a frozen Qwen-Image-Edit-2509 transformer.

- `MarigoldV2Resources.swift`: repository identity, checkpoint variants, install
  layout, artifact pins, and validation.
- `MarigoldV2ModelConfigs.swift`: the narrow base config this runtime needs.
- `MarigoldV2LoRAAdapter.swift`: rank-128 adapter installation onto `MMDiT`.
- `MarigoldV2VAEDecoder.swift`: the fine-tuned decoder some checkpoints ship.
- `MarigoldV2PromptConditioning.swift`: precomputed prompt embeddings, which
  stand in for the base text encoder.
- `MarigoldV2Generator.swift`: encode, one rectified-flow step, decode, read out.
- `MarigoldV2DepthNormalization.swift` and `MarigoldV2DepthExport.swift`:
  affine-relative normalization and the EXR, preview, and manifest artifacts.

The transformer is quantized to 4 bits before the adapters are installed, matching
how the checkpoints were trained. Output is affine-invariant: depth is recovered up
to an unknown scale and shift, so no camera or point cloud is written.

Keep the fixed timestep, the velocity step, and the channel readout covered by tests
before changing inference defaults.
