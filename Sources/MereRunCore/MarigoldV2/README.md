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

The transformer uses MLX affine 4-bit quantization before the adapters are
installed. The reference uses bitsandbytes NF4 with selected layers excluded
from quantization or dequantized. The native VAE encoder also uses the posterior
mode; the reference samples the posterior. These differences require separate
checkpoint accuracy validation. A build or unit-test pass does not establish
reference parity.

The runtime loads the BF16 transformer before quantization. The catalog's
64 GB minimum and 96 GB recommendation reserve loading headroom; they are
provisional, not measured inference requirements. Output is affine-invariant:
depth is recovered up to an unknown scale and shift, so no camera or point
cloud is written.

Keep the fixed timestep, the velocity step, and the channel readout covered by tests
before changing inference defaults.
