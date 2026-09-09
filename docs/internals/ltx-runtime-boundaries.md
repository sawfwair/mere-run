# LTX runtime boundaries

Use this guide when editing LTX inference. `MereRunLTXModel` builds with MLX,
MLXFast, and MLXNN. It does not depend on Core, tokenizers, installed models,
media codecs, or command parsing.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| `MereRunLTXModel` | Transformer layers, convolutional and diffusion VAEs, upsampling, duration prediction, position layouts, tiled decode, sampler math, and model caches |
| `MereRunCore/LTX` | Checkpoint selection and loading, weight mapping, prompt preparation, input media, generation actors, denoising orchestration, vocoding, and output |
| `MereRunCLI` | Command parsing, admission, API adapters, and diagnostics |
| `LTXRuntimeTests` | Independent tensor, temporal-layout, cached-attention, sampler, and tiled-decoder fixtures |
| `MereRunCoreTests` | Checkpoint adapters, resource layouts, generation contracts, and media integration |

Core re-exports public LTX model types. The duration-head and DiffVAE loading
methods remain Core extensions, so existing Core callers keep the same API.
Package access supports generation and adapter code without exposing private
model machinery as public API.

## Reading order

1. Read `LTXUnifiedAVGenerationTypes.swift` and `LTXUnifiedAVGenerator.swift`
   for inputs, outputs, and resident state.
2. Read the generator's loading, full-loading, and component-preparation extensions
   for checkpoint selection and adapter installation.
3. Read its generation, audio-generation, and audio-to-video extensions for
   execution order and resource lifetime.
4. Read the denoising files for the selected sampler and guidance path.
5. Read the model library for transformer, VAE, and cache computation.
6. Read `LTXVideoMP4Writer.swift` for final media assembly.

The legacy distilled generator has its own state, loading, and generation
files. Core also retains specialized HDR, DFR, retake, and reference-input policy.

## Preserved execution contracts

The extraction preserves tensor arithmetic, checkpoint keys, compiled-block
variants, random-stream order, evaluation boundaries, and actor isolation.
The original cleanup scopes retain ownership of adapter activation and resident
state. Decoder tiling retains causal temporal overlap and weighted accumulation.
Audio and video use their existing distinct sampling grids and shared time origin.

The boundary does not establish checkpoint quality, cancellation responsiveness,
or lower memory use. Those require execution measurements. Specialized generation
paths remain in Core and can be refined independently of model compilation.

## Validation

```bash
swift build --target MereRunLTXModel
swift build --target LTXRuntimeTests
swift test --filter LTXRuntimeTests
bash scripts/check-model-boundaries.sh
./scripts/check.sh
```

The boundary guard rejects transitive Core, tokenizer, codec, and CLI dependencies
in the model and its tests. It also rejects MLX test support in shipped products.
The model tests cover causal convolution, cached and masked attention, DiffVAE
query tiling, frame geometry, weighted decode overlap, duration prediction,
sampler updates, and synchronized guidance-cache reuse.

Published-checkpoint tests remain opt-in. Metal-specific attention tests require
a GPU; a CPU fixture pass does not qualify those kernels or CUDA execution.
