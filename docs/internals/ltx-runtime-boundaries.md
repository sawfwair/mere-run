# LTX runtime boundaries

Use this map when you edit LTX inference. `MereRunLTXModel` builds with MLX,
MLXFast, and MLXNN only. It doesn't depend on Core, tokenizers, installed
models, media codecs, or command parsing. Core re-exports its public types.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `MereRunLTXModel` | Transformer layers, convolutional and diffusion VAEs, upsampling, duration prediction, position layouts, tiled decode, sampler math, and model caches |
| `MereRunCore/LTX` | Checkpoint selection and loading, weight mapping, prompt preparation, input media, generation actors, denoising orchestration, vocoding, and output |
| `MereRunCLI` | Command parsing, admission, API adapters, and diagnostics |
| `LTXRuntimeTests` | Independent tensor, temporal-layout, cached-attention, sampler, and tiled-decoder fixtures |
| `MereRunCoreTests` | Checkpoint adapters, resource layouts, generation contracts, and media integration |

The duration-head and DiffVAE loading methods stay as Core extensions, so
existing Core callers keep the same API. Package access lets generation and
adapter code reach model internals without making them public API.

## Reading order

All files are in `Sources/MereRunCore/LTX` unless noted.

1. `LTXUnifiedAVGenerationTypes.swift` and `LTXUnifiedAVGenerator.swift`
   define inputs, outputs, and resident state.
2. The generator's `+Loading`, `+FullLoading`, and `+ComponentPreparation`
   extensions handle checkpoint selection and adapter installation.
3. Its `+Generation`, `+AudioGeneration`, and `+AudioToVideo` extensions set
   execution order and resource lifetime.
4. The denoising files implement the selected sampler and guidance path.
5. `Sources/MereRunLTXModel` holds transformer, VAE, and cache computation.
6. `LTXVideoMP4Writer.swift` assembles the final media.

`LTXDistilledLatentGenerator` has its own state, loading, and generation files.
Core also keeps the specialized HDR, DFR, retake, and reference-input policy.

## Contracts to preserve

Preserve tensor arithmetic, checkpoint keys, compiled-block variants,
random-stream order, evaluation boundaries, and actor isolation. The cleanup
scopes own adapter activation and resident state. Decoder tiling keeps causal
temporal overlap and weighted accumulation. Audio and video keep their distinct
sampling grids and shared time origin.

This boundary doesn't establish checkpoint quality, cancellation
responsiveness, or lower memory use; those need execution measurements.
Specialized generation paths stay in Core and can change without recompiling
the model library.

## Validate a change

```bash
swift build --target MereRunLTXModel
swift build --target LTXRuntimeTests
swift test --filter LTXRuntimeTests
bash scripts/check-model-boundaries.sh
./scripts/check.sh
```

The boundary guard rejects transitive Core, tokenizer, codec, and CLI
dependencies in the model and its tests. It also rejects MLX test support in
shipped products. The model tests cover causal convolution, cached and masked
attention, DiffVAE query tiling, frame geometry, weighted decode overlap,
duration prediction, sampler updates, and synchronized guidance-cache reuse.

Published-checkpoint tests are opt-in. Metal-specific attention tests need a
GPU; a CPU fixture pass doesn't qualify those kernels or CUDA execution.
