# H3 and Laguna runtime boundaries

Use this map to place an H3 or Laguna change in the owning library and choose
its validation. The model targets separate tensor computation from request and
resource policy. They have no CLI, tokenizer, media, audio codec, or Core
dependency. Core re-exports their public types for existing callers.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `MereRunH3Model` | H3 transformer layers, VAEs, packed geometry, numerical schedules, in-memory AdaLN caches, low-rank projection wrappers, and fused kernels |
| `MereRunLagunaModel` | Laguna layers, typed configuration, ragged caches, DFlash context and verification, and guarded model acceleration |
| `MereRunAudioModels` | BigVGAN computation shared by H3 and MMAudio |
| `MereRunTensor` | Shared tensor loading and routed quantization primitives |
| `MereRunGemmaModel` | Attention-cache implementations reused by Laguna |
| `MereRunDecode` | Sampling, incremental text decoding, and shared decode contracts |
| `MereRunCore` | Installed resources, checkpoint mapping, prompts, media preparation, request defaults, adapter installation, batching, and cleanup |

Package access connects model hooks to Core without adding a public runtime
API. The dependency checker validates the targets' transitive dependencies in
the macOS and Linux gates.

Laguna reuses the Gemma attention-cache implementation, so its model boundary
includes `MereRunGemmaModel` instead of duplicating those caches. Routed
quantization kernels live in `MereRunTensor`, where other runtimes share them.

## Reading order

### H3

1. `Sources/MereRunCore/MiniMaxH3/MiniMaxH3Generator.swift` holds retained
   runtime state.
2. Its extensions cover loading, denoising, schedule application, generation,
   reference generation and preparation, and conditioning.
   `MiniMaxH3SlidingWindow.swift` keeps its own orchestration.
3. In `Sources/MereRunH3Model`, follow `MiniMaxH3Transformer.swift` and its
   forward, block, and compilation extensions. Projection dispatch and the
   fused Metal kernels have separate files.

Generation modes and execution policy select acceleration, memory behavior, and
schedule defaults; the model library implements those selections. It doesn't
choose installed checkpoints, resolve adapter recipes, decode media, or export
results. AdaLN persistence and adapter cache augmentation stay as Core adapters
around the model-owned in-memory tensors.

### Laguna

1. `Sources/MereRunCore/Laguna/LagunaGenerator.swift` holds the actor's public
   entry points and retained state.
2. Its extensions own loading, generation, prefill, ordinary continuous
   batching, and DFlash continuous batching.
3. `Sources/MereRunLagunaModel` owns the target and draft models.

DFlash verification receives explicit callbacks for token decoding, output, and
cancellation. Core keeps tokenizer loading, prompt rendering, request admission
to its batching loops, adapter switching, and model release.

## Contracts to preserve

For H3, preserve module keys, global QKV slab ordering, evaluation boundaries,
compiled-runner invalidation, and modality-specific cache rules.

For both runtimes, keep the platform guards around Metal kernels. For Laguna's
guarded acceleration paths and their rollback controls, see the
[guarded acceleration audit](./guarded-acceleration.md).

## Validate a change

Run the focused fixtures while you edit:

```bash
swift test --filter 'H3RuntimeTests|LagunaRuntimeTests'
```

The isolated targets cover packed geometry, attention and cache behavior,
projection equivalence, speculative verification, batch isolation, and
cancellation before decode. Resource loading, adapter installation, training,
and tokenizer integration stay in `MereRunCoreTests`.

Before you open a pull request, run the repository gate:

```bash
./scripts/check.sh
```

A CPU fixture, target build, or package-manifest check doesn't establish
Metal or CUDA numerical parity, published-checkpoint quality, inference
performance, or full application acceptance. Report that evidence separately.
