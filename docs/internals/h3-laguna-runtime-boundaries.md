# H3 and Laguna runtime boundaries

This guide helps runtime contributors place H3 and Laguna changes in the owning
library and choose the relevant validation.

## Ownership

The model targets separate tensor computation from request and resource policy.

| Owner | Responsibility |
| --- | --- |
| `MereRunH3Model` | H3 transformer layers, VAEs, packed geometry, numerical schedules, in-memory AdaLN caches, low-rank projection wrappers, and fused kernels |
| `MereRunLagunaModel` | Laguna layers, typed configuration, ragged caches, DFlash context and verification, and guarded model acceleration |
| `MereRunAudioModels` | BigVGAN computation shared by H3 and MMAudio |
| `MereRunTensor` | Shared tensor loading and routed quantization primitives |
| `MereRunGemmaModel` | Attention-cache implementations reused by Laguna |
| `MereRunDecode` | Sampling, incremental text decoding, and shared decode contracts |
| `MereRunCore` | Installed resources, checkpoint mapping, prompts, media preparation, request defaults, adapter installation, batching, and cleanup |

Core re-exports public model types for existing callers. Package access connects
model hooks to Core without adding a public runtime API. Model targets have no
CLI, tokenizer, media, audio codec, or Core dependency. The dependency checker
validates their transitive dependencies in the macOS and Linux gates.

## Follow an H3 request

Start at `Sources/MereRunCore/MiniMaxH3/MiniMaxH3Generator.swift` for retained
runtime state. Follow the generator extensions for loading, denoising, schedule
application, FL2VA generation, Ref2VA generation, reference preparation, and
conditioning. Sliding-window generation retains its own orchestration file.

Generation modes and execution policy select acceleration, memory behavior, and
schedule defaults. The model library implements those selections. It does not
choose installed checkpoints, resolve adapter recipes, decode media, or export
results. AdaLN persistence and adapter cache augmentation remain Core adapters
around model-owned in-memory tensors.

In `Sources/MereRunH3Model`, follow the transformer entry, forward methods, block
execution, and compilation extensions. Projection dispatch and Metal kernels
have separate files. Preserve module keys, global QKV slab ordering, evaluation
boundaries, compiled-runner invalidation, and modality-specific cache rules when
you change these paths.

## Follow a Laguna request

Start at `Sources/MereRunCore/Laguna/LagunaGenerator.swift` for the actor's public
entry points and retained state. Its extensions own loading, generation,
prefill, ordinary continuous batching, and DFlash continuous batching.

`Sources/MereRunLagunaModel` owns the target and draft models. DFlash verification
receives explicit token decoding, output, and cancellation callbacks. Core
retains tokenizer loading, prompt rendering, request admission to its batching
loops, adapter switching, and model release.

Laguna reuses the Gemma attention-cache implementation. Its model boundary
therefore includes `MereRunGemmaModel`; it does not duplicate those caches to
create an artificial dependency separation. Routed quantization kernels live in
`MereRunTensor`, where their other runtime callers use the same implementation.

## Validate a change

Run focused fixtures while editing:

```bash
swift test --filter 'H3RuntimeTests|LagunaRuntimeTests'
```

The isolated targets cover packed geometry, attention and cache behavior,
projection equivalence, speculative verification, batch isolation, and
cancellation before decode. Resource loading, adapter installation, training,
and tokenizer integration remain in `MereRunCoreTests`.

Before opening a pull request, run the repository gate:

```bash
./scripts/check.sh
```

Preserve platform guards around Metal kernels. A CPU fixture, target build, or
package-manifest check does not establish Metal/CUDA numerical parity,
published-checkpoint quality, inference performance, or full application
acceptance. Report those evidence boundaries separately.
