# Qwen runtime boundaries

Use `MereRunQwenModel` when you need Qwen-family model layers without managed
model lookup, downloads, chat templates, or the CLI. Core re-exports the public
types for existing callers.

## Ownership

| Location | Responsibility |
| --- | --- |
| `MereRunQwenModel` | Typed configuration, dense and hybrid attention, expert routing, vision layers, MTP heads, draft history, QSA and PLE state, and compiled activations |
| `MereRunTensor` | Safetensors metadata and array loading, quantized projections, projection fusion, and shared tensor kernels |
| `MereRunKVCache` | Full-attention cache storage, forks, batching, and affine quantization |
| `MereRunCore/Q35` | Resource selection, checkpoint mapping, tokenizers, prompts, request scheduling, prefill, decode, target verification, and output |

The model target depends on the tensor, cache, and text-encoder libraries and
MLX. The package boundary check rejects a dependency on Core, downloads,
tokenizers, HTTP, or command parsing. `QwenRuntimeTests` has the same boundary.
Runtime hooks and constructors use package access. `Q35SwitchLinear` and its
call method are open because Inkling's LoRA adapter subclasses that projection
across the module boundary.

## Generator reading order

`Q35Generator.swift` owns actor state, request entry points, stream leases,
unloading, and statistics. Its extensions retain the same actor isolation:

1. `Q35Generator+Loading.swift` resolves resources and constructs loaded models.
2. `+TextWeights`, `+MTPWeights`, and `+WeightMapping` load and map checkpoints.
3. `+Request` prepares prompts and assembles responses. `+Vision` builds image
   embeddings and multimodal positions. `+Policy` selects resource bounds.
4. `+Prefill` evaluates prompt chunks and manages prefix snapshots.
5. `+Decode` selects and runs decode routes. `+Speculation` verifies MTP
   proposals and repairs rejected suffixes. `+Batching` schedules active rows.
6. `+Benchmark` measures target verification. `+RuntimeTypes` contains the
   internal records shared by those stages.

## State and arithmetic contracts

Preserve each request's stream lease and compiled-graph owner across suspension.
Synchronize a completed stream before returning it to the pool. A reused stream
does not reuse a previous request's compiled graph owner.

A prefix snapshot forks target caches and draft history together. It retains
pending transitions and the last hidden state required by the next draft.
Speculative evaluation uses disposable forks. Rollback restores the accepted
prefix of recurrent, convolution, QSA, and PLE state before decode resumes.
Target verification remains the authority for every emitted token.

Checkpoint mapping keeps direct RMSNorm scales and zero-centered offsets
distinct. Prepare and evaluate expert fusion one layer at a time before
releasing its source arrays. Vision resource selection stays in Core; model
installation retains quantization and bounded evaluation of mapped arrays.

## Validation

```bash
swift build --target MereRunQwenModel
swift build --target QwenRuntimeTests
swift test --filter QwenRuntimeTests
./scripts/check.sh
```

The isolated suite covers compiled-stream ownership, synthetic dense and hybrid
prefill, prefix forks, verification rollback, draft-history forks, PLE disk
metadata, and sparse attention. Existing Core tests cover generator policies,
checkpoint mappings, chunked prefill, prefix admission, and request behavior.
GPU-specific arithmetic tests remain opt-in. CPU fixtures and Linux compilation
do not qualify published checkpoints, Metal performance, or sampled-output
equivalence across different decode routes.
