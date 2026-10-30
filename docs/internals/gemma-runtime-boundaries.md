# Gemma runtime boundaries

Use this map when you change Gemma model computation or generation.
`MereRunGemmaModel` builds without Core, installed-model resolution,
tokenizers, HTTP, or CLI parsing. Core re-exports its public types for existing
callers.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `MereRunGemmaModel` | Typed configuration, text and vision layers, expert routing, attention caches, quantization, fused kernels, and MTP draft computation |
| `MereRunTensor` | Safetensors loading and shared quantized projection fusion |
| `MereRunDecode` | Sampling primitives used by assistant drafting and generation |
| `MereRunCore/Gemma4` | Installed resources, checkpoint mapping, templates, image preprocessing, generation policies, scheduling, target verification, and LoRA orchestration |

The dependency check enforces the model and isolated-test boundaries. Runtime
protocols and methods use package access. Public configuration and model types
stay available through Core imports. Other Core families, such as Laguna, reach
the Gemma cache protocol and implementations through the model library.

## Reading order

`Gemma4Generator.swift` owns actor state, request entry points, unload, and
statistics. Its extensions keep actor isolation:

1. `+Loading` resolves resources, loads target and assistant weights, and
   applies adapters. The text and unified loaders also serve training.
2. `+Generation` prepares the request and assembles the response. `+Policies`
   owns quantization transitions, token bans, and response cleanup.
3. `+Prefill` evaluates text and image prompts. `+PrefixCache` selects and
   stores reusable prefixes.
4. `+Decode` selects and runs serial, pipelined, or batched decoding.
5. `+Speculation` verifies assistant proposals. `+PromptLookup` verifies
   proposals from repeated prompt sequences.
6. `+Batching` queues, cancels, combines, and finishes rows.

`Gemma4GeneratorState.swift` holds the records these stages share.

## Contracts to preserve

### Cache state

Sliding-cache decode can use storage order because single-token attention
doesn't depend on that order. Multi-token attention needs chronological state
and enough preceding context for the earliest query in the chunk. Each forward
call keeps its starting offsets and the producer's attention context for shared
layers, even after resident storage advances. Specialized single-token decode
keeps its direct quantized-attention path. Cache forks, batching, and
quantization preserve valid rows, total offsets, and row identity.

### Numerics

Keep proportional RoPE frequencies, shared KV routing, per-layer inputs,
key-equals-value behavior, MoE routing, and final-logit softcapping. Fused and
compiled segments are invalidated when LoRA or requantization changes their
source modules or captured parameters. Optional fused kernels keep their
activation policy.

### Speculation

The assistant reads target KV state without mutating it. Target verification
decides every emitted token. On rejection, generation restores the retained
prefix and replays accepted tokens before it continues.

### Templates

Tokenizer templates stay in Core. The canonical-template overlay applies only
to an exact known-stale template checksum with a matching released model
profile; current and custom templates win otherwise. The E4B generation primer
stays separate from the 12B, 26B, and 31B template.

## Validate a change

```bash
swift build --target MereRunGemmaModel
swift build --target GemmaRuntimeTests
swift test --filter GemmaRuntimeTests
./scripts/check.sh
```

The isolated suite covers synthetic text and image forward paths, shared KV,
cache forks and batching, quantized storage, MTP draft isolation, and target
verification. Core tests cover production adapter replacement, tokenizer
behavior, and generation policies. GPU-specific kernel tests are opt-in.

Linux target compilation and CPU fixtures don't qualify published checkpoints,
Metal performance, or equivalence between sampled decode routes.
