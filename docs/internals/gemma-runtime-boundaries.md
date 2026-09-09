# Gemma runtime boundaries

Use this map when you change Gemma model computation or generation.
`MereRunGemmaModel` builds without Core, installed-model resolution, tokenizers,
HTTP, or CLI parsing. Core re-exports its public types for existing callers.

## Ownership

| Location | Responsibility |
| --- | --- |
| `MereRunGemmaModel` | Typed configuration, text and vision layers, expert routing, attention caches, quantization, fused kernels, and MTP draft computation |
| `MereRunTensor` | Safetensors loading and shared quantized projection fusion |
| `MereRunDecode` | Sampling primitives used by assistant drafting and generation |
| `MereRunCore/Gemma4` | Installed resources, checkpoint mapping, templates, image preprocessing, generation policies, scheduling, target verification, and LoRA orchestration |

The dependency check enforces the model and isolated-test boundaries. Runtime
protocols and methods use package access. Public configuration and model types
remain available through Core imports. Other Core families retain access to
the Gemma cache protocol and implementations through the model library.

## Generator reading order

`Gemma4Generator.swift` owns actor state, request entry points, unload, and
statistics. Its extensions retain actor isolation:

1. `+Loading` resolves resources, loads target and assistant weights, and applies
   adapters. Text and unified loaders also serve training.
2. `+Generation` prepares the request and assembles the response. `+Policies`
   owns quantization transitions, token bans, and response cleanup.
3. `+Prefill` evaluates text and image prompts. `+PrefixCache` selects and stores
   reusable prefixes.
4. `+Decode` selects and runs serial, pipelined, or batched decoding.
5. `+Speculation` verifies assistant proposals. `+PromptLookup` verifies proposals
   from repeated prompt sequences.
6. `+Batching` queues, cancels, combines, and finishes rows.
   `Gemma4GeneratorState.swift` holds the records shared by these stages.

## State and numerical contracts

Sliding-cache decode may use storage order because single-token attention is
invariant to that order. Multi-token attention requires chronological state
and enough preceding context for the earliest query in the chunk. Each forward
call retains its starting offsets and producer attention context for shared
layers, even after resident storage advances. Specialized single-token decode
keeps its direct quantized-attention path. Cache forks,
batching, and quantization preserve valid rows, total offsets, and row identity.

Retain proportional RoPE frequencies, shared KV routing, per-layer inputs,
key-equals-value behavior, MoE routing, and final-logit softcapping. Fusion and
compiled segments invalidate when LoRA or requantization changes source modules
or captured parameters. Optional fused kernels retain their activation policy.

The assistant reads target KV state without mutating it. Target verification
remains authoritative for emitted tokens. On rejection, generation restores
the retained prefix and replays accepted tokens before continuing.

Tokenizer templates remain in Core. Only an exact known-stale template checksum
and matching released model profile permit the canonical-template overlay.
Current and custom templates remain authoritative. The E4B generation primer
remains separate from the 12B, 26B, and 31B template.

## Validation

```bash
swift build --target MereRunGemmaModel
swift build --target GemmaRuntimeTests
swift test --filter GemmaRuntimeTests
./scripts/check.sh
```

The isolated suite covers synthetic text and image forward paths, shared KV,
cache forks and batching, quantized storage, MTP draft isolation, and target
verification. Core tests cover production adapter replacement, tokenizer
behavior, and generation policies. GPU-specific kernel tests remain opt-in.
Linux target compilation and CPU fixtures do not qualify published checkpoints,
Metal performance, or equivalence between different sampled decode routes.
