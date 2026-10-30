# Decode runtime boundaries

Use `MereRunDecode` for shared sampling and token-loop behavior when you
maintain an autoregressive model runtime. The library builds with MLX and
MLXRandom only. It has no dependency on Core, tokenizers, model catalogs,
checkpoint loaders, or application services. Core re-exports its public API, so
existing imports keep working.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `MereRunDecode` | Sampling configuration, penalties, token selection, pipelined decode loops, incremental text emission, and logprob diagnostics |
| `MereRunKVCache` | Attention cache protocol, full-precision caches, and optional affine quantized caches |
| Family runtime | Model loading, prompt tokenization, forward callbacks, cache selection, request seeds, MLX streams, and cleanup |
| Core API adapters | Chat requests, tool policies, and OpenAI-compatible response serialization |

A model's forward callback supplies logits. The shared loop doesn't depend on
any model architecture or cache implementation.

## Reading order

1. `AutoregressiveDecodeEngine.swift` defines `decode` and `decodeStateful`
   and their callbacks.
2. `SamplingLogprobs.swift` separates diagnostic host reads from token
   selection.
3. `ChatLogprobTypes.swift` holds the Codable diagnostic types, including the
   `final_target` source marker.

## Contracts to preserve

### Callbacks and scheduling

`decode` confirms the first token before it opens the deeper pipeline. Later
iterations schedule a dependent forward before they confirm the preceding
token. The final budget boundary skips the unused forward. EOS or a stop
callback can discard work that's already queued.

The two loops order their callbacks differently:

- `decode` appends a confirmed non-EOS token, then calls `shouldContinue`.
- `decodeStateful` calls `didSampleToken`, then `shouldContinue`, then checks
  EOS before it appends. Its forward callback runs one step ahead, even for the
  final token.

Keep these contracts distinct when you adapt a runtime.

Cancellation and forward failures propagate out of `decode`. The family runtime
owns synchronization and cleanup of pending work, the model, the cache, and any
lease. The shared loop never releases those resources.

### Sampling and logprobs

Penalty history belongs to each decode invocation. Preserve prompt exclusions,
filtering order, token bans, and top-p policy overrides. Logprob capture uses
exact policy selection and redacts reasoning text. It's opt-in and only `decode`
supports it; `decodeStateful` returns tokens only.

### Cache

Affine quantization is an explicit runtime choice; `MereRunKVCache` doesn't
change default cache selection. Packed state preserves source dtype, allocation
headroom, offsets, fork isolation, and row splitting.

Packed weights, scales, and biases must be contiguous before Metal
dequantization. A same-shape reshape gives each fork an independent wrapper. A
same-dtype cast can return the original wrapper and break mutation isolation.

## Validate a change

Build the narrow targets before you run the full suite:

```bash
swift build --target MereRunDecode
swift build --target DecodeRuntimeTests
swift test --filter DecodeRuntimeTests
./scripts/check.sh
```

The package guard checks transitive dependencies in the Apple and Linux
manifests, including the CUDA prebuilt configuration. It also keeps shipped
products from depending on MLX test support.

Fixtures cover sampling, penalties, streaming, stop callbacks, cancellation,
forward failures, diagnostics, and the affine cache lifecycle. They don't
qualify real-checkpoint output, GPU timing, or opt-in long BF16 cache layouts;
those need separate runs.
