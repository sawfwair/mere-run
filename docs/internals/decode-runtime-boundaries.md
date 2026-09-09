# Shared decode boundaries

If you maintain an autoregressive model runtime, use `MereRunDecode` for shared
sampling and token-loop behavior. The library builds with MLX and MLXRandom,
without Core, tokenizers, model catalogs, checkpoint loaders, or application
services. Core re-exports its public API to preserve existing imports.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `MereRunDecode` | Sampling configuration, penalties, token selection, pipelined decode loops, incremental text emission, and logprob diagnostics |
| `MereRunKVCache` | Attention cache protocol, full-precision caches, and optional affine quantized caches |
| Family runtime | Model loading, prompt tokenization, forward callbacks, cache selection, request seeds, MLX streams, and cleanup |
| Core API adapters | Chat requests, tool policies, and OpenAI-compatible response serialization |

`SamplingLogprobs.swift` separates diagnostic host reads from token selection.
The diagnostic types retain their Codable fields and `final_target` source marker. A model's
forward callback supplies logits; the shared loop does not depend on any model
architecture or cache implementation.

## Preserve callback and scheduling behavior

`decode` confirms the first token before opening the deeper pipeline. Later
iterations schedule a dependent forward before confirming the preceding token.
The final budget boundary avoids an unused forward. EOS or a stop callback can
discard work already queued.

`decode` appends a confirmed non-EOS token before calling `shouldContinue`.
`decodeStateful` calls `didSampleToken`, then `shouldContinue`, then checks EOS
before appending. Its forward callback runs one step ahead even for the final
token. Preserve these distinct contracts when adapting a runtime.

Cancellation and forward failures propagate from `decode`. The family runtime
owns synchronization and cleanup of any pending work, model, cache, and lease.
The shared loop never releases those resources.

Penalty history belongs to each decode invocation. Preserve prompt exclusions,
filtering order, token bans, and top-p policy overrides. Logprob capture uses
exact policy selection and redacts reasoning text. It remains opt-in and is
supported by `decode`; the stateful loop retains token-only results.

## Preserve cache behavior

Affine quantization remains an explicit runtime choice. Moving the cache into
`MereRunKVCache` does not change default cache selection. Packed state preserves
source dtype, allocation headroom, offsets, fork isolation, and row splitting.

Before Metal dequantization, packed weights, scales, and biases must be
contiguous. Same-shape reshaping provides independent wrappers for forks;
a same-dtype cast can return the original wrapper and break mutation isolation.

## Validate a change

Build the narrow targets before running the full suite:

```bash
swift build --target MereRunDecode
swift build --target DecodeRuntimeTests
swift test --filter DecodeRuntimeTests
./scripts/check.sh
```

The package guard checks transitive dependencies on Apple and Linux manifests,
including the CUDA prebuilt configuration. It also prevents shipped products
from depending on MLX test support.

Fixtures cover sampling, penalties, streaming, stop callbacks, cancellation,
forward failures, diagnostics, and affine cache lifecycle. Real-checkpoint
output, GPU timing, and opt-in long BF16 cache layouts require separate
qualification. Local fixtures do not establish those results.
