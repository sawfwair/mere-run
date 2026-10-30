# Recursive Flint

Recursive Flint is a sparse mixture-of-experts chat model with tool calling and
visible reasoning. `mere.run` runs it natively in Swift/MLX from either the
published MLX INT4 export or the original BF16 safetensors, with no conversion
step. Use INT4 on a 48 GB or larger Mac. Use BF16 when you have 128 GB and want
the original weights.

## Install

Both checkpoints are explicit installs; inference commands don't download them
automatically.

| Model ID | Weights | Download | Unified memory (minimum / recommended) |
| --- | --- | --- | --- |
| `text-chat-flint-4bit` | Published INT4 export | About 22.6 GB | 48 GB / 64 GB |
| `text-chat-flint-bf16` | Original BF16 | About 79.6 GB | 128 GB / 128 GB |

The memory values are catalog admission estimates, not measurements of this
runtime's performance on the released checkpoints.

```bash
mere.run model pull text-chat-flint-4bit
```

To use a snapshot you already have, pass its directory to `--model`, or pass
the matching managed model ID with `--model-root`. The directory's
`config.json` must set `model_type` to `flint`. NVFP4 and GGUF checkpoints use
different tensor formats, and this runtime rejects them.

## Chat

The following command sets sampling and the output budget explicitly:

```bash
mere.run text chat --model text-chat-flint-4bit \
    --prompt "Write a Swift function that merges overlapping intervals." \
    --temperature 1 --top-p 0.95 --top-k 20 --max-tokens 4096
```

## Serve the local API

After installation, start the native serving engine:

```bash
mere.run api serve --engine text-chat-flint --model text-chat-flint-4bit
```

The API accepts tools, stop sequences, seeds, penalties, and log probabilities.
A single response can return up to 32,768 tokens.

## Behavior

- **Context.** Chat defaults to an 8,192-token working context. Prompt and
  output share the configured limit, up to the checkpoint's 262,144-token
  maximum. An oversized prompt fails instead of truncating history.
- **Cache.** Sliding-window layers keep the last 512 tokens. Full-attention
  cache memory grows with the context.
- **Reasoning.** The model controls its own reasoning. Display options hide or
  show it but don't turn it off. Assistant history keeps reasoning between
  turns.
- **Tools.** Tool messages use the checkpoint's chat template. The runtime reads
  tool calls only outside reasoning blocks and applies the shared tool
  allowlist and parallel-call policy.

## Limits

This runtime doesn't implement grammar-constrained JSON, strict tool-schema
decoding, LoRA, speculative decoding, continuous batching, quantized KV caches,
or NVFP4 loading. Publisher performance parity and long-context quality are
outside the validation described on this page.

## Validation

The local tests compare all logits of a deterministic three-layer synthetic
model with an independent scalar reference. They cover dense and shared experts,
sigmoid routing with selection-only probability bias, learned attention sinks,
RoPE placement, and sliding-cache chunk boundaries. The fixture generator uses
only the Python standard library and is test tooling; inference uses Swift/MLX.

On an M5 Max with 128 GiB of unified memory, both released checkpoints pass
native Metal arithmetic, a typed `calculate` tool round trip, and retrieval of
an exact access code from a prompt longer than 1,024 tokens. The BF16 run uses
the original weights, and all fifteen shard hashes match the pinned publisher
metadata. The INT4 run uses the published weights and the production loader's
pinned artifact verification. Both runs use production precision defaults.

A five-token probe (`[1, 33, 204, 1024, 42]`) compares all 750,720 logits with
an independent PyTorch 2.10 CPU implementation, which was first checked against
the scalar fixture. For each checkpoint, native full-sequence prefill and chunk
sizes 1, 3, and 5 agree with that reference on all five next-token choices. The
logits differ by the following amounts:

| Checkpoint | Full-sequence maximum absolute difference | RMS difference |
| --- | --- | --- |
| Original BF16 | 3.6158 | 0.6936 |
| Published INT4 | 1.3944 | 0.3199 |

These results establish bounded greedy-choice agreement. They don't establish
numerical equivalence, broad model quality, or the declared context limit.

To repeat the installed functional regression, point it at either checkpoint:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_FLINT_CHECKPOINT_ROOT=/path/to/flint-checkpoint \
    swift test --filter FlintInstalledQualificationTests
```

Linux preparation includes the CUDA quantized-matrix synchronization and BF16
accumulation corrections. Their patching tests cover source drift, repeat
application, and SwiftPM's read-only checkout files. CUDA execution on the
refreshed dependency pin needs separate qualification.

## Sources and license

The managed checkpoints are pinned to these immutable revisions:

| Model ID | Source | Revision |
| --- | --- | --- |
| `text-chat-flint-4bit` | `RecursiveSI/flint-int4` | `3b93f9053b530488405ae9fca51d4900d6ce21f0` |
| `text-chat-flint-bf16` | `RecursiveSI/flint` | `2a247be0c0aec895ccb0ac7ca92229ec6df37288` |

Both checkpoints are published under Apache-2.0. Architecture references include
the BF16 checkpoint's SGLang implementation and
[Recursive's inference engine](https://github.com/recursive-oss/rec/tree/e53663b262af4cb1ca498e596ca8b1a2d79e0800).
The INT4 loader verifies the publisher's SHA-256 values before it loads the
packed tensors, and it keeps the export's INT8 embeddings and INT4 projections
without requantization. The BF16 loader checks shard ownership, complete tensor
coverage, shapes, and floating-point types before use.

For other text models, see the [text runtime](./text.md).
