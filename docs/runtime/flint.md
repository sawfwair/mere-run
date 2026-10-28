# Run Recursive Flint

This guide is for developers running Recursive Flint through the Swift/MLX
runtime in `mere.run`. The runtime loads the published MLX INT4 export and the
original BF16 safetensors directly. Model execution stays on-device.

## Install and chat

Install the quantized checkpoint, then run a text prompt:

```bash
mere.run model pull text-chat-flint-4bit
mere.run text chat --model text-chat-flint-4bit \
    --prompt "Write a Swift function that merges overlapping intervals." \
    --temperature 1 --top-p 0.95 --top-k 20 --max-tokens 4096
```

The INT4 download is approximately 22.6 GB. The catalog requires at least 48 GB
of unified memory and recommends 64 GB. These are admission estimates, not
measurements of this runtime's released-checkpoint performance. The BF16 model
ID is `text-chat-flint-bf16`; its weights occupy approximately 79.6 GB, and its
catalog memory floor is 128 GB.

To load an existing snapshot, pass its directory through `--model` or use
`--model-root` together with the matching managed model ID. The `config.json`
file must identify the architecture as `flint`. NVFP4 and GGUF checkpoints use
different tensor formats and are not accepted by this runtime.

## Serve the local API

After installation, start the native serving engine:

```bash
mere.run api serve --engine text-chat-flint --model text-chat-flint-4bit
```

The runtime preserves reasoning in assistant history and uses the checkpoint's
chat template for tool messages. It reads tool calls only outside reasoning
blocks and applies the shared tool allowlist and parallel-call policy. It does
not provide grammar-constrained JSON or strict tool-schema decoding. Reasoning
is model-controlled; display options do not disable the model's reasoning.
LoRA, speculative decoding, continuous batching, and quantized KV caches are
not implemented for this family.

The runtime defaults to an 8,192-token working context. Prompt and output share
the configured limit, up to the checkpoint's 262,144-token maximum. Oversized
prompts fail rather than truncate history. Sliding layers retain 512 past tokens;
full-attention cache memory grows with the context.

## Provenance and validation

The managed checkpoints are pinned to these immutable revisions:

| Model ID | Source | Revision |
| --- | --- | --- |
| `text-chat-flint-4bit` | `RecursiveSI/flint-int4` | `3b93f9053b530488405ae9fca51d4900d6ce21f0` |
| `text-chat-flint-bf16` | `RecursiveSI/flint` | `2a247be0c0aec895ccb0ac7ca92229ec6df37288` |

Both checkpoints are published under Apache-2.0. Architecture references include
the BF16 checkpoint's SGLang implementation and
[Recursive's inference engine](https://github.com/recursive-oss/rec/tree/e53663b262af4cb1ca498e596ca8b1a2d79e0800).
The native INT4 loader verifies the vendor's published SHA-256 values before
loading the packed tensors. It preserves the export's INT8 embeddings and INT4
projections without requantization. The BF16 loader validates shard ownership,
complete tensor coverage, shapes, and floating-point types before use.

The local tests compare all logits of a deterministic three-layer synthetic
model with an independent scalar reference. They cover dense and shared experts,
sigmoid routing with selection-only probability bias, learned attention sinks,
RoPE placement, and sliding-cache chunk boundaries. The fixture generator uses
only the Python standard library and is test tooling; inference uses Swift/MLX.

On an M5 Max with 128 GiB unified memory, both released checkpoints pass
native Metal arithmetic, a typed `calculate` tool round trip, and retrieval of
an exact access code from a prompt longer than 1,024 tokens. The BF16 run uses
the original weights; all fifteen shard hashes match the pinned publisher
metadata. INT4 uses the published weights and the production loader's pinned
artifact verification. These runs use production precision defaults.

A five-token probe (`[1, 33, 204, 1024, 42]`) compares all 750,720 logits with an
independent PyTorch 2.10 CPU implementation, first checked against the scalar
fixture. Native full-sequence and chunk sizes 1, 3, and 5 agree with that
reference on all five next-token choices for each checkpoint. Logits differ:

| Checkpoint | Full-sequence maximum absolute difference | RMS difference |
| --- | --- | --- |
| Original BF16 | 3.6158 | 0.6936 |
| Published INT4 | 1.3944 | 0.3199 |

This establishes bounded greedy-choice agreement, not numerical equivalence,
broad model quality, or qualification of the declared context limit.

To repeat the installed functional regression, point it at either checkpoint:

```bash
MERERUN_TEST_MLX_DEVICE=gpu \
MERERUN_FLINT_CHECKPOINT_ROOT=/path/to/flint-checkpoint \
    swift test --filter FlintInstalledQualificationTests
```

Linux preparation includes the CUDA quantized-matrix synchronization and BF16
accumulation corrections. Their patching tests cover source drift, repeat
application, and SwiftPM's read-only checkout files. CUDA execution on the
refreshed dependency pin still requires separate qualification. NVFP4 loading
is not implemented. Publisher performance parity and broad long-context quality
are outside this BF16/INT4 runtime's current validation claims.
