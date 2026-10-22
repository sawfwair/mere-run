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

The [2026-10-21 Spark validation](../qualification/flint-spark-2026-10-21.md)
ran the released INT4 checkpoint through the native Swift/MLX CUDA runtime.
It covered bounded numerical comparisons, cached decoding, CLI generation,
API requests, and a tool-call round trip. The initial 4.3-token/s sample used
dense fallbacks because the source-build environment omitted CUDA JIT headers.
The [corrected performance investigation](../qualification/flint-spark-performance-2026-10-21.md)
measured 27.71 and 28.73 tokens/s with native quantized kernels and an optimized
bridge. These are short diagnostics; performance parity with the publisher's
NVFP4 engine remains unqualified.

The [CUDA kernel investigation](../qualification/flint-spark-kernels-2026-10-21.md)
records a quantized matrix shared-memory race, BF16 accumulation loss, and the
native preparation fixes. It also covers fixed-capacity caches, asynchronous
decode submission, and the remaining 36 GB Mac qualification boundary.

### CUDA performance on Spark

The kernel and decode investigation measures approximately 55 tokens/s with
standard CUDA graph limits and 74 tokens/s with the tested graph profile on GB10.
The measured build uses CUDA 13, architecture 121, and release builds of both
Swift and the native libraries. For a source build, set
`MERERUN_CUDA_ARCHITECTURES=121` during native preparation and follow the
[Linux setup instructions](../linux-quickstart.md) for exports and compilation.

For a dedicated Flint CUDA process, apply the profile before starting the CLI:

```bash
MERERUN_MLX_CUDA_NATIVE_QUANT=native MLX_ENABLE_TF32=0 \
MLX_MAX_OPS_PER_BUFFER=256 MLX_MAX_MB_PER_BUFFER=2048 \
mere.run api serve --engine text-chat-flint --model text-chat-flint-4bit
```

These settings control graph grouping for that process. They retain the catalog
memory floor. The measurements use bounded 128- and 1,024-token requests and do
not establish publisher performance parity or long-context throughput.

Released-weight numerical equivalence, broad tool-use quality, long-context
accuracy, and Metal execution remain unqualified. Keep the catalog's memory
estimates: the Spark memory measurement used a short prompt and does not cover
worst-case working memory. No publisher benchmark is a measurement of this
implementation.
