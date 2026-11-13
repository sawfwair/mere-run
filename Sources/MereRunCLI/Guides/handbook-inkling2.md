# Inkling 2 Small

## Example to adapt

Install `vision-chat-inkling-2-small`, then start with a bounded prompt:

```bash
mere.run model pull vision-chat-inkling-2-small
mere.run text chat --model vision-chat-inkling-2-small \
  --prompt "Explain the data flow in this diagram." --image ./diagram.png \
  --reasoning-effort 0.2 --max-tokens 512
```

The original checkpoint occupies about 394 GB of disk space. The native
reference path streams selected experts from disk. Disk speed limits
throughput; this example has not been validated with full-checkpoint inference.

## Controls and variants

`--reasoning-effort` accepts zero through 0.99 and defaults to 0.9. Chat defaults
to an 8,192-token context shared by prompt and output. `--context-size` can
increase the limit, subject to memory for global attention. Oversized prompts
fail without truncation. Use local image files or base64 image data.

The runtime supports text, images, reasoning, and validated tool calls. LoRA,
constrained JSON, quantized attention caches, audio, and video are unavailable.
Generated output is buffered until channel parsing finishes. Local original
MXFP8 snapshots also work through a checkpoint directory. Converted checkpoint
layouts require a separate loader.

After installation, serve the local API with:

```bash
mere.run api serve --engine text-chat-inkling2 \
  --model vision-chat-inkling-2-small
```

## Sources and validation

The [published checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Small)
is pinned to `aac974122acd33ddb0d3e1f54433a9c5dd2249ce` under Apache-2.0.
Native tests compare decoder and vision computation with independent NumPy
fixtures and verify all 1,770 checkpoint tensor headers. Expert activations
use BF16 rather than the upstream low-precision activation kernels. MTP,
full-checkpoint quality, memory fit, and sustained throughput remain unvalidated.
The runtime guide is `docs/runtime/inkling2.md` in the repository.
