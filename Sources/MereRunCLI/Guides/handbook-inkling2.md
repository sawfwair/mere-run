# Inkling 2 Mini and Small

## Example to adapt

Install `vision-chat-inkling-2-mini` and try an image prompt.

```bash
mere.run model pull vision-chat-inkling-2-mini
mere.run text chat --model vision-chat-inkling-2-mini \
  --prompt "Explain the data flow in this diagram." --image ./diagram.png \
  --reasoning-effort 0.2 --max-tokens 512
```

Mini's original checkpoint occupies about 20 GB. Small
(`vision-chat-inkling-2-small`) occupies about 394 GB. On macOS with enough
unified memory, Mini retains the original packed weights and uses Metal
matmuls for routed experts. Small streams selected experts from disk.

## Controls and variants

`--reasoning-effort` accepts zero through 0.99 and defaults to 0.9. Chat defaults
to an 8,192-token context shared by prompt and output. `--context-size` can
increase the limit, subject to memory for global attention. Oversized prompts
fail without truncation. Use local image files or base64 image data.

Mini defaults to temperature `1` with unrestricted sampling. With resident
Metal experts and the bundled draft weights, eligible sampled requests use
adaptive DSpark. Use `--stats` to see draft acceptance and
`MERERUN_INKLING2_DSPARK=0` to disable drafting.

The runtime supports text, images, reasoning, and validated tool calls. LoRA,
constrained JSON, quantized attention caches, audio, and video are unavailable.
Generated output is buffered until channel parsing finishes. Local original
MXFP8 snapshots also work through a checkpoint directory. Converted checkpoint
layouts require a separate loader.

After installation, serve the local API with:

```bash
mere.run api serve --engine text-chat-inkling2 \
  --model vision-chat-inkling-2-mini
```

## Sources and validation

Native tests compare decoder and vision computation with independent NumPy
fixtures and verify the published Mini and Small checkpoint tensor headers.
Expert activations use BF16. Short text and image prompts have been tested
with the original Mini checkpoint on an Apple M5 Max. See
`docs/runtime/inkling2.md` for the pinned Apache-2.0 checkpoints and runtime
details.
