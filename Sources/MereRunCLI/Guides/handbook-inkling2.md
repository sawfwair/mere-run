# Inkling 2 Mini and Small

## Example to adapt

Install `vision-chat-inkling-2-mini` and try an image prompt.

```bash
mere.run model pull vision-chat-inkling-2-mini
mere.run text chat --model vision-chat-inkling-2-mini \
  --prompt "Explain the data flow in this diagram." --image ./diagram.png \
  --reasoning-effort 0.2 --max-tokens 512
```

Mini Speed is the default and occupies about 18.4 GB. One download includes
the quantized attention and output projections, the vision weights, and the
DSpark draft assets. On macOS with enough unified memory, Mini runs packed
experts on Metal. Small (`vision-chat-inkling-2-small`) occupies about 394 GB
and streams selected experts from disk.

## Controls and variants

Use `vision-chat-inkling-2-mini-q8-head` to install Mini with a prepacked 8-bit
output head. Its checkpoint bundles the original expert and vision weights and
the official DSpark draft. Use `vision-chat-inkling-2-mini-official` for the
original checkpoint. Both variants support the same commands as Mini Speed.

`--reasoning-effort` accepts zero through 0.99 and defaults to 0.9. Chat defaults
to an 8,192-token context shared by prompt and output. `--context-size` can
increase the limit, subject to memory for global attention. Oversized prompts
fail without truncation. Use local image files or base64 image data.

Mini defaults to temperature `1` with unrestricted sampling. With resident
Metal experts and the bundled draft weights, eligible sampled requests use
DSpark. Mini Speed uses three proposals per round. Use `--stats` to see draft acceptance and
`MERERUN_INKLING2_DSPARK=0` to disable drafting.

Train a Mini output-head adapter with `text train-lora --model vision-chat-inkling-2-mini`
and load it in chat with `--lora`. Adapter requests use autoregressive decoding.

The runtime supports text, images, reasoning, and validated tool calls. Constrained JSON, quantized attention caches, audio, and video are unavailable.
Generated output is buffered until channel parsing finishes. Local original
MXFP8 snapshots also work through a checkpoint directory.

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
