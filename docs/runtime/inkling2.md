# Native Inkling 2

`mere.run` runs text and image chat with Inkling 2 Mini and Small through
Swift/MLX. The native runtime reads the original mixed NVFP4/MXFP8
checkpoints and supports reasoning and tool calls.

| Model | Managed ID | Parameters | Checkpoint disk space |
| --- | --- | --- | --- |
| Mini | `vision-chat-inkling-2-mini` | 26.4B total, 5.4B effective active per token | About 20 GB |
| Small | `vision-chat-inkling-2-small` | 594B total, 17B active per token | About 394 GB |

On macOS, Mini keeps its packed experts resident on Metal when the checkpoint
fits within three quarters of unified memory. Small uses a reference path
that streams selected experts from disk. Mini also uses that path on other
devices, on smaller memory configurations, or when launched with
`MERERUN_INKLING2_PACKED_EXPERTS=0`.

## Install and chat

Installation is explicit; chat doesn't download the checkpoint automatically:

```bash
mere.run model pull vision-chat-inkling-2-mini
mere.run text chat --model vision-chat-inkling-2-mini \
  --prompt "Explain block attention residuals." \
  --reasoning-effort 0.2 --max-tokens 512
```

To describe an image, add `--image ./diagram.png`. To use an existing original
snapshot, pass its directory to `--model`, or use the matching managed model ID
with `--model-root /path/to/checkpoint`.

The runtime reads the checkpoint's SafeTensors index and recognizes
`model_type` values `inkling2_loop` for fixed-pass Mini and `inkling2` for
Small, including the original Small MXFP8 checkpoint.

### Mini with a prepacked Q8 output head

The managed `vision-chat-inkling-2-mini-q8-head` checkpoint keeps the original
expert and vision weights and adds an MLX affine 8-bit output head. Pull it
from the public Sawfwair repository. [Inkling 2 Mini Q8 Head checkpoint](https://huggingface.co/Sawfwair/Inkling-2-Mini-Q8-Head)

```bash
mere.run model pull vision-chat-inkling-2-mini-q8-head
mere.run text chat --model vision-chat-inkling-2-mini-q8-head \
  --prompt "Explain why the Moon changes phases." --stats
```

The same snapshot contains the tokenizer, image processor, chat template,
and DSpark weights.

The target and draft use the same packed output head. Precomputed DSpark
projections let the draft load directly from the snapshot. The original Mini
checkpoint continues to use its BF16 output head for target verification.

## Speculative decoding on Mini

Mini defaults to temperature `1` with unrestricted sampling. For text and
image chat on Metal, Mini uses adaptive DSpark by default when the packed
experts are resident, the draft weights are installed, and the request uses
unrestricted sampling. With the original checkpoint, DSpark uses 8-bit
copies of the draft vocabulary projections and the target verifies with its
BF16 head.

```bash
mere.run text chat --model vision-chat-inkling-2-mini \
  --prompt "Explain why the Moon looks different throughout the month." \
  --max-tokens 256 --stats
```

Set `MERERUN_INKLING2_DSPARK=0` to disable drafting. Set it to `1` to
enable drafting for eligible requests with greedy or filtered sampling,
or when Mini uses the reference path.

Mini starts with three proposals per round and adjusts the depth using the
accepted prefixes from recent rounds. With resident Metal experts, adaptive
DSpark can also propose up to 16 tokens at a time from repeated sequences in
the generated text. The target verifies these proposals using the request's
sampling settings. The `inkling_dspark` stats line reports
proposed and accepted tokens and how many rounds ran at each depth. To keep
the depth fixed, set `MERERUN_INKLING2_DSPARK_TOKENS` to a value from 1 to 7.
DSpark requires default penalties and requests without custom stop sequences
or logprob capture. Batched verification can change greedy tokens because
floating-point rounding differs from single-token decode. Sampled runs can
produce different output with the same seed because speculation consumes a
different sequence of random draws.

Mini's managed download includes `mtp.safetensors`. Older checkpoints without
this file use standard decoding. To enable drafting for an older checkpoint,
add the draft weights to its directory. [Official Mini draft weights](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/ed40e7da625f594fc9c4990f5f66be24873c22d7/mtp.safetensors)

## Serve the local API

```bash
mere.run api serve --engine text-chat-inkling2 \
  --model vision-chat-inkling-2-mini
```

The API accepts text, local image files, base64 image data, tool declarations,
stop sequences, seeds, and sampling penalties. The runtime doesn't fetch remote
image URLs. Channel parsing separates reasoning, visible text, and validated
tool calls. Output is buffered until channel parsing finishes.

## Use Mini with Pi

On an Apple Silicon Mac with at least 32 GB of unified memory, start Pi with
the managed Mini checkpoint.

```bash
mere.run agent start --model vision-chat-inkling-2-mini
```

The launcher starts the local API server and configures Pi for text, images,
and tool calls. Pass `--no-bootstrap` to use an existing checkpoint and Pi
installation.

## Runtime behavior

Chat uses an 8,192-token context by default. Prompt and output share the limit;
oversized prompts fail without truncating history. You can set
`--context-size` up to the checkpoint's 1,048,576-token architectural limit.
Global attention cache and attention workspace grow with context. The
catalog lists an estimated 32 GB minimum for both checkpoints and recommends
64 GB for Small. Full-checkpoint memory fit remains unmeasured.

`--reasoning-effort` accepts values from zero through 0.99. The default is 0.9.
The released template supplies the effort message and intentionally appends
no assistant generation prefix. Display controls determine whether reasoning
appears in terminal output.

Images keep their original dimensions. The processor composites transparency
over an eight-pixel checkerboard, pads to 40-pixel token boundaries, and
normalizes pixels. The vision tower encodes 20-pixel native patches and merges
each two-by-two group into one language token.

The decoder uses BF16 attention caches, learned sinks, clipped query and key
heads, relative-position bias, static and dynamic short convolutions, and
attention over residual blocks. Experts use corrected sigmoid routing, eight
selected experts, interleaved gate/up weights, and separate activation softcaps.
Mini runs the middle 12 layers three times with shared weights. Each pass
keeps its own attention and short convolution state. Learned controllers
inject messages from the previous pass into the query, key, value, and expert
inputs. On Metal, Mini batches the query, key, and value controller
projections and fuses controller gating with input normalization. Small runs
its 48 layers once.

The Metal path gathers selected NVFP4/MXFP8 experts directly from their packed
weights. It submits the decoder and its cache updates together and fuses short
convolution taps into one kernel. It also combines projections that read the
same query input. Single-token decode fuses query, key, value, and relative
head normalization, and uses fused kernels for attention over up to 1,024
cached keys per layer and for depth residuals. The Metal path also fuses
top-eight expert routing and combines scaling with the activation and
weighted reduction around the expert matmuls. The reference path dequantizes
one selected expert at a time.
Both use BF16 activations and FP32 routing and accumulation.
Rounding can differ between the two paths and change sampled tokens.
LoRA, constrained JSON output, quantized attention caches, audio, and video
are unavailable.

## Validation and provenance

`Inkling2RuntimeTests` compares the decoder and vision tower against independent
FP64 NumPy fixtures. The decoder checks one-shot, token-by-token, and chunked
execution across convolution, residual-block, and sliding-window boundaries.
Tests also cover Mini's cross-pass messages, packed formats, image padding
and transparency, indexed expert reads, and the 1,244 Mini and 1,770 Small
checkpoint tensor headers.

Metal tests compare packed gathers, short convolution, head normalization,
grouped decode attention, depth residuals, expert routing, and expert
activation and reduction against the reference arithmetic.
Core and CLI tests cover model identification, native serving selection,
the released chat template, tool-call filtering, and local paths with mixed
case. Short text and image prompts have been smoke-tested with the original
Mini checkpoint on an Apple M5 Max.

The [model source list](../model-sources.md#inkling-2) records the pinned
Apache-2.0 checkpoints. The architecture follows the Transformers
[Small implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2)
and the [Mini implementation](https://github.com/huggingface/transformers/pull/51050).
