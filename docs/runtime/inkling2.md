# Native Inkling 2 Small

`mere.run` runs Inkling 2 Small text and image chat through Swift/MLX. The
native reference path reads the original checkpoint without Python, a remote
inference service, or a conversion step. It supports reasoning and tool calls.

The checkpoint has 594B parameters, with 17B active per token. Its 24 weight
shards occupy about 394 GB. Selected experts stream from disk, and ordinary
weights stay in memory. Disk throughput limits generation speed. This path has
no full-checkpoint throughput or answer-quality qualification.

## Install and chat

Installation is explicit; chat doesn't download the checkpoint automatically:

```bash
mere.run model pull vision-chat-inkling-2-small
mere.run text chat --model vision-chat-inkling-2-small \
  --prompt "Explain block attention residuals." \
  --reasoning-effort 0.2 --max-tokens 512
```

To describe an image, add `--image ./diagram.png`. To use an existing original
snapshot, pass its directory to `--model`, or use the managed model ID with
`--model-root /path/to/Inkling-2-Small`.

The runtime also recognizes local original MXFP8 checkpoints whose
`config.json` sets `model_type` to `inkling2`. It requires the original tensor
names and index. Converted Transformers or MLX checkpoint layouts are separate
formats.

## Serve the local API

```bash
mere.run api serve --engine text-chat-inkling2 \
  --model vision-chat-inkling-2-small
```

The API accepts text, local image files, base64 image data, tool declarations,
stop sequences, seeds, and sampling penalties. The runtime doesn't fetch remote
image URLs. Channel parsing separates reasoning, visible text, and validated
tool calls. Output is buffered until channel parsing finishes.

## Runtime behavior

Chat uses an 8,192-token context by default. Prompt and output share the limit;
oversized prompts fail without truncating history. You can set
`--context-size` up to the checkpoint's 1,048,576-token architectural limit.
Global attention cache and attention workspace grow with context. The
catalog's 32 GB minimum and 64 GB recommendation are admission estimates,
not measured memory qualifications.

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
The main checkpoint stores expert layers zero through 39 in NVFP4 and layers
40 through 47 in MXFP8.

Expert weights decode one selected expert at a time. Activations use BF16,
with FP32 routing and expert-output accumulation. This reference path does
not emulate the upstream NVFP4/MXFP8 activation-quantization kernels. It doesn't
use the auxiliary `mtp.safetensors` predictor. LoRA, constrained JSON output,
quantized attention caches, audio, and video are unavailable.

## Validation and provenance

`Inkling2RuntimeTests` compares the decoder and vision tower against independent
FP64 NumPy fixtures. The decoder checks one-shot, token-by-token, and chunked
execution across convolution, residual-block, and sliding-window boundaries.
Tests also cover packed formats, image padding and transparency, indexed
expert reads, and all 1,770 published checkpoint tensor headers.

Core and CLI tests cover model identification, native serving selection,
the released chat template, tool-call filtering, and local paths with mixed
case. Real-checkpoint generation, quality, GPU behavior, and sustained
performance remain unvalidated.

The managed source is
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
at revision `aac974122acd33ddb0d3e1f54433a9c5dd2249ce`, under Apache-2.0.
The architecture follows the official
[Transformers Inkling 2 implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2).
