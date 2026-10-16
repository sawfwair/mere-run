# Nemotron 3.5 Bolt (NVIDIA)

## Purpose

This guide is for mere.run Studio and command-line users running local text
and image chat with `vision-chat-nemotron-35-bolt-24gb`.

## Start here

Install the original packed checkpoint explicitly before inference:

```bash
mere.run model pull vision-chat-nemotron-35-bolt-24gb
mere.run text chat --model vision-chat-nemotron-35-bolt-24gb --prompt "Explain a Bloom filter."
```

The native Swift/MLX runtime loads the safetensors directly. You don't need
Python, vLLM, or an offline conversion. The 24.1 GB of weights need additional
memory for recurrent state, image encoding, attention caches, and execution
buffers. Shared admission checks still apply.

## Example to adapt

This prompt illustrates an image workflow. It isn't a recorded model result.

```bash
mere.run text chat --model vision-chat-nemotron-35-bolt-24gb \
  --image ./diagram.png \
  --prompt "List the visible components and their connections. Mark any text you cannot read."
```

Review the response against the source image. Distinguish visible labels from
inferred relationships.

## Controls and variants

The managed entry selects the 74B-A9B-24GB checkpoint. It combines packed
IQ2_XXS experts, Q8_0 projections, NVFP4 Mamba projections, and BF16 vision
weights. Existing Lightning and Omni model IDs retain their separate runtimes.

Text and image inputs, streamed generation, tools, thinking controls, stop
sequences, and log probabilities use the shared chat path. Audio, video input,
LoRA, constrained JSON, and LiLiCorr speculative decoding are unsupported.

## Read this guide offline

Reading the guide requires neither weights nor network access:

```bash
mere.run guide --model vision-chat-nemotron-35-bolt-24gb
mere.run text chat --help
```

Download the checkpoint before running inference offline.

## Sources and validation

The upstream checkpoint is pinned to
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6`.
Review the [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
and [OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/).

Local structural and arithmetic tests do not establish released-checkpoint
quality, throughput, peak process memory, or long-context behavior. Those
measurements require a complete installed checkpoint and separate evidence.
