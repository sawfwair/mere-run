# Recursive Flint

## Purpose

This guide is for mere.run Studio and command-line users running Flint locally.
Use the native Swift/MLX runtime for text, reasoning, and tool conversations.

## Start here

Install `text-chat-flint-4bit`, then give the model a bounded coding task. Include
the relevant source and the behavior that you expect. The published INT4 export
uses approximately 22.6 GB of disk space. The catalog memory floor is 48 GB;
this remains a conservative admission estimate. Both formats pass bounded
released-checkpoint checks on an M5 Max with 128 GiB unified memory.

## Example to adapt

This prompt is a workflow example, not a recorded inference result:

```text
Write a Swift function that merges overlapping closed integer intervals.
State how it handles empty input and adjacent intervals. Return the function
and tests for unsorted input, containment, and disjoint intervals.
```

## Controls and variants

Use temperature 1.0, top-p 0.95, and top-k 20 for the publisher's sampling recipe.
Allow enough output tokens for reasoning and the answer. The default working
context is 8,192 tokens; prompt and output share the context budget.

The managed model IDs are `text-chat-flint-4bit` and `text-chat-flint-bf16`.
BF16 weights occupy approximately 79.6 GB and require a catalog memory floor of
128 GB. These floors are estimates. Local paths must contain a supported Flint
safetensors snapshot. NVFP4 and GGUF require different runtimes.

Reasoning is controlled by the model. Display controls do not disable reasoning.
The runtime supports tool history but does not constrain tool arguments with a
grammar. LoRA and structured JSON output are unsupported.

## Iterate and review

Run generated code against meaningful tests. For tool use, supply a narrow tool
allowlist and review proposed actions. Preserve reasoning returned by the model
when continuing an assistant turn. Save the prompt, model ID, parameters, and
output together when comparing runs.

## Read this guide offline

Reading this guide requires neither model weights nor a network connection.
Install the checkpoint before running inference offline.

```bash
mere.run guide --model text-chat-flint-4bit
mere.run model pull text-chat-flint-4bit
mere.run text chat --model text-chat-flint-4bit --prompt "Explain Swift actors." \
    --temperature 1 --top-p 0.95 --top-k 20
```

## Covered models

- `text-chat-flint-4bit`
- `text-chat-flint-bf16`

## Sources and validation

[Flint model card](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288)
and [native runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/flint.md)
record the architecture, source revisions, and runtime limits. Source links
require a network connection. The recipe itself is available offline.

Released BF16 and INT4 checks cover arithmetic, a typed tool round trip, and
retrieval from a prompt longer than 1,024 tokens. A five-token numerical probe
agrees with an independent reference on next-token choices across the tested
prefill chunk sizes; logits are not bit-identical. Broader quality, long-context
limits, minimum-memory fit, and throughput remain unqualified.
