# Qwen3 VL embeddings (Qwen)

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare image and text retrieval inputs for Qwen3 VL embeddings. An embedding is
a numeric representation used to compare inputs.

## Start here

Describe the visible concept you want to retrieve and keep image preprocessing
consistent with the indexed collection. Use task instructions supported by the
local embedding interface rather than a conversational request for an answer.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Retrieval query: A white utility vehicle parked beside an orange traffic
cone.
Positive example: a clear side view showing both objects.
Negative example: a white vehicle with no cone visible.
```

## Controls and variants

Use the same model and normalization for query and indexed embeddings. Keep the
distinction between text queries, image inputs, and combined inputs explicit.
Compare retrieval on examples from the actual collection.

## Iterate and review

If results rely on color alone, add the relevant object relationship. Check
crops and backgrounds for shortcuts. The local guide documents L2-normalized
embeddings, which have unit vector length. Embedding similarity can't establish
identity, ownership, or causation.

## Read this guide offline

Reading the handbook requires neither model weights nor a network connection.
Before running inference offline, download the checkpoint and any optional
components that your workflow uses.

To read the guide in macOS Studio, follow these steps:

1. In **Help**, select **mere.run Guide**.
2. In **Guide collection**, select **Models**.
3. In the search field, enter a family name or model ID.
4. Select the guide.
5. In **Model**, select a variant.

Prompt examples include a **Copy** control. Input examples serve as
file-preparation checklists. Reading a guide doesn't start inference.

To read this guide in a terminal, run the following command:

```bash
mere.run guide --model vision-embed-qwen3-vl-2b
```

To inspect the available command options, run the following command:

```bash
mere.run vision embed --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `vision-embed-qwen3-vl-2b` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [VL embedding usage](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
