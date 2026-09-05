# Qwen3 text embeddings (Qwen)

## Purpose

This guide is for mere.run Studio and command-line users.

Prepare retrieval queries and documents for Qwen3 embeddings, which are numeric
representations used to compare text.

## Start here

State the retrieval task in the query instruction, then provide the user's
query. Index document text in a consistent format. Keep instructions out of
ordinary document content unless the selected embedding interface requires them.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
Task instruction: Retrieve maintenance notes that answer the question.
Query: Which pump needed a seal replacement?
Document: Pump B was removed from service after a leaking seal was found.
```

## Controls and variants

The provider distinguishes query instructions from document text. Check the
local `text embed` command's instruction handling before manually adding
prefixes, so that instructions aren't applied twice. Keep normalization and
similarity computation consistent across the index.

## Iterate and review

If retrieval returns topical but unhelpful passages, improve chunk boundaries
and the task instruction. Compare positive and negative examples. Similarity is
a ranking signal, not evidence that a retrieved statement is true.

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
mere.run guide --model text-embed-qwen3-0.6b
```

To inspect the available command options, run the following command:

```bash
mere.run text embed --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the `text-embed-qwen3-0.6b` model.

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Embedding usage](https://github.com/QwenLM/Qwen3-Embedding)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
