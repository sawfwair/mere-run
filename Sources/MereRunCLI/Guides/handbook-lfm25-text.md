# LFM2.5 text (Liquid AI)

## Purpose

This guide is for mere.run Studio and command-line users.

Write compact instructions and examples for LFM2.5 text.

## Start here

Use system instructions for stable behavior and the user message for the task.
One short input and output example can clarify formatting. Keep domain data
separate from instructions and let the runtime construct the conversation
template.

## Example to adapt

This original example illustrates the workflow. It isn't a recorded model
result.

```text
System: Extract only values present in the supplied text. Use null for
missing fields.
User: Return name, quantity, and due_date for: "Two replacement filters are
needed by Friday."
```

## Controls and variants

Use the defaults for the exact managed model. Variants with different
architectures or weight precision can require different settings. The provider
describes assistant prefill, which starts the response with supplied text. If
the selected local interface supports prefill, you can use it. Don't insert raw
role delimiters into ordinary text.

## Iterate and review

If a small model misses a long requirement list, simplify the task and add one
representative example. For schema-constrained output, use the local
structured-output facility where supported and validate the result. Check
missing-value and adversarial-input cases.

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
mere.run guide --model text-chat-lfm25-a1b-8bit
```

To inspect the available command options, run the following command:

```bash
mere.run text chat --help
```

For your first run, use the selected command's defaults. To compare later runs,
save the prompt or input, model ID, parameters, and output together.

## Covered models

This guide covers the following managed model IDs:

- `text-chat-lfm25-a1b-8bit`
- `text-chat-lfm25-a1b-bf16`
- `text-chat-lfm25-1.2b-bf16`
- `text-chat-lfm25-1.2b-qad-4bit`
- `text-chat-lfm25-2.6b-4bit`
- `text-chat-lfm25-2.6b-qad-4bit`
- `text-chat-lfm25-2.6b-bf16`

## Sources and validation

This original mere.run recipe draws on provider material and local command
documentation. Check the local controls before applying provider examples.

Editorial review date: September 4, 2026.

These recipes have not been validated with model inference. Review generated
results before relying on a recipe.

For model and runtime details, see the following sources:

- [Prompting guide](https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting)
- [Text runtime documentation](https://github.com/sawfwair/mere-run/blob/main/docs/runtime/text.md)

Source links require a network connection. The complete recipe and examples are
bundled for offline reading.
